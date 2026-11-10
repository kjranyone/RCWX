"""M-T4 検証: 3-step 学習パリティ golden 生成。

tiny 設定の G (enc_p + emb_g + enc_q + flow + dec) と MPD (disc_s + disc_p ×2) を
PyTorch で構築し、Rust Trainer::step と同一の手順で 3 step 実行して
各 step の損失 (d/adv/fm/mel/kl) と初期・最終パラメータを保存する。

Rust 側 (crates/rcwx-train/tests/step_parity.rs) は同じ初期重み・入力で
3 step 回し、損失 rtol 1e-3・param delta cos ≥ 0.999 を検証する。

対応の要点 (Rust 実装に合わせてある):
- enc_q の z は golden randn を注入 (内部 torch.randn 不使用)
- dec.m_source は固定 har (golden) に差し替え
- D step → D 更新 → **旧 D 重みでの再 forward** (Rust は step 内で WN refold しないため)
- disc_p は末尾 reflect pad (upstream F.pad(y, (0, n_pad), "reflect"))
- 正規化は weight_norm (spectral_norm ではない — Rust ポートと同一)
- KL は 0.5 因子なし: (logs_p − logs_q − 0.5 + 0.5·(z_p−m_p)²·exp(−2·logs_p)) の mask 平均
- mel: reflect pad (2048−480)/2 両側 → hann periodic → |STFT| sqrt(+2e-7) →
  slaney mel → log clamp 2e-6 (Rust stft_mel と同一)

使い方: uv run python tools/train_step_ref.py
出力: ~/.cache/rcwx/train/golden/step3_golden.safetensors
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rcwx.models.infer_pack import attentions, commons, models as im  # noqa: E402

B, T = 2, 2310
IN, H, INTER, FILT, HEADS, LAYERS, K, WS = 8, 8, 6, 16, 2, 1, 3, 3
SPEC_BINS, GIN, UPP = 16, 4, 8
# SEG_T·UPP = 18400 samples — disc_s (ks 15/41×4/41/5, stride 1/4×4/1) が
# 6 段全て通過可能なサイズ (実 48k 学習の segment 17280 相当)
SEG_T = 2300
START = 5
SR = 48000
PERIODS = [2, 3]
N_STEPS = 3
SEED = 777


class EncP(torch.nn.Module):
    """Rust enc_p と同一構成 (train_ref.py のものから debug を省いた形)。"""

    def __init__(self):
        super().__init__()
        self.emb_phone = torch.nn.Linear(IN, H)
        self.emb_pitch = torch.nn.Embedding(256, H)
        self.encoder = attentions.Encoder(H, FILT, HEADS, LAYERS, K, 0.0, window_size=WS)
        self.proj = torch.nn.Conv1d(H, INTER * 2, 1)

    def forward(self, phone, pitch, lengths):
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(H)
        x = F.leaky_relu(x, 0.1)
        x = x.transpose(1, 2)
        x_mask = torch.unsqueeze(
            commons.sequence_mask(lengths, x.size(2)), 1
        ).to(x.dtype)
        x = x * x_mask
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, INTER, dim=1)
        return m_p, logs_p, x_mask


class G(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_p = EncP()
        self.emb_g = torch.nn.Embedding(1, GIN)
        self.enc_q = im.PosteriorEncoder(
            SPEC_BINS, INTER, H, kernel_size=5, dilation_rate=1, n_layers=2,
            gin_channels=GIN,
        )
        self.flow = im.ResidualCouplingBlock(
            INTER, H, 5, 1, 2, n_flows=2, gin_channels=GIN,
        )
        self.dec = im.GeneratorNSF(
            initial_channel=INTER,
            resblock="1",
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3], [1, 3]],
            upsample_rates=[4, 2],
            upsample_initial_channel=16,
            upsample_kernel_sizes=[8, 4],
            gin_channels=GIN,
            sr=SR,
        )


class DiscS(torch.nn.Module):
    """Rust disc_s と同一 (v2 チャネル構成・lrelu 0.1 固定)。"""

    def __init__(self):
        super().__init__()
        chans = [(1, 16), (16, 64), (64, 256), (256, 1024), (1024, 1024), (1024, 1024)]
        ks = [15, 41, 41, 41, 41, 5]
        strides = [1, 4, 4, 4, 4, 1]
        groups = [1, 4, 16, 64, 256, 1]

        def wn_conv(cin, cout, k, s, g):
            c = torch.nn.Conv1d(cin, cout, k, stride=s, padding=0, groups=g)
            return torch.nn.utils.weight_norm(c)

        self.convs = torch.nn.ModuleList(
            [
                wn_conv(chans[i][0], chans[i][1], ks[i], strides[i], groups[i])
                for i in range(len(chans))
            ]
        )
        self.conv_post = wn_conv(1024, 1, 3, 1, 1)

    def forward(self, x):
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        return x, fmap


class DiscP(torch.nn.Module):
    """Rust disc_p と同一 (末尾 reflect pad・5×5 conv stride (3,1) pad (2,2))。"""

    def __init__(self, period):
        super().__init__()
        self.period = period
        chans = [32, 128, 512, 1024, 1024]

        def wn_conv(cin, cout):
            c = torch.nn.Conv2d(cin, cout, (5, 1), (3, 1), padding=(2, 0))
            return torch.nn.utils.weight_norm(c)

        self.convs = torch.nn.ModuleList([wn_conv(1 if i == 0 else chans[i - 1], chans[i])
                                          for i in range(len(chans))])
        self.conv_post = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(chans[-1], 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, y):
        fmap = []
        b, c, t = y.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            y = F.pad(y, (0, n_pad), "reflect")
        y = y.view(b, c, -1, self.period)
        for conv in self.convs:
            y = F.leaky_relu(conv(y), 0.1)
            fmap.append(y)
        y = self.conv_post(y)
        return y, fmap


class MPD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_s = DiscS()
        self.disc_p = torch.nn.ModuleList([DiscP(p) for p in PERIODS])

    def forward(self, x):
        logits, fmaps = [], []
        l, f = self.disc_s(x)
        logits.append(l)
        fmaps.extend(f)
        for d in self.disc_p:
            l, f = d(x)
            logits.append(l)
            fmaps.extend(f)
        return logits, fmaps


# ---- slaney mel 基底 (Rust stft_mel.rs と同一式) ----

def hz_to_mel(f):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = math.log(6.4) / 27.0
    if f < min_log_hz:
        return f / f_sp
    return min_log_mel + math.log(f / min_log_hz) / logstep


def mel_to_hz(m):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = math.log(6.4) / 27.0
    if m < min_log_mel:
        return m * f_sp
    return min_log_hz * math.exp((m - min_log_mel) * logstep)


def slaney_mel(n_mels, bins, sr, n_fft):
    fmin, fmax = 0.0, sr / 2.0
    min_mel, max_mel = hz_to_mel(fmin), hz_to_mel(fmax)
    centers_mel = [
        min_mel + (max_mel - min_mel) * i / (n_mels + 1) for i in range(n_mels + 2)
    ]
    centers_hz = [mel_to_hz(m) for m in centers_mel]
    fft_freqs = [k * sr / n_fft for k in range(bins)]
    out = np.zeros((n_mels, bins), dtype=np.float32)
    for m in range(n_mels):
        left, mid, right = centers_hz[m], centers_hz[m + 1], centers_hz[m + 2]
        for k, f in enumerate(fft_freqs):
            if f <= left or f >= right:
                v = 0.0
            elif f <= mid:
                v = (f - left) / (mid - left)
            else:
                v = (right - f) / (right - mid)
            out[m, k] = v * 2.0 / (right - left)
    return out


N_FFT, HOP, N_MELS = 2048, 480, 128
MEL_BASIS = None
HANN = None


def logmel(y):
    """y [B, samples] → log-mel [B, 128, nf]。Rust StftMel::logmel と同一。"""
    global MEL_BASIS, HANN
    if MEL_BASIS is None:
        MEL_BASIS = torch.from_numpy(
            slaney_mel(N_MELS, N_FFT // 2 + 1, SR, N_FFT)
        ).float()
        HANN = torch.hann_window(N_FFT).float()
    pad = (N_FFT - HOP) // 2
    yy = F.pad(y, (pad, pad), mode="reflect")
    spec = torch.stft(
        yy, N_FFT, hop_length=HOP, win_length=N_FFT, window=HANN,
        center=False, return_complex=True,
    )  # [B, bins, nf]
    mag = torch.sqrt(spec.real**2 + spec.imag**2 + 2e-7)
    mel = torch.matmul(MEL_BASIS, mag)
    return torch.log(torch.clamp(mel, min=2e-6))


def mse_mean(x, c):
    return torch.mean((x - c) ** 2)


def main() -> None:
    torch.manual_seed(SEED)
    g_model = G()
    mpd = MPD()

    gen = torch.Generator().manual_seed(SEED + 1)
    # ---- 入力バッチ (固定・全 step 同一) ----
    phone = (torch.randn(B, T, IN, generator=gen) * 0.5).float()
    pitch = torch.randint(60, 200, (B, T), generator=gen)
    spec = (torch.randn(B, SPEC_BINS, T, generator=gen) * 0.5).float()
    randn = torch.randn(B, INTER, T, generator=gen)
    lengths = torch.tensor([T, T])
    # har: SineGen (training mode) を固定シードで 1 回だけ実行した出力を使用。
    # Rust は StepInputs.har として外部から受け取る (train_ref2.py と同一方式)
    f0 = torch.empty(B, T).uniform_(120.0, 400.0, generator=gen)
    torch.manual_seed(SEED + 2)
    with torch.no_grad():
        har, _, _ = g_model.dec.m_source(f0[:, :SEG_T], UPP)  # [B, seg*upp, 1]
    har_t = har.transpose(1, 2)  # [B, 1, seg*upp]
    seg_samples = SEG_T * UPP
    wave = (0.3 * torch.sin(
        2 * math.pi * 220 * torch.arange(seg_samples).float() / SR
    )).unsqueeze(0).expand(B, -1).contiguous() + 0.01 * torch.randn(
        B, seg_samples, generator=gen
    )

    # dec.m_source を固定 har へ差し替え
    class FixedSource(torch.nn.Module):
        def forward(self, x, upp=1):
            return har, None, None

    g_model.dec.m_source = FixedSource()

    g0 = g_model.emb_g(torch.zeros(B, dtype=torch.long))  # [B, GIN]
    gd0_snapshot: dict[str, torch.Tensor] = {}
    gg0_snapshot: dict[str, torch.Tensor] = {}
    ggc_snapshot: dict[str, torch.Tensor] = {}

    # ---- 初期重みでの D 参照 (logits / fmap / 実効重み / real 勾配) ----
    dbg_capture: dict[str, torch.Tensor] = {}
    # ---- 初期重みでの G 段階参照 (z / z_p) ----
    with torch.no_grad():
        _g0 = g_model.emb_g(torch.zeros(B, dtype=torch.long))
        _mp, _lpg, _xm = g_model.enc_p(phone, pitch, lengths)
        _z, _mq, _lq, _ = g_model.enc_q(spec, lengths, g=_g0.unsqueeze(-1))
        _z = (_mq + randn * torch.exp(_lq)) * _xm
        _zp = g_model.flow(_z, _xm, g=_g0.unsqueeze(-1), reverse=False)
        dbg_capture["g_z"] = _z.detach().reshape(-1).float()
        dbg_capture["g_zseg"] = _z[:, :, 5:5+SEG_T].detach().reshape(-1).float()
        dbg_capture["g_zp"] = _zp.detach().reshape(-1).float()
        dbg_capture["g_mq"] = _mq.detach().reshape(-1).float()
        dbg_capture["g_lq"] = _lq.detach().reshape(-1).float()


    _mpd2 = MPD()
    _mpd2.load_state_dict(mpd.state_dict())
    _dp0 = _mpd2.disc_p[0]
    _orig_fwd = _dp0.forward

    def _cap_fwd(y):
        b, c, t = y.shape
        n_pad = _dp0.period - (t % _dp0.period) if t % _dp0.period != 0 else 0
        yy = F.pad(y, (0, n_pad), "reflect") if n_pad else y
        dbg_capture["dp0_pad"] = yy.reshape(-1).clone()
        fmap = []
        x = yy.view(b, c, -1, _dp0.period)
        for ci, conv in enumerate(_dp0.convs):
            x = F.leaky_relu(conv(x), 0.1)
            if ci < 2:
                dbg_capture[f"dp0_fmap{ci}"] = x.detach().reshape(-1).clone()
            fmap.append(x)
        return _dp0.conv_post(x), fmap

    _dp0.forward = _cap_fwd
    _rl, _ = _mpd2(wave.unsqueeze(1))
    _dp0.forward = _orig_fwd
    for i, lg in enumerate(_rl):
        dbg_capture[f"dlog.{i}"] = lg.detach().reshape(-1)
    # disc_p[0].convs.{0,2} の実効重み
    for ci in [0, 2]:
        conv = _dp0.convs[ci]
        v = conv.weight_v.detach()
        g = conv.weight_g.detach()
        nrm = v.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        dbg_capture[f"dp0_w{ci}"] = (v * g / nrm).reshape(-1)
    # real のみ lsgan の勾配 (初期重み)
    for p_ in _mpd2.parameters():
        p_.grad = None
    _rl3, _ = _mpd2(wave.unsqueeze(1))
    _loss_r = sum(mse_mean(t, 1.0) for t in _rl3)
    _loss_r.backward()
    for name, p in _mpd2.named_parameters():
        gd0_snapshot[name] = (
            p.grad.detach().clone().float().reshape(-1) if p.grad is not None
            else torch.zeros_like(p).reshape(-1)
        )


    opt_g = torch.optim.AdamW(
        g_model.parameters(), lr=1e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01
    )
    opt_d = torch.optim.AdamW(
        mpd.parameters(), lr=1e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01
    )

    def flat_sd(m):
        # shape 付きで保存 (Rust 側で元形状を復元する)
        return {k: p.detach().clone().float() for k, p in m.named_parameters()}

    g_init = flat_sd(g_model)
    d_init = flat_sd(mpd)
    losses = []

    for step in range(N_STEPS):
        # ---- G forward ----
        # g0 は step 毎に再計算 (Rust は毎 step emb_g 行 0 から組む)
        g0 = g_model.emb_g(torch.zeros(B, dtype=torch.long))
        m_p, logs_p, x_mask = g_model.enc_p(phone, pitch, lengths)
        z, m_q, logs_q, _ = g_model.enc_q(spec, lengths, g=g0.unsqueeze(-1))
        z = (m_q + randn * torch.exp(logs_q)) * x_mask
        z_p = g_model.flow(z, x_mask, g=g0.unsqueeze(-1), reverse=False)
        z_slice = z[:, :, START : START + SEG_T]
        y_hat = g_model.dec(
            z_slice, f0[:, :SEG_T], g=g0.unsqueeze(-1)
        ).squeeze(1)  # [B, seg]

        # ---- D step ----
        dr_logits, dr_fmaps = mpd(wave.unsqueeze(1))
        df_logits, _ = mpd(y_hat.detach().unsqueeze(1))
        d_loss = sum(mse_mean(r, 1.0) for r in dr_logits) + sum(
            mse_mean(f, 0.0) for f in df_logits
        )
        opt_d.zero_grad()
        # dr_fmaps は G 側でも使うので graph を保持 (Rust tape は 1 graph 多 seed 相当)
        d_loss.backward(retain_graph=True)
        # upstream RVC と同一: D 更新 → 更新後の重みで再 forward (WN fold は更新後に再計算)
        opt_d.step()

        # ---- D 再 forward (更新後) → G losses ----
        dg2_logits, dg2_fmaps = mpd(y_hat.unsqueeze(1))
        g_adv = sum(mse_mean(f, 1.0) for f in dg2_logits)
        g_fm = sum(
            2 * torch.mean(torch.abs(r.detach() - g2))
            for r, g2 in zip(dr_fmaps, dg2_fmaps)
        )
        g_mel = 45.0 * torch.mean(torch.abs(logmel(wave) - logmel(y_hat)))
        d_ = z_p - m_p
        kl_elem = (
            logs_p - logs_q - 0.5 + 0.5 * d_ * d_ * torch.exp(-2 * logs_p)
        ) * x_mask
        n_mask = lengths.sum().item() * INTER
        g_kl = kl_elem.sum() / n_mask
        g_total = g_adv + g_fm + g_mel + g_kl
        if step == 0:
            # コンポーネント単独の G 勾配 (g_total.backward 前に実施)
            for tag, comp in [
                ("mel", g_mel),
                ("gan", g_adv + g_fm),
                ("kl", g_kl),
            ]:
                opt_g.zero_grad()
                comp.backward(retain_graph=True)
                for name, p_ in g_model.named_parameters():
                    ggc_snapshot[f"{tag}.{name}"] = (
                        p_.grad.detach().clone().float().reshape(-1)
                        if p_.grad is not None
                        else torch.zeros_like(p_).reshape(-1)
                    )
        opt_g.zero_grad()
        g_total.backward()
        opt_g.step()

        losses.append(
            (d_loss.item(), g_adv.item(), g_fm.item(), g_mel.item(), g_kl.item())
        )
        if step == 0:
            yhat1_snapshot = y_hat.detach().clone().float().reshape(-1)
            # G 勾配 (g_total.backward 直後・初期重み)
            for name, p_ in g_model.named_parameters():
                gg0_snapshot[name] = (
                    p_.grad.detach().clone().float().reshape(-1) if p_.grad is not None
                    else torch.zeros_like(p_).reshape(-1)
                )
            dbg_capture["g_z_loop"] = z.detach().reshape(-1).float()
            dbg_capture["g_mq_loop"] = m_q.detach().reshape(-1).float()
            g1_snapshot = flat_sd(g_model)
            d1_snapshot = flat_sd(mpd)
        print(
            f"step {step + 1}: d={d_loss.item():.5f} adv={g_adv.item():.5f} "
            f"fm={g_fm.item():.5f} mel={g_mel.item():.5f} kl={g_kl.item():.5f}"
        )

    g_fin = flat_sd(g_model)
    d_fin = flat_sd(mpd)

    out: dict[str, torch.Tensor] = {}
    for k, v in g_init.items():
        out[f"g0.{k}"] = v
    for k, v in d_init.items():
        out[f"d0.{k}"] = v
    for k, v in g1_snapshot.items():
        out[f"g1.{k}"] = v
    for k, v in d1_snapshot.items():
        out[f"d1.{k}"] = v
    for k, v in g_fin.items():
        out[f"g3.{k}"] = v
    for k, v in d_fin.items():
        out[f"d3.{k}"] = v
    out["in.phone"] = phone
    out["in.pitch"] = pitch.float()
    out["in.spec"] = spec
    out["in.randn"] = randn
    out["in.har"] = har_t.reshape(-1)
    out["in.wave"] = wave
    out["dbg.yhat1"] = yhat1_snapshot
    for k, v in gd0_snapshot.items():
        out[f"gd0.{k}"] = v
    for k, v in gg0_snapshot.items():
        out[f"gg0.{k}"] = v
    for k, v in ggc_snapshot.items():
        out[f"ggc.{k}"] = v
    for k, v in dbg_capture.items():
        out[f"dbg.{k}"] = v.float()
    for i, (d, adv, fm, mel, kl) in enumerate(losses):
        out[f"loss.{i}"] = torch.tensor([d, adv, fm, mel, kl])

    out_dir = os.path.join(os.path.expanduser("~"), ".cache", "rcwx", "train", "golden")
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file

    save_file(out, os.path.join(out_dir, "step3_golden.safetensors"))
    print(f"saved → step3_golden.safetensors ({len(out)} tensors)")


if __name__ == "__main__":
    main()
