"""Audit HuBERT weight mapping: fairseq -> transformers.

Checks:
1. Are all fairseq weights mapped to transformers?
2. Are shapes consistent?
3. Are there any NaN/Inf in loaded weights?
4. Parameter count comparison
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from transformers import HubertModel, HubertConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from rcwx.models.hubert_loader import _setup_fake_fairseq, _map_fairseq_to_transformers


def main():
    hubert_path = Path.home() / ".cache" / "rcwx" / "models" / "hubert" / "hubert_base.pt"
    if not hubert_path.exists():
        print(f"SKIP: HuBERT not found at {hubert_path}")
        return

    # Load raw fairseq checkpoint
    _setup_fake_fairseq()
    checkpoint = torch.load(hubert_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        fairseq_state = checkpoint["model"]
    else:
        fairseq_state = checkpoint

    print("=" * 80)
    print("FAIRSEQ CHECKPOINT ANALYSIS")
    print("=" * 80)
    print(f"  Total keys: {len(fairseq_state)}")

    # Categorize fairseq keys
    categories = {}
    for k in fairseq_state:
        prefix = k.split(".")[0]
        categories.setdefault(prefix, []).append(k)
    for cat, keys in sorted(categories.items()):
        total_params = sum(fairseq_state[k].numel() for k in keys)
        print(f"  {cat}: {len(keys)} keys, {total_params:,} params")

    # Map to transformers
    mapped = _map_fairseq_to_transformers(fairseq_state)

    # Separate final_proj (handled outside mapping)
    final_proj_keys = {k for k in mapped if k.startswith("final_proj.")}
    transformers_keys = {k for k in mapped if k not in final_proj_keys}

    print(f"\n  Mapped keys (transformers): {len(transformers_keys)}")
    print(f"  Final proj keys: {len(final_proj_keys)}")

    # Load into fresh transformers model
    config = HubertConfig()
    model = HubertModel(config)
    model_state = model.state_dict()

    print(f"\n  Transformers model keys: {len(model_state)}")

    # Check: mapped keys that exist in model
    mapped_in_model = transformers_keys & set(model_state.keys())
    mapped_not_in_model = transformers_keys - set(model_state.keys())
    model_not_mapped = set(model_state.keys()) - transformers_keys

    print(f"\n  Mapped & in model: {len(mapped_in_model)}")
    print(f"  Mapped but NOT in model: {len(mapped_not_in_model)}")
    if mapped_not_in_model:
        for k in sorted(mapped_not_in_model):
            print(f"    UNMAPPED: {k} shape={mapped[k].shape}")

    print(f"  In model but NOT mapped: {len(model_not_mapped)}")
    if model_not_mapped:
        for k in sorted(model_not_mapped):
            print(f"    MISSING: {k} shape={model_state[k].shape}")

    # Shape comparison for mapped keys
    print(f"\n--- Shape mismatches ---")
    shape_mismatches = 0
    for k in sorted(mapped_in_model):
        if mapped[k].shape != model_state[k].shape:
            print(f"  SHAPE MISMATCH: {k}: fairseq={mapped[k].shape} vs transformers={model_state[k].shape}")
            shape_mismatches += 1
    if shape_mismatches == 0:
        print("  None (all shapes match)")

    # Load and check for NaN/Inf
    print(f"\n--- NaN/Inf check ---")
    transformers_only = {k: v for k, v in mapped.items() if k not in final_proj_keys}
    missing, unexpected = model.load_state_dict(transformers_only, strict=False)
    nan_count = 0
    inf_count = 0
    for k, v in model.state_dict().items():
        if torch.isnan(v).any():
            print(f"  NaN in: {k}")
            nan_count += 1
        if torch.isinf(v).any():
            print(f"  Inf in: {k}")
            inf_count += 1
    if nan_count == 0 and inf_count == 0:
        print("  None (all weights are finite)")

    # Parameter count comparison
    print(f"\n--- Parameter count ---")
    fairseq_params = sum(v.numel() for v in fairseq_state.values())
    model_params = sum(v.numel() for v in model.state_dict().values())
    mapped_params = sum(v.numel() for v in transformers_only.values())
    fp_params = sum(v.numel() for k, v in mapped.items() if k in final_proj_keys)
    print(f"  Fairseq total:     {fairseq_params:>12,}")
    print(f"  Mapped (no fp):    {mapped_params:>12,}")
    print(f"  Final proj:        {fp_params:>12,}")
    print(f"  Mapped + fp:       {mapped_params + fp_params:>12,}")
    print(f"  Transformers model:{model_params:>12,}")
    diff = fairseq_params - (mapped_params + fp_params)
    if diff != 0:
        print(f"  ** PARAM COUNT DIFF: {diff:,} (fairseq has {diff:,} more params)")
        # Find unmapped fairseq keys
        mapped_fairseq_keys = set()
        for fk in fairseq_state:
            mk = _map_fairseq_to_transformers({fk: fairseq_state[fk]})
            if mk:
                mapped_fairseq_keys.add(fk)
        unmapped_fairseq = set(fairseq_state.keys()) - mapped_fairseq_keys
        if unmapped_fairseq:
            print(f"  Unmapped fairseq keys ({len(unmapped_fairseq)}):")
            for k in sorted(unmapped_fairseq):
                print(f"    {k}: shape={fairseq_state[k].shape}, params={fairseq_state[k].numel()}")
    else:
        print("  OK: parameter counts match")


if __name__ == "__main__":
    main()
