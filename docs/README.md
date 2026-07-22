# docs 索引

RCWX リポジトリ内の補足ドキュメント。  
**現行の製品仕様・開発ガイド**はルートの次を正とする:

| 文書 | 用途 |
|------|------|
| [README.md](../README.md) | ユーザー向け（セットアップ・Latency・設定） |
| [AGENTS.md](../AGENTS.md) / [CLAUDE.md](../CLAUDE.md) | エージェント/開発者向け AS-IS ガイド |
| [SETUP.md](SETUP.md) | 環境構築の詳細（`rcwx.ps1` 中心） |

## このディレクトリの文書

| ファイル | 種別 | 内容 |
|----------|------|------|
| [SETUP.md](SETUP.md) | **現行** | Windows + Intel Arc の構築手順、トラブルシュート |
| [moe_boost_feminization.md](moe_boost_feminization.md) | **現行設計** | Moe Boost の数式・係数・テスト参照 |
| [moe_spec.md](moe_spec.md) | 背景リサーチ | 萌声/ロリ声の一般的 F0 帯域メモ（設定値ではない） |
| [diff_report.md](diff_report.md) | 監査メモ | オリジナル RVC との互換性作業記録 + **冒頭に現行乖離メモ** |
| [pipeline-comparison-wokada.md](pipeline-comparison-wokada.md) | 比較メモ | w-okada/voice-changer との設計比較 + **冒頭に現行 RCWX メモ** |

## 読み方の注意

- 監査・比較文書は執筆時点のスナップショットを含む。冒頭の **現行メモ (AS-IS)** とルートの AGENTS/README が食い違う場合は後者を優先する。
- CUDA / NVIDIA 手順は未検証のため、セットアップ文書からは外している。
- 最適化の「変遷・before/after」はユーザー向け文書から削除済み。性能目安は README Latency の表を参照。
