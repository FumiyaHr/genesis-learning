# Genesis Learning

Genesis 物理シミュレーションエンジンの学習プロジェクト。

## 概要

このプロジェクトは Genesis の公式ドキュメントに沿って、物理シミュレーションとロボット制御を学ぶためのものです。

**対象者**: Genesis・強化学習ともに初心者

## セットアップ

### 必要条件
- Python 3.12 以上
- uv（パッケージマネージャー）

### インストール

```bash
cd genesis-learning
uv sync
```

## 実行方法

```bash
uv run python <script>.py
```

## 学習スクリプト

| スクリプト | 内容 |
|-----------|------|
| `hello_genesis.py` | 最初のシミュレーション |
| `visualization.py` | 可視化とカメラ操作 |
| `control_robot.py` | ロボット制御（PD制御） |
| `parallel_simulation.py` | 並列シミュレーション |

## ドキュメント

詳細な学習記録は `../docs/` にあります。

- `00_genesis_overview.md` - Genesis の概要
- `01_setup_and_hello_genesis.md` - 環境構築
- `02_visualization.md` - 可視化
- `03_control_your_robot.md` - ロボット制御
- `04_parallel_simulation.md` - 並列シミュレーション

## 依存パッケージ

- `genesis-world` - Genesis 物理エンジン
- `torch` - PyTorch（機械学習ライブラリ）
