"""
Hello Genesis - 最初のチュートリアル

このスクリプトは Genesis の基本的な使い方を示します。
Franka ロボットアームを床の上に配置し、重力でシミュレーションします。
"""

import genesis as gs

# Genesis を初期化
# backend: 計算を実行するデバイス
#   - gs.cpu: CPU で実行（最も互換性が高い）
#   - gs.gpu: GPU を自動選択（Apple Silicon では gs.metal が選ばれる）
#   - gs.metal: Apple Silicon GPU
#   - gs.cuda: NVIDIA GPU
gs.init(backend=gs.cpu)

# シーンを作成
# シーン = シミュレーション空間。すべてのオブジェクトはここに配置される
# show_viewer=True: 可視化ウィンドウを表示
scene = gs.Scene(show_viewer=True)

# 床（平面）を追加
# Entity: シーン内のオブジェクト
# Morph: Entity の形状を定義（Plane = 無限平面）
plane = scene.add_entity(gs.morphs.Plane())

# Franka ロボットアームを追加
# MJCF: MuJoCo 形式のロボット定義ファイル
# file: Genesis 内蔵のアセットから Franka Panda ロボットを読み込む
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# シーンをビルド
# JIT (Just-In-Time) コンパイルでカーネルを生成
# 初回実行時は時間がかかるが、キャッシュされる
scene.build()

# シミュレーションを1000ステップ実行
# 各ステップでロボットに物理法則（重力など）が適用される
for i in range(1000):
    scene.step()
