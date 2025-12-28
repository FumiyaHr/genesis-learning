"""
Interactive Debugging - Genesis チュートリアル 09

このスクリプトでは Genesis の対話的デバッグ機能を学びます：
1. IPython を使った対話モード
2. Genesis オブジェクトの構造探索
3. __repr__() による美しい出力

Genesis の全クラスには __repr__() メソッドが実装されており、
オブジェクトを入力するだけで内部構造が美しくフォーマットされて表示されます。
"""

import genesis as gs

########## 初期化 ##########

# macOS では Metal バックエンドを使用
gs.init(backend=gs.metal)

########## シーン作成 ##########

# show_viewer=False: 対話デバッグがメインなのでビューアは不要
scene = gs.Scene(show_viewer=False)

########## Entity 追加 ##########

# 床
plane = scene.add_entity(gs.morphs.Plane())

# Franka ロボットアーム
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########## カメラ追加 ##########

cam = scene.add_camera()

########## ビルド ##########

scene.build()

########## 対話モードの使い方 ##########
#
# IPython 対話モードで以下のオブジェクトを入力して探索してみましょう：
#
# ■ シーン全体
#   scene           - シーンの情報（is_built, dt, uid, solvers など）
#   scene.solvers   - 物理ソルバーのリスト
#
# ■ エンティティ
#   franka          - Franka ロボットの情報（geoms, links など）
#   franka.links    - リンクのリスト
#   franka.joints   - 関節のリスト
#   franka.geoms    - ジオメトリのリスト
#
# ■ リンク詳細
#   franka.links[0]         - 最初のリンクの詳細
#   franka.links[0].geoms   - 衝突ジオメトリ
#   franka.links[0].vgeoms  - 視覚ジオメトリ
#   franka.links[0].joint   - 関連する関節
#   franka.links[0].inertial_mass  - 慣性質量
#
# ■ カメラ
#   cam             - カメラの情報
#
# 終了: exit() または Ctrl+D
#

print("=" * 60)
print("Genesis Interactive Debugging - チュートリアル 09")
print("=" * 60)
print("""
このスクリプトでは IPython を使って Genesis オブジェクトを
対話的に探索します。

■ 探索例:
  scene           - シーン全体の情報
  scene.solvers   - 物理ソルバーのリスト
  franka          - Franka ロボットの情報
  franka.links    - リンクのリスト
  franka.links[0] - 最初のリンクの詳細
  franka.joints   - 関節のリスト
  cam             - カメラの情報

■ 終了方法:
  exit() または Ctrl+D
""")

########## IPython 対話モード ##########

import IPython
IPython.embed()

print("\n対話モード終了")
print("完了！")
