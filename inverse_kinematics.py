"""
Inverse Kinematics & Motion Planning - Genesis チュートリアル 05

このスクリプトでは Genesis の逆運動学とモーションプランニングを学びます：
1. 逆運動学（IK）でエンドエフェクタの目標位置から関節角度を計算
2. モーションプランニングで障害物を避けながらパスを計算
3. 力制御でオブジェクトを把持（グリッピング）
"""

import numpy as np

import genesis as gs

########## 初期化 ##########

# macOS では CPU バックエンドを使用
gs.init(backend=gs.cpu)

########## シーン作成 ##########

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,  # タイムステップ（秒）
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)

########## カメラ追加（録画用） ##########

cam = scene.add_camera(
    res=(1280, 720),
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    fov=30,
    GUI=False,
)

########## Entity 追加 ##########

# 床
plane = scene.add_entity(gs.morphs.Plane())

# 把持対象のキューブ（4cm角）
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),  # 4cm x 4cm x 4cm
        pos=(0.65, 0.0, 0.02),    # ロボットの前方に配置
    )
)

# Franka ロボットアーム
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########## ビルド ##########

scene.build()

########## 録画開始 ##########

print("録画を開始...")
cam.start_recording()

########## 制御ゲインの設定 ##########

# DOF インデックス
# motors_dof: アーム部分（7軸）
# fingers_dof: グリッパー部分（2軸）
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 制御ゲインを設定（Franka 用に調整済み）
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

########## エンドエフェクタの取得 ##########

# 'hand' リンク = グリッパーの付け根
end_effector = franka.get_link('hand')
print(f"エンドエフェクタ: {end_effector}")

########## ステップ 1: プリグラスプ位置へ移動 ##########

print("\n--- ステップ 1: プリグラスプ位置へ移動 ---")

# 逆運動学（IK）で目標姿勢から関節角度を計算
# pos: エンドエフェクタの目標位置 (x, y, z)
# quat: 目標姿勢（クォータニオン w, x, y, z）
#       [0, 1, 0, 0] = 下向き（Z軸を-Z方向に向ける）
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),  # キューブの上方
    quat=np.array([0, 1, 0, 0]),      # 下向き
)
print(f"IK 解: {qpos}")

# グリッパーを開く（0.04m = 4cm）
qpos[-2:] = 0.04

# モーションプランニングでパスを計算
# num_waypoints: 経路上の中間点の数（= シミュレーションステップ数）
print("モーションプランニング中...")
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 200ステップ = 2秒（dt=0.01）
)
print(f"パス計算完了: {len(path)} ウェイポイント")

# パスを実行
print("パス実行中...")
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    cam.render()

# PD制御の遅れを補償（最後のウェイポイントに到達するまで待機）
for i in range(100):
    scene.step()
    cam.render()

print("プリグラスプ位置に到達")

########## ステップ 2: キューブに接近 ##########

print("\n--- ステップ 2: キューブに接近 ---")

# IK でキューブ把持位置を計算
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),  # キューブの高さ
    quat=np.array([0, 1, 0, 0]),
)

# アーム部分のみ制御（グリッパーは開いたまま）
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()
    cam.render()

print("キューブに接近完了")

########## ステップ 3: 把持（グリッピング） ##########

print("\n--- ステップ 3: 把持 ---")

# アームの位置を維持しながら、グリッパーに力を加える
franka.control_dofs_position(qpos[:-2], motors_dof)

# 力制御でグリッパーを閉じる
# 負の力 = 内側に閉じる方向
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(100):
    scene.step()
    cam.render()

print("把持完了")

########## ステップ 4: 持ち上げ ##########

print("\n--- ステップ 4: 持ち上げ ---")

# IK で持ち上げ位置を計算
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),  # 上方に移動
    quat=np.array([0, 1, 0, 0]),
)

# アーム部分を制御（グリッパーは力制御を継続）
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(200):
    scene.step()
    cam.render()

print("持ち上げ完了！")

########## 録画停止・保存 ##########

cam.stop_recording(save_to_filename='inverse_kinematics.mp4', fps=60)
print("動画を 'inverse_kinematics.mp4' に保存しました")

########## まとめ ##########

print("\n" + "=" * 50)
print("逆運動学 & モーションプランニングのまとめ")
print("=" * 50)
print("""
1. 逆運動学（IK）:
   qpos = franka.inverse_kinematics(
       link=end_effector,
       pos=np.array([x, y, z]),
       quat=np.array([w, x, y, z]),
   )

2. モーションプランニング:
   path = franka.plan_path(
       qpos_goal=qpos,
       num_waypoints=200,
   )

3. パス実行:
   for waypoint in path:
       franka.control_dofs_position(waypoint)
       scene.step()

4. 力制御で把持:
   franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
""")

# ビューアを維持
print("完了！ビューアを閉じるまで待機中...")
for i in range(500):
    scene.step()
