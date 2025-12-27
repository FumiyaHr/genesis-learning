"""
Advanced IK - Genesis チュートリアル 06

このスクリプトでは Genesis の高度な逆運動学機能を学びます：
1. 複数エンドエフェクタの同時IK解決
2. 回転マスク（rot_mask）による柔軟な目標設定
3. 並列環境でのバッチIK
"""

import numpy as np

import genesis as gs

########## パート 1: 複数リンクの IK ##########

print("=" * 50)
print("パート 1: 複数リンクの IK")
print("=" * 50)

########## 初期化 ##########

# macOS では CPU バックエンドを使用
gs.init(backend=gs.cpu)

########## シーン作成 ##########

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, -2, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=60,
    ),
    # IK のデモなので衝突と関節制限を無効化
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=False,
        enable_collision=False,
    ),
    show_viewer=True,
)

########## カメラ追加（録画用） ##########

cam = scene.add_camera(
    res=(1280, 720),
    pos=(2.0, -2, 1.5),
    lookat=(0.0, 0.0, 0.0),
    fov=40,
    GUI=False,
)

########## Entity 追加 ##########

# 床
plane = scene.add_entity(gs.morphs.Plane())

# Franka ロボットアーム
robot = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# 左指のターゲット可視化用（赤）
target_left = scene.add_entity(
    gs.morphs.Mesh(
        file='meshes/axis.obj',
        scale=0.1,
    ),
    surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
)

# 右指のターゲット可視化用（緑）
target_right = scene.add_entity(
    gs.morphs.Mesh(
        file='meshes/axis.obj',
        scale=0.1,
    ),
    surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),
)

########## ビルド ##########

scene.build()

########## 録画開始 ##########

print("録画を開始...")
cam.start_recording()

########## 複数リンク IK のデモ ##########

# 円運動のパラメータ
target_quat = np.array([0, 1, 0, 0])  # 下向き
center = np.array([0.4, -0.2, 0.25])  # 円の中心
r = 0.1  # 円の半径

# 左右の指リンクを取得
left_finger = robot.get_link('left_finger')
right_finger = robot.get_link('right_finger')

print(f"左指リンク: {left_finger}")
print(f"右指リンク: {right_finger}")

print("\n複数リンク IK を実行中...")
print("- inverse_kinematics_multilink() で左右の指を同時に制御")
print("- rot_mask=[False, False, True] で Z 軸方向のみ制限")

for i in range(500):  # 短縮版（元は2000）
    # 左指のターゲット位置（円運動）
    target_pos_left = center + np.array([
        np.cos(i / 360 * np.pi),
        np.sin(i / 360 * np.pi),
        0
    ]) * r

    # 右指のターゲット位置（左指から少しずらす）
    target_pos_right = target_pos_left + np.array([0.0, 0.03, 0])

    # ターゲットマーカーの位置を更新
    # qpos = [x, y, z, w, qx, qy, qz]（位置 + クォータニオン）
    target_left.set_qpos(np.concatenate([target_pos_left, target_quat]))
    target_right.set_qpos(np.concatenate([target_pos_right, target_quat]))

    # 複数リンクの IK を解く
    # rot_mask: [x軸, y軸, z軸] の制限
    #   False = 制限しない、True = 制限する
    #   [False, False, True] = Z軸方向のみ下向きに制限（水平回転は自由）
    q = robot.inverse_kinematics_multilink(
        links=[left_finger, right_finger],
        poss=[target_pos_left, target_pos_right],
        quats=[target_quat, target_quat],
        rot_mask=[False, False, True],
    )

    # IK の結果をロボットに適用
    # 注意: このデモは物理シミュレーションなし
    # scene.step() は呼ばず、状態を直接設定して可視化のみ更新
    robot.set_dofs_position(q)
    scene.visualizer.update()

    # 録画のためにカメラをレンダリング
    cam.render()

print("複数リンク IK デモ完了")

########## 録画停止・保存（パート1） ##########

cam.stop_recording(save_to_filename='advanced_ik_multilink.mp4', fps=60)
print("動画を 'advanced_ik_multilink.mp4' に保存しました")

########## パート 2: 並列環境での IK ##########

print("\n" + "=" * 50)
print("パート 2: 並列環境での IK")
print("=" * 50)

########## 新しいシーン作成（並列環境用） ##########

# 複数シーンでは同時にビューアを使えないため、show_viewer=False
# カメラ録画のみで可視化
scene2 = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=False,
    ),
    show_viewer=False,
)

########## カメラ追加（録画用） ##########

cam2 = scene2.add_camera(
    res=(1280, 720),
    pos=(0.0, -4, 2.5),
    lookat=(0.0, 0.0, 0.5),
    fov=50,
    GUI=False,
)

########## Entity 追加 ##########

plane2 = scene2.add_entity(gs.morphs.Plane())
robot2 = scene2.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########## 並列環境でビルド ##########

n_envs = 16
print(f"並列環境数: {n_envs}")
scene2.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

########## 録画開始 ##########

print("録画を開始...")
cam2.start_recording()

########## バッチ IK のデモ ##########

# 各環境で同じ初期値を使用
target_quat_batch = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # 下向き
center_batch = np.tile(np.array([0.4, -0.2, 0.25]), [n_envs, 1])

# 各環境ごとに異なる角速度（-10 〜 10）
np.random.seed(42)  # 再現性のため
angular_speed = np.random.uniform(-10, 10, n_envs)
print(f"各環境の角速度: {angular_speed.round(2)}")

r = 0.1

# エンドエフェクタのリンクを取得
ee_link = robot2.get_link('hand')

print("\nバッチ IK を実行中...")
print("- 各環境のエンドエフェクタが異なる速度で円運動")

for i in range(500):  # 短縮版（元は1000）
    # 各環境のターゲット位置を計算
    # バッチ次元を追加: (n_envs, 3)
    target_pos = np.zeros([n_envs, 3])
    target_pos[:, 0] = center_batch[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
    target_pos[:, 1] = center_batch[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
    target_pos[:, 2] = center_batch[:, 2]

    # バッチ IK を解く
    # 入力にバッチ次元を追加するだけで並列処理される
    q = robot2.inverse_kinematics(
        link=ee_link,
        pos=target_pos,  # (n_envs, 3)
        quat=target_quat_batch,  # (n_envs, 4)
        rot_mask=[False, False, True],  # Z軸方向のみ制限
    )

    # qpos で状態を設定
    robot2.set_qpos(q)
    scene2.step()
    cam2.render()

print("バッチ IK デモ完了")

########## 録画停止・保存（パート2） ##########

cam2.stop_recording(save_to_filename='advanced_ik_parallel.mp4', fps=60)
print("動画を 'advanced_ik_parallel.mp4' に保存しました")

########## まとめ ##########

print("\n" + "=" * 50)
print("Advanced IK のまとめ")
print("=" * 50)
print("""
1. 複数リンクの IK:
   q = robot.inverse_kinematics_multilink(
       links=[left_finger, right_finger],
       poss=[pos1, pos2],
       quats=[quat1, quat2],
       rot_mask=[False, False, True],  # Z軸のみ制限
   )

2. 並列環境での IK:
   # バッチ次元を追加するだけ
   q = robot.inverse_kinematics(
       link=ee_link,
       pos=target_pos,   # (n_envs, 3)
       quat=target_quat, # (n_envs, 4)
   )

3. qpos vs dofs_position:
   - qpos: 一般化座標（位置 + クォータニオン = 7次元）
   - dofs_position: 関節角度（DOF数次元）
   - フリージョイントの場合は qpos を使用

4. 可視化のみの更新（物理シミュレーションなし）:
   robot.set_dofs_position(q)
   scene.visualizer.update()
""")

print("完了！")
