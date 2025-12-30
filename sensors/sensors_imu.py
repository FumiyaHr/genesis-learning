"""
Sensors - IMU センサー - Genesis チュートリアル 08a

このスクリプトでは Genesis の IMU（慣性計測装置）センサーを学びます：
1. ロボットアームにIMUセンサーを取り付け
2. 円運動で加速度・角速度を計測
3. read() と read_ground_truth() の違い
"""

import numpy as np

import genesis as gs

########## 初期化 ##########

# macOS では Metal バックエンドを使用
gs.init(backend=gs.metal)

########## シーン作成 ##########

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,  # タイムステップ 10ms
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=False,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    show_viewer=True,
)

########## カメラ追加（録画用） ##########

cam = scene.add_camera(
    res=(1280, 720),
    pos=(3.5, 0.0, 2.5),
    lookat=(0.0, 0.0, 0.5),
    fov=40,
    GUI=False,
)

########## Entity 追加 ##########

# 床
plane = scene.add_entity(gs.morphs.Plane())

# Franka ロボットアーム
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# エンドエフェクタのリンクを取得
end_effector = franka.get_link('hand')

# 制御する関節のインデックス（グリッパー以外の7関節）
motors_dof = (0, 1, 2, 3, 4, 5, 6)

########## IMU センサー追加 ##########

# IMU をエンドエフェクタに取り付け
imu = scene.add_sensor(
    gs.sensors.IMU(
        # 取り付け先の指定
        entity_idx=franka.idx,                    # ロボットのインデックス
        link_idx_local=end_effector.idx_local,    # リンクのローカルインデックス
        pos_offset=(0.0, 0.0, 0.15),              # リンクからのオフセット位置

        # ノイズパラメータ（センサーの特性をシミュレート）
        acc_cross_axis_coupling=(0.0, 0.01, 0.02),  # 加速度計の軸間干渉
        gyro_cross_axis_coupling=(0.03, 0.04, 0.05),  # ジャイロの軸間干渉
        acc_noise=(0.01, 0.01, 0.01),              # 加速度計のノイズ標準偏差
        gyro_noise=(0.01, 0.01, 0.01),             # ジャイロのノイズ標準偏差
        acc_random_walk=(0.001, 0.001, 0.001),     # 加速度計のドリフト
        gyro_random_walk=(0.001, 0.001, 0.001),    # ジャイロのドリフト
        delay=0.01,                                # 遅延 (秒)
        jitter=0.01,                               # ジッター (秒)
        interpolate=True,                          # 遅延時の補間

        # デバッグ表示
        draw_debug=True,                           # センサー位置を可視化
    )
)

print("IMU センサーを追加しました")
print(f"  取り付け先: {end_effector}")
print(f"  ノイズあり: acc_noise=(0.01, 0.01, 0.01)")

########## ビルド ##########

scene.build()

########## PD ゲイン設定 ##########

# Franka の各関節に PD ゲインを設定
franka.set_dofs_kp(
    np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]),
)
franka.set_dofs_kv(
    np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]),
)
franka.set_dofs_force_range(
    np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0]),
    np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]),
)

########## 円運動のパラメータ ##########

circle_center = np.array([0.4, 0.0, 0.5])  # 円の中心
circle_radius = 0.15                        # 円の半径
rate = np.deg2rad(2.0)                      # 角速度 (rad/step)

########## 録画開始 ##########

print("録画を開始...")
cam.start_recording()

########## シミュレーション実行 ##########

print("\nIMU センサーシミュレーション実行中...")
print("- ロボットアームが円運動")
print("- IMU で加速度・角速度を計測")
print("- read(): ノイズあり、read_ground_truth(): 真値")

horizon = 300  # ステップ数

for i in range(horizon):
    scene.step()

    # 円運動の目標位置を計算
    pos = circle_center + np.array([
        np.cos(i * rate),
        np.sin(i * rate),
        0
    ]) * circle_radius

    # IK でロボットを目標位置に移動
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=pos,
        quat=np.array([0.0, 1.0, 0.0, 0.0]),  # 下向き
    )

    # グリッパー以外の関節を制御
    franka.control_dofs_position(qpos[:-2], motors_dof)

    # 目標位置を可視化
    scene.draw_debug_sphere(pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5))

    # カメラレンダリング
    cam.render()

    # 50ステップごとにセンサーデータを表示
    if i % 50 == 0:
        # 計測値（ノイズあり）
        data = imu.read()
        # 真値（ノイズなし）
        ground_truth = imu.read_ground_truth()

        print(f"\n[Step {i}]")
        print(f"  計測値 加速度: [{data.lin_acc[0]:7.3f}, {data.lin_acc[1]:7.3f}, {data.lin_acc[2]:7.3f}] m/s²")
        print(f"  真値   加速度: [{ground_truth.lin_acc[0]:7.3f}, {ground_truth.lin_acc[1]:7.3f}, {ground_truth.lin_acc[2]:7.3f}] m/s²")
        print(f"  計測値 角速度: [{data.ang_vel[0]:7.3f}, {data.ang_vel[1]:7.3f}, {data.ang_vel[2]:7.3f}] rad/s")
        print(f"  真値   角速度: [{ground_truth.ang_vel[0]:7.3f}, {ground_truth.ang_vel[1]:7.3f}, {ground_truth.ang_vel[2]:7.3f}] rad/s")

print("\nIMU センサーシミュレーション完了")

########## 最終データ表示 ##########

print("\n" + "=" * 50)
print("最終センサーデータ")
print("=" * 50)

data = imu.read()
ground_truth = imu.read_ground_truth()

print("\n計測値（ノイズあり）:")
print(f"  線形加速度: {data.lin_acc} m/s²")
print(f"  角速度: {data.ang_vel} rad/s")

print("\n真値（ノイズなし）:")
print(f"  線形加速度: {ground_truth.lin_acc} m/s²")
print(f"  角速度: {ground_truth.ang_vel} rad/s")

########## 録画停止・保存 ##########

cam.stop_recording(save_to_filename='sensors_imu.mp4', fps=60)
print("\n動画を 'sensors_imu.mp4' に保存しました")

########## まとめ ##########

print("\n" + "=" * 50)
print("IMU センサーのまとめ")
print("=" * 50)
print("""
1. センサー追加:
   imu = scene.add_sensor(
       gs.sensors.IMU(
           entity_idx=robot.idx,
           link_idx_local=link.idx_local,
           pos_offset=(0, 0, 0.15),
           acc_noise=(0.01, 0.01, 0.01),
           gyro_noise=(0.01, 0.01, 0.01),
           draw_debug=True,
       )
   )

2. データ読み取り:
   data = imu.read()              # ノイズあり
   ground_truth = imu.read_ground_truth()  # 真値

3. 出力データ:
   data.lin_acc  # 線形加速度 (x, y, z) m/s²
   data.ang_vel  # 角速度 (x, y, z) rad/s

4. ノイズパラメータ:
   - acc_noise, gyro_noise: ガウスノイズ
   - acc_random_walk, gyro_random_walk: ドリフト
   - delay, jitter: タイミングノイズ
   - acc_cross_axis_coupling: 軸間干渉
""")

print("完了！")
