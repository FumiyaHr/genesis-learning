"""
Sensors - 接触力センサー - Genesis チュートリアル 08b

このスクリプトでは Genesis の接触力センサーを学びます：
1. 四足ロボット（Go2）の各足に接触力センサーを取り付け
2. 地面との接触力を計測
3. デバッグ表示で力の方向を可視化
"""

import genesis as gs

########## 初期化 ##########

# macOS では Metal バックエンドを使用
gs.init(backend=gs.metal)

########## シーン作成 ##########

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,  # タイムステップ 10ms
    ),
    rigid_options=gs.options.RigidOptions(
        constraint_timeconst=0.02,     # 拘束ソルバーの時定数
        use_gjk_collision=True,        # GJK 衝突検出を使用
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, -2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=50,
        max_FPS=60,
    ),
    show_viewer=True,
)

########## カメラ追加（録画用） ##########

cam = scene.add_camera(
    res=(1280, 720),
    pos=(2.0, -2.0, 1.5),
    lookat=(0.0, 0.0, 0.3),
    fov=50,
    GUI=False,
)

########## Entity 追加 ##########

# 床
plane = scene.add_entity(gs.morphs.Plane())

# Go2 四足ロボット
# links_to_keep: 衝突検出を行うリンクを指定
foot_link_names = ('FR_foot', 'FL_foot', 'RR_foot', 'RL_foot')
go2 = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/go2/urdf/go2.urdf',
        pos=(0.0, 0.0, 0.5),  # 少し高い位置からスタート（落下させる）
        links_to_keep=foot_link_names,  # 足のリンクのみ衝突検出
    ),
)

print("Go2 ロボットを追加しました")
print(f"  足のリンク: {foot_link_names}")

########## 接触力センサー追加 ##########

# 各足に接触力センサーを取り付け
sensors = {}
for foot_name in foot_link_names:
    sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            # 取り付け先の指定
            entity_idx=go2.idx,                           # ロボットのインデックス
            link_idx_local=go2.get_link(foot_name).idx_local,  # 足リンクのインデックス

            # デバッグ表示
            draw_debug=True,   # 接触力の方向を矢印で表示
        )
    )
    sensors[foot_name] = sensor
    print(f"  {foot_name} に接触力センサーを追加")

########## ビルド ##########

scene.build()

########## 録画開始 ##########

print("\n録画を開始...")
cam.start_recording()

########## シミュレーション実行 ##########

print("\n接触力センサーシミュレーション実行中...")
print("- Go2 ロボットが落下して地面に着地")
print("- 各足の接触力を計測")
print("- draw_debug=True で力の方向が可視化されます")

horizon = 300  # ステップ数

for i in range(horizon):
    scene.step()
    cam.render()

    # 50ステップごとにセンサーデータを表示
    if i % 50 == 0:
        print(f"\n[Step {i}]")
        for foot_name, sensor in sensors.items():
            # 接触力を読み取り
            # ContactForce センサーは直接テンソルを返す (3次元: x, y, z)
            force = sensor.read()

            # テンソルから値を取得
            fx, fy, fz = float(force[0]), float(force[1]), float(force[2])

            # 力の大きさを計算
            force_magnitude = (fx**2 + fy**2 + fz**2)**0.5

            print(f"  {foot_name}: force=[{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] N, |F|={force_magnitude:7.2f} N")

print("\n接触力センサーシミュレーション完了")

########## 最終データ表示 ##########

print("\n" + "=" * 60)
print("最終センサーデータ")
print("=" * 60)

for foot_name, sensor in sensors.items():
    force = sensor.read()
    fx, fy, fz = float(force[0]), float(force[1]), float(force[2])
    force_magnitude = (fx**2 + fy**2 + fz**2)**0.5
    print(f"{foot_name}: force=[{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] N, |F|={force_magnitude:7.2f} N")

########## 録画停止・保存 ##########

cam.stop_recording(save_to_filename='sensors_contact.mp4', fps=60)
print("\n動画を 'sensors_contact.mp4' に保存しました")

########## まとめ ##########

print("\n" + "=" * 60)
print("接触力センサーのまとめ")
print("=" * 60)
print("""
1. センサー追加:
   sensor = scene.add_sensor(
       gs.sensors.ContactForce(
           entity_idx=robot.idx,
           link_idx_local=link.idx_local,
           draw_debug=True,  # 力の方向を矢印で可視化
       )
   )

2. データ読み取り:
   force_data = sensor.read()
   force = force_data.force  # (x, y, z) ニュートン

3. 接触センサー（ブール値）:
   # 接触有無のみ知りたい場合
   sensor = scene.add_sensor(
       gs.sensors.Contact(
           entity_idx=robot.idx,
           link_idx_local=link.idx_local,
       )
   )
   data = sensor.read()
   in_contact = data.in_contact  # True/False

4. デバッグ表示:
   - draw_debug=True で接触力の方向が矢印として表示される
   - 矢印の長さは力の大きさに比例

5. 用途:
   - 歩行ロボットの接地検出
   - 把持力の計測
   - 衝突検出
""")

print("完了！")
