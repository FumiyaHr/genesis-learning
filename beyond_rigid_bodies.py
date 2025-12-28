"""
Beyond Rigid Bodies - Genesis チュートリアル 07

このスクリプトでは剛体以外の物理シミュレーションを学びます：
1. SPH Solver - 液体シミュレーション
2. MPM Solver - 変形物体シミュレーション
3. PBD Solver - 布シミュレーション
"""

import genesis as gs

########## パート 1: SPH 液体シミュレーション ##########

print("=" * 50)
print("パート 1: SPH 液体シミュレーション")
print("=" * 50)

########## 初期化 ##########

# macOS では CPU バックエンドを使用
gs.init(backend=gs.metal)

########## シーン作成 ##########

# SPH ソルバーを使用するシーン
scene1 = gs.Scene(
    # シミュレーション設定
    # 非剛体シミュレーションでは substeps が重要
    sim_options=gs.options.SimOptions(
        dt=4e-3,        # タイムステップ
        substeps=10,    # 1ステップあたりのサブステップ数
    ),
    # SPH ソルバーの設定
    sph_options=gs.options.SPHOptions(
        lower_bound=(-0.5, -0.5, 0.0),   # ソルバー境界（下限）
        upper_bound=(0.5, 0.5, 1.0),     # ソルバー境界（上限）
        particle_size=0.01,               # パーティクルサイズ
    ),
    # 可視化設定
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary=True,      # SPH境界を表示
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, -1.5, 1.0),
        camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=40,
        max_FPS=60,
    ),
    show_viewer=True,
)

########## カメラ追加（録画用） ##########

cam1 = scene1.add_camera(
    res=(1280, 720),
    pos=(1.5, -1.5, 1.0),
    lookat=(0.0, 0.0, 0.3),
    fov=40,
    GUI=False,
)

########## Entity 追加 ##########

# 床
plane1 = scene1.add_entity(morph=gs.morphs.Plane())

# 水ブロック
# material で物質を指定（SPH.Liquid = 液体）
# 注意: 'pbs' サンプラーは Linux x86 のみ対応、macOS では 'regular' を使用
liquid = scene1.add_entity(
    material=gs.materials.SPH.Liquid(
        sampler='regular',  # グリッドパターンでサンプリング
    ),
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.65),
        size=(0.4, 0.4, 0.4),
    ),
    # surface で可視化プロパティを設定
    surface=gs.surfaces.Default(
        color=(0.4, 0.8, 1.0),      # 水色
        vis_mode='particle',         # パーティクルとして表示
    ),
)

########## ビルド ##########

scene1.build()

########## 録画開始 ##########

print("録画を開始...")
cam1.start_recording()

########## シミュレーション実行 ##########

print("液体シミュレーション実行中...")
print("- SPH（Smooth Particle Hydrodynamics）ソルバー")
print("- パーティクルで液体を表現")

horizon = 200  # ステップ数（CPUでは重いため短縮）
for i in range(horizon):
    scene1.step()
    cam1.render()

print("SPH 液体シミュレーション完了")

########## 録画停止・保存 ##########

cam1.stop_recording(save_to_filename='beyond_rigid_sph.mp4', fps=60)
print("動画を 'beyond_rigid_sph.mp4' に保存しました")

########## パート 2: MPM 変形物体シミュレーション ##########

print("\n" + "=" * 50)
print("パート 2: MPM 変形物体シミュレーション")
print("=" * 50)

########## 新しいシーン作成 ##########

# 複数シーンでは同時にビューアを使えないため、show_viewer=False
scene2 = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    # MPM ソルバーの設定
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.5, -1.0, 0.0),
        upper_bound=(0.5, 1.0, 1.0),
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    show_viewer=False,
)

########## カメラ追加（録画用） ##########

cam2 = scene2.add_camera(
    res=(1280, 720),
    pos=(1.5, -2.0, 1.5),
    lookat=(0.0, 0.0, 0.3),
    fov=40,
    GUI=False,
)

########## Entity 追加 ##########

plane2 = scene2.add_entity(morph=gs.morphs.Plane())

# 弾性体（赤）- vis_mode='visual' でスキニング表示
obj_elastic = scene2.add_entity(
    material=gs.materials.MPM.Elastic(),
    morph=gs.morphs.Box(
        pos=(0.0, -0.5, 0.25),
        size=(0.2, 0.2, 0.2),
    ),
    surface=gs.surfaces.Default(
        color=(1.0, 0.4, 0.4),
        vis_mode='visual',  # メッシュとして表示（スキニング）
    ),
)

# 液体（青）- vis_mode='particle' でパーティクル表示
obj_liquid = scene2.add_entity(
    material=gs.materials.MPM.Liquid(),
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.25),
        size=(0.3, 0.3, 0.3),
    ),
    surface=gs.surfaces.Default(
        color=(0.3, 0.3, 1.0),
        vis_mode='particle',
    ),
)

# 弾塑性体（緑）- 永久変形する素材
obj_plastic = scene2.add_entity(
    material=gs.materials.MPM.ElastoPlastic(),
    morph=gs.morphs.Sphere(
        pos=(0.0, 0.5, 0.35),
        radius=0.1,
    ),
    surface=gs.surfaces.Default(
        color=(0.4, 1.0, 0.4),
        vis_mode='particle',
    ),
)

########## ビルド ##########

scene2.build()

########## 録画開始 ##########

print("録画を開始...")
cam2.start_recording()

########## シミュレーション実行 ##########

print("MPM シミュレーション実行中...")
print("- Elastic（弾性体）: 変形後に元に戻る")
print("- Liquid（液体）: 流動する")
print("- ElastoPlastic（弾塑性体）: 永久変形する")

horizon = 200  # CPUでは重いため短縮
for i in range(horizon):
    scene2.step()
    cam2.render()

print("MPM 変形物体シミュレーション完了")

########## 録画停止・保存 ##########

cam2.stop_recording(save_to_filename='beyond_rigid_mpm.mp4', fps=60)
print("動画を 'beyond_rigid_mpm.mp4' に保存しました")

########## パート 3: PBD 布シミュレーション ##########

print("\n" + "=" * 50)
print("パート 3: PBD 布シミュレーション")
print("=" * 50)

########## 新しいシーン作成 ##########

scene3 = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
    ),
    show_viewer=False,
)

########## カメラ追加（録画用） ##########

cam3 = scene3.add_camera(
    res=(1280, 720),
    pos=(3.0, -3.0, 2.0),
    lookat=(0.0, 0.0, 0.5),
    fov=30,
    GUI=False,
)

########## Entity 追加 ##########

plane3 = scene3.add_entity(morph=gs.morphs.Plane())

# 布 1（青）- 4隅を固定
cloth_1 = scene3.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file='meshes/cloth.obj',
        scale=2.0,
        pos=(0, 0, 0.5),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode='visual',
    ),
)

# 布 2（オレンジ）- 1隅のみ固定、落下する
cloth_2 = scene3.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file='meshes/cloth.obj',
        scale=2.0,
        pos=(0, 0, 1.0),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.8, 0.4, 0.2, 1.0),
        vis_mode='particle',
    ),
)

########## ビルド ##########

scene3.build()

########## パーティクルを固定 ##########

# 布 1: 4隅を固定
# find_closest_particle() で位置から最も近いパーティクルを検索
# fix_particles() で複数のパーティクルを固定（リストで渡す）
cloth_1.fix_particles([
    cloth_1.find_closest_particle((-1, -1, 1.0)),
    cloth_1.find_closest_particle((1, 1, 1.0)),
    cloth_1.find_closest_particle((-1, 1, 1.0)),
    cloth_1.find_closest_particle((1, -1, 1.0)),
])

# 布 2: 1隅のみ固定
cloth_2.fix_particles([cloth_2.find_closest_particle((-1, -1, 1.0))])

print("布のパーティクルを固定しました")

########## 録画開始 ##########

print("録画を開始...")
cam3.start_recording()

########## シミュレーション実行 ##########

print("PBD 布シミュレーション実行中...")
print("- 布 1: 4隅固定（ハンモック状）")
print("- 布 2: 1隅固定（落下して布 1 の上に）")

horizon = 200  # CPUでは重いため短縮
for i in range(horizon):
    scene3.step()
    cam3.render()

print("PBD 布シミュレーション完了")

########## 録画停止・保存 ##########

cam3.stop_recording(save_to_filename='beyond_rigid_pbd.mp4', fps=60)
print("動画を 'beyond_rigid_pbd.mp4' に保存しました")

########## まとめ ##########

print("\n" + "=" * 50)
print("Beyond Rigid Bodies のまとめ")
print("=" * 50)
print("""
1. SPH Solver（Smooth Particle Hydrodynamics）:
   - 液体シミュレーション
   - material=gs.materials.SPH.Liquid()
   - sph_options で境界とパーティクルサイズを設定

2. MPM Solver（Material Point Method）:
   - 変形物体シミュレーション
   - Elastic（弾性体）、Liquid（液体）、ElastoPlastic（弾塑性体）
   - Sand（砂）、Snow（雪）なども利用可能
   - mpm_options で境界を設定

3. PBD Solver（Position Based Dynamics）:
   - 布シミュレーション
   - material=gs.materials.PBD.Cloth()
   - fix_particle() でパーティクルを固定
   - find_closest_particle() で位置からパーティクル検索

4. 共通設定:
   - sim_options: dt, substeps（非剛体は substeps=10 推奨）
   - surface: color, vis_mode（'particle' or 'visual'）
""")

print("完了！")
