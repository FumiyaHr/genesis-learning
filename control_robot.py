"""
Control Your Robot - Genesis チュートリアル 03

このスクリプトでは Genesis でのロボット制御方法を学びます：
1. Joint（関節）と DOF（自由度）の概念
2. 制御ゲインの設定
3. Hard Reset（物理を無視した状態設定）
4. PD 制御（位置・速度・力制御）
"""

import numpy as np

import genesis as gs

########## 初期化 ##########

gs.init(backend=gs.cpu)

########## シーン作成 ##########

scene = gs.Scene(
    # シミュレーション設定
    sim_options=gs.options.SimOptions(
        dt=0.01,  # タイムステップ（秒）: 1ステップ = 0.01秒
    ),
    # ビューア設定
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    show_viewer=True,
)

########## Entity 追加 ##########

# 床
plane = scene.add_entity(gs.morphs.Plane())

# Franka ロボットアーム
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########## シーンビルド ##########

scene.build()

########## Joint と DOF のマッピング ##########

# Franka ロボットの関節名
# - joint1〜7: アームの7関節（各1 DOF）
# - finger_joint1, finger_joint2: グリッパーの2関節（各1 DOF）
# 合計: 9 DOF

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]

# 関節名から DOF インデックスを取得
# dof_idx_local: そのロボット内でのローカルインデックス
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

print("関節名と DOF インデックス:")
for name, idx in zip(jnt_names, dofs_idx):
    print(f"  {name}: DOF {idx}")

########## 制御ゲインの設定 ##########

# PD 制御器のゲインを設定
# kp: 位置ゲイン（目標位置との差に比例した力を発生）
# kv: 速度ゲイン（速度に比例した減衰力を発生）

# 位置ゲイン（kp）を設定
# 値が大きいほど、目標位置に速く到達しようとする
franka.set_dofs_kp(
    kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local=dofs_idx,
)

# 速度ゲイン（kv）を設定
# 値が大きいほど、振動が少なくなる（ダンピング効果）
franka.set_dofs_kv(
    kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local=dofs_idx,
)

# 力の範囲を設定（安全のため）
# 各関節が出せる最大・最小トルク/力を制限
franka.set_dofs_force_range(
    lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    dofs_idx_local=dofs_idx,
)

print("\n制御ゲインを設定しました")

########## デモ 1: Hard Reset ##########

print("\n--- デモ 1: Hard Reset ---")
print("物理を無視して、ロボットの姿勢を直接変更します")

# 3つの姿勢を順番に設定
# set_dofs_position: 物理シミュレーションを無視して即座に移動
for i in range(150):
    if i < 50:
        # 姿勢 1: joint1 と joint2 を 1 rad に
        franka.set_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx
        )
    elif i < 100:
        # 姿勢 2: 複雑な姿勢
        franka.set_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx
        )
    else:
        # 姿勢 3: ホームポジション（全関節 0）
        franka.set_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx
        )
    scene.step()

print("Hard Reset デモ完了")

########## デモ 2: PD 制御（位置制御） ##########

print("\n--- デモ 2: PD 制御（位置制御） ---")
print("PD 制御器を使って、滑らかにロボットを動かします")

# 位置制御: 目標位置を設定すると、制御器が自動的に力を計算
# control_dofs_position: 一度設定すれば、目標が変わるまで維持される

for i in range(750):
    if i == 0:
        # 目標姿勢 1
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
        print(f"Step {i}: 姿勢 1 へ移動開始")
    elif i == 250:
        # 目標姿勢 2
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx,
        )
        print(f"Step {i}: 姿勢 2 へ移動開始")
    elif i == 500:
        # 目標姿勢 3（ホームポジション）
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
        print(f"Step {i}: ホームポジションへ移動開始")

    scene.step()

print("位置制御デモ完了")

########## デモ 3: ハイブリッド制御（位置 + 速度） ##########

print("\n--- デモ 3: ハイブリッド制御 ---")
print("joint1 は速度制御、他は位置制御")

for i in range(250):
    if i == 0:
        # joint1（dofs_idx[0]）は速度制御
        franka.control_dofs_velocity(
            np.array([1.0]),  # 1.0 rad/s で回転
            dofs_idx[:1],
        )
        # 残りの関節は位置制御
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx[1:],
        )
        print("joint1: 速度 1.0 rad/s で回転開始")

    scene.step()

print("ハイブリッド制御デモ完了")

########## デモ 4: 力制御（トルク制御） ##########

print("\n--- デモ 4: 力制御 ---")
print("全関節のトルクを 0 にして、重力で落下させます")

for i in range(200):
    if i == 0:
        # 全関節のトルクを 0 に設定
        # これにより、ロボットは重力で落下する
        franka.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
        print("全トルク = 0: 重力で落下開始")

    scene.step()

print("力制御デモ完了")

########## 力の取得 ##########

print("\n--- 力の情報 ---")

# 制御力: 制御器が計算した力
control_force = franka.get_dofs_control_force(dofs_idx)
print(f"制御力: {control_force}")

# 実際の力: 制御力 + 衝突力 + コリオリ力など
actual_force = franka.get_dofs_force(dofs_idx)
print(f"実際の力: {actual_force}")

print("\n完了！")
