"""
Parallel Simulation - Genesis チュートリアル 04

このスクリプトでは Genesis の並列シミュレーション機能を学びます：
1. 複数の環境を同時にシミュレート（バッチ処理）
2. 全環境の一括制御
3. 特定環境のみの制御
4. 各環境に異なる姿勢を設定
"""

import torch

import genesis as gs

########## 初期化 ##########

# macOS では CPU バックエンドを使用
# GPU があれば gs.gpu を使うことで大幅に高速化できる
gs.init(backend=gs.cpu)

########## シーン作成 ##########

scene = gs.Scene(
    # シミュレーション設定
    sim_options=gs.options.SimOptions(
        dt=0.01,  # タイムステップ（秒）
    ),
    # ビューア設定
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(5.0, -3.0, 3.0),  # 並列環境を見渡せる位置
        camera_lookat=(2.0, 1.0, 0.5),
        camera_fov=40,
        res=(1280, 720),
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

########## 並列環境でビルド ##########

# n_envs: 並列環境の数（バッチサイズ）
# env_spacing: 可視化時の環境間隔（メートル）
#   - 物理には影響しない（各環境は独立）
#   - ビューアで環境を見分けやすくするため
B = 10  # 10個の並列環境

print(f"並列環境数: {B}")
print("シーンをビルド中...")

scene.build(n_envs=B, env_spacing=(2.0, 2.0))

print("ビルド完了！")

########## 関節インデックスの取得 ##########

jnt_names = [
    'joint1', 'joint2', 'joint3', 'joint4',
    'joint5', 'joint6', 'joint7',
    'finger_joint1', 'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

########## デモ 1: 全環境を一括制御 ##########

print("\n--- デモ 1: 全環境を一括制御 ---")
print("全ロボットを同じ姿勢に制御します")

# 目標姿勢（9 DOF）
target_position = torch.tensor(
    [0, 0, 0, -1.0, 0, 1.0, 0, 0.04, 0.04],
    device=gs.device
)

# バッチ次元を追加: (9,) -> (B, 9)
# torch.tile で B 回繰り返す
batched_position = torch.tile(target_position, (B, 1))
print(f"制御テンソルの形状: {batched_position.shape}")

# 全環境を制御
franka.control_dofs_position(batched_position, dofs_idx)

for i in range(200):
    scene.step()

print("全環境が同じ姿勢になりました")

########## デモ 2: 各環境に異なる姿勢を設定 ##########

print("\n--- デモ 2: 各環境に異なる姿勢を設定 ---")
print("各ロボットに異なる姿勢を設定します")

# 各環境ごとに異なる姿勢を作成
# joint1 の角度を環境ごとに変える
different_positions = torch.zeros(B, 9, device=gs.device)

for env_i in range(B):
    # joint1 を環境番号に応じて -1.5 〜 1.5 rad に設定
    joint1_angle = -1.5 + (3.0 * env_i / (B - 1))
    different_positions[env_i] = torch.tensor(
        [joint1_angle, 0.5, 0, -1.5, 0, 1.0, 0, 0.04, 0.04],
        device=gs.device
    )
    print(f"  環境 {env_i}: joint1 = {joint1_angle:.2f} rad")

# 異なる姿勢で制御
franka.control_dofs_position(different_positions, dofs_idx)

for i in range(300):
    scene.step()

print("各環境が異なる姿勢になりました")

########## デモ 3: 特定の環境のみを制御 ##########

print("\n--- デモ 3: 特定の環境のみを制御 ---")
print("環境 0, 2, 4 のみをホームポジションに戻します")

# 制御する環境のインデックス
envs_to_control = torch.tensor([0, 2, 4], device=gs.device)
num_envs_to_control = len(envs_to_control)

# 対象環境用の姿勢テンソル（3環境分）
home_position = torch.zeros(num_envs_to_control, 9, device=gs.device)

# 特定の環境のみ制御
franka.control_dofs_position(
    home_position,
    dofs_idx,
    envs_idx=envs_to_control,
)

for i in range(300):
    scene.step()

print("環境 0, 2, 4 がホームポジションに戻りました")

########## デモ 4: 状態の取得 ##########

print("\n--- デモ 4: 状態の取得 ---")

# 全環境の関節位置を取得
positions = franka.get_dofs_position(dofs_idx)
print(f"取得したテンソルの形状: {positions.shape}")
print(f"  = (環境数, DOF数) = ({B}, 9)")

# 各環境の joint1 の角度を表示
print("\n各環境の joint1 角度:")
for env_i in range(B):
    joint1_pos = positions[env_i, 0].item()
    print(f"  環境 {env_i}: {joint1_pos:.3f} rad")

########## デモ 5: 動的な制御 ##########

print("\n--- デモ 5: 動的な制御 ---")
print("全ロボットを同時に動かします")

for i in range(300):
    # 時間に応じて変化する角度
    angle = torch.sin(torch.tensor(i * 0.05))

    # 全環境に同じ姿勢を設定
    dynamic_position = torch.tensor(
        [angle.item(), 0.5, 0, -1.5, 0, 1.0, 0, 0.04, 0.04],
        device=gs.device
    )
    batched_dynamic = torch.tile(dynamic_position, (B, 1))

    franka.control_dofs_position(batched_dynamic, dofs_idx)
    scene.step()

print("動的制御デモ完了")

########## まとめ ##########

print("\n" + "=" * 50)
print("並列シミュレーションのまとめ")
print("=" * 50)
print(f"""
1. 並列環境の作成:
   scene.build(n_envs={B}, env_spacing=(2.0, 2.0))

2. 全環境を一括制御:
   franka.control_dofs_position(torch.zeros({B}, 9, device=gs.device))

3. 特定環境のみ制御:
   franka.control_dofs_position(
       position=torch.zeros(3, 9, device=gs.device),
       envs_idx=torch.tensor([0, 2, 4], device=gs.device),
   )

4. 状態取得（バッチ次元付き）:
   positions = franka.get_dofs_position(dofs_idx)
   # 形状: ({B}, 9)
""")
print("完了！")
