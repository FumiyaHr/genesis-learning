"""
Hover Training - Genesis チュートリアル 11

ドローンのホバリングポリシーを PPO で訓練。

使用方法:
    uv run python hover_train.py

TensorBoard でモニタリング:
    uv run tensorboard --logdir logs

注意:
    - macOS では Metal バックエンドを使用
    - 公式は 8192 環境だが、macOS では 16 環境に削減
    - 訓練時間は公式より大幅に長くなる

参照: Genesis/examples/drone/hover_train.py
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata

########## rsl-rl-lib のバージョン確認 ##########

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from hover_env import HoverEnv


def get_train_cfg(exp_name, max_iterations):
    """
    訓練設定を取得

    PPO のハイパーパラメータ:
    - clip_param: ポリシー更新のクリッピング範囲
    - gamma: 割引率（将来報酬の重み）
    - learning_rate: 学習率（locomotion より小さい）
    - entropy_coef: エントロピー係数（locomotion より小さい）
    """
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,  # locomotion は 0.01
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,  # locomotion は 0.001
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",  # locomotion は elu
            "actor_hidden_dims": [128, 128],  # locomotion は [512, 256, 128]
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,  # locomotion は 24
        "save_interval": 50,  # macOS 向け（公式は 100）
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """
    環境設定を取得

    env_cfg: 環境パラメータ
    - num_actions: アクション次元（4つのプロペラ）
    - termination_if_*: 終了条件（墜落判定）
    - base_init_pos: 初期位置（高さ1m）

    obs_cfg: 観測パラメータ
    - num_obs: 観測次元（17 = 3+4+3+3+4）
    - obs_scales: 観測値のスケーリング

    reward_cfg: 報酬パラメータ
    - reward_scales: 各報酬関数の重み

    command_cfg: コマンドパラメータ
    - pos_*_range: ターゲット位置の範囲
    """
    env_cfg = {
        "num_actions": 4,  # 4つのプロペラ
        # 終了条件
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,  # 地面に近づきすぎ
        "termination_if_x_greater_than": 3.0,  # X方向に離れすぎ
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        # 初期姿勢
        "base_init_pos": [0.0, 0.0, 1.0],  # 高さ1m
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # 水平
        # エピソード設定
        "episode_length_s": 15.0,  # 15秒
        "at_target_threshold": 0.1,  # 10cmでターゲット到達判定
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # 可視化設定
        "visualize_target": True,  # ターゲットを表示
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }

    obs_cfg = {
        "num_obs": 17,  # 観測次元
        "obs_scales": {
            "rel_pos": 1 / 3.0,  # 相対位置（3m範囲を想定）
            "lin_vel": 1 / 3.0,  # 線速度
            "ang_vel": 1 / 3.14159,  # 角速度（ラジアン）
        },
    }

    reward_cfg = {
        "yaw_lambda": -10.0,  # ヨー報酬の減衰係数
        "reward_scales": {
            "target": 10.0,  # ターゲット到達（正）
            "smooth": -1e-4,  # アクション変化ペナルティ（負）
            "yaw": 0.01,  # ヨー安定化（正）
            "angular": -2e-4,  # 角速度ペナルティ（負）
            "crash": -10.0,  # 墜落ペナルティ（負）
        },
    }

    command_cfg = {
        "num_commands": 3,  # ターゲット位置 (x, y, z)
        "pos_x_range": [-1.0, 1.0],  # X: ±1m
        "pos_y_range": [-1.0, 1.0],  # Y: ±1m
        "pos_z_range": [1.0, 1.0],  # Z: 固定1m
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    # macOS では環境数を削減（公式は 8192）
    parser.add_argument("-B", "--num_envs", type=int, default=16)
    # macOS ではイテレーション数を削減（公式は 301）
    parser.add_argument("--max_iterations", type=int, default=50)
    args = parser.parse_args()

    ########## Genesis 初期化 ##########

    # macOS では Metal バックエンド
    gs.init(
        backend=gs.metal,
        precision="32",
        logging_level="warning",
        performance_mode=True,
    )

    ########## ログディレクトリ ##########

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # 既存のログを削除
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 設定を保存（評価時に使用）
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    ########## 環境作成 ##########

    print("=" * 60)
    print("Hover Training - Genesis チュートリアル 11")
    print("=" * 60)
    print(f"  実験名: {args.exp_name}")
    print(f"  並列環境数: {args.num_envs}")
    print(f"  最大イテレーション: {args.max_iterations}")
    print(f"  ログディレクトリ: {log_dir}")
    print("=" * 60)

    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        # show_viewer=True,  # 訓練中の様子を表示する場合はコメント解除
    )

    ########## 訓練 ##########

    print("\n訓練を開始...")
    print("TensorBoard でモニタリング: uv run tensorboard --logdir logs")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("\n訓練完了!")
    print(f"モデルは {log_dir} に保存されました")


if __name__ == "__main__":
    main()
