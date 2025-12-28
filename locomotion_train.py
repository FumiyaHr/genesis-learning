"""
Locomotion Training - Genesis チュートリアル 10

Go2 四足歩行ロボットの歩行ポリシーを PPO で訓練。

使用方法:
    uv run python locomotion_train.py

TensorBoard でモニタリング:
    uv run tensorboard --logdir logs

注意:
    - macOS では Metal バックエンドを使用
    - 公式は 4096 環境だが、macOS では 16 環境に削減
    - 訓練時間は公式より大幅に長くなる

参照: Genesis/examples/locomotion/go2_train.py
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

from locomotion_env import Go2Env


def get_train_cfg(exp_name, max_iterations):
    """
    訓練設定を取得

    PPO (Proximal Policy Optimization) のハイパーパラメータ:
    - clip_param: ポリシー更新のクリッピング範囲
    - gamma: 割引率（将来報酬の重み）
    - lam: GAE のλパラメータ
    - learning_rate: 学習率
    - num_learning_epochs: 1回の更新あたりの学習エポック数
    - num_mini_batches: ミニバッチ数
    """
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],  # アクターネットワーク
            "critic_hidden_dims": [512, 256, 128],  # クリティックネットワーク
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
        "num_steps_per_env": 24,  # 1環境あたりのステップ数
        "save_interval": 50,  # モデル保存間隔
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """
    環境設定を取得

    env_cfg: 環境パラメータ
    - default_joint_angles: デフォルト関節角度 [rad]
    - joint_names: 関節名のリスト
    - kp, kd: PD コントローラのゲイン
    - termination_if_*: 転倒判定の閾値

    obs_cfg: 観測パラメータ
    - num_obs: 観測次元（45 = 3+3+3+12+12+12）
    - obs_scales: 観測値のスケーリング

    reward_cfg: 報酬パラメータ
    - tracking_sigma: 速度追従報酬の感度
    - reward_scales: 各報酬関数の重み

    command_cfg: コマンドパラメータ
    - lin_vel_x_range: 前進速度の範囲 [m/s]
    - lin_vel_y_range: 横移動速度の範囲 [m/s]
    - ang_vel_range: 旋回速度の範囲 [rad/s]
    """
    env_cfg = {
        "num_actions": 12,  # Go2 の12関節
        # デフォルト関節角度 [rad]
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        # 関節名（制御順序）
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD コントローラゲイン
        "kp": 20.0,
        "kd": 0.5,
        # 転倒判定閾値 [degree]
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        # 初期姿勢
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # エピソード設定
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }

    obs_cfg = {
        "num_obs": 45,  # 観測次元
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,    # 前進速度追従（正）
            "tracking_ang_vel": 0.2,    # 旋回速度追従（正）
            "lin_vel_z": -1.0,          # 上下動ペナルティ（負）
            "base_height": -50.0,       # 高さペナルティ（負）
            "action_rate": -0.005,      # アクション変化ペナルティ（負）
            "similar_to_default": -0.1,  # 姿勢ペナルティ（負）
        },
    }

    command_cfg = {
        "num_commands": 3,
        # 前進速度 0.5 m/s で固定（訓練を安定化）
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    # macOS では環境数を削減（公式は 4096）
    parser.add_argument("-B", "--num_envs", type=int, default=16)
    # macOS ではイテレーション数を削減（公式は 101）
    parser.add_argument("--max_iterations", type=int, default=50)
    args = parser.parse_args()

    ########## Genesis 初期化 ##########

    # macOS では Metal バックエンド
    # performance_mode=True でパフォーマンス優先
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
    print("Locomotion Training - Genesis チュートリアル 10")
    print("=" * 60)
    print(f"  実験名: {args.exp_name}")
    print(f"  並列環境数: {args.num_envs}")
    print(f"  最大イテレーション: {args.max_iterations}")
    print(f"  ログディレクトリ: {log_dir}")
    print("=" * 60)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
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
