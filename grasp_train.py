"""
Grasp Training - Genesis チュートリアル 12

Franka Panda ロボットアームの把持ポリシーを PPO で訓練。

## 使用方法

### RL訓練（Stage 1: 教師ポリシー）
```bash
uv run python grasp_train.py --stage rl
```

### IL訓練（Stage 2: 学生ポリシー）※要 Stage 1 完了
```bash
uv run python grasp_train.py --stage il
```

### TensorBoard でモニタリング
```bash
uv run tensorboard --logdir logs
```

## 注意
- macOS では Metal バックエンドを使用
- 公式は 2048 環境だが、macOS では 64 環境に削減
- 訓練時間は公式より大幅に長くなる

参照: Genesis/examples/manipulation/grasp_train.py
"""

import argparse
import os
import pickle
import re
from importlib import metadata
from pathlib import Path

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

from grasp_env import GraspEnv


########## 設定 ##########


def get_train_cfg(exp_name: str, max_iterations: int) -> tuple[dict, dict]:
    """
    訓練設定を取得

    Returns:
        rl_cfg: RL（PPO）訓練設定
        il_cfg: IL（模倣学習）訓練設定
    """
    # Stage 1: PPO による強化学習
    rl_cfg = {
        "algorithm": {
            "class_name": "PPO",
            # PPO パラメータ
            "clip_param": 0.2,  # クリッピング範囲
            "desired_kl": 0.01,  # 目標KLダイバージェンス
            "entropy_coef": 0.0,  # エントロピー係数（探索促進）
            "gamma": 0.99,  # 割引率
            "lam": 0.95,  # GAE λ
            "learning_rate": 0.0003,  # 学習率（locomotion: 0.001）
            "max_grad_norm": 1.0,  # 勾配クリッピング
            "num_learning_epochs": 5,  # 1更新あたりの学習エポック
            "num_mini_batches": 4,  # ミニバッチ数
            "schedule": "adaptive",  # 学習率スケジュール
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "relu",
            # ネットワーク構造（locomotion より小さい）
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
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

    # Stage 2: 模倣学習（Behavior Cloning）
    il_cfg = {
        # 基本パラメータ
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        # ネットワーク構造
        "policy": {
            # 視覚エンコーダ（CNN）
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
                "pooling": "adaptive_avg",
            },
            # アクション予測ヘッド
            "action_head": {
                "state_obs_dim": 7,  # エンドエフェクタ姿勢
                "hidden_dims": [128, 128, 64],
            },
            # 姿勢予測ヘッド（補助タスク）
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        # 訓練設定
        "buffer_size": 1000,  # 経験バッファサイズ
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
    }

    return rl_cfg, il_cfg


def get_task_cfgs() -> tuple[dict, dict, dict]:
    """
    タスク設定を取得

    Returns:
        env_cfg: 環境設定
        reward_scales: 報酬スケール
        robot_cfg: ロボット設定
    """
    env_cfg = {
        "num_envs": 10,  # 環境数（引数で上書き）
        "num_obs": 14,  # 観測次元
        "num_actions": 6,  # アクション次元（6DoF）
        # アクションスケール: 各軸の最大変化量
        # Note: 小さくすると安定するが、動きが遅くなる
        "action_scales": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        "episode_length_s": 3.0,  # エピソード長（秒）
        "ctrl_dt": 0.01,  # 制御周期（秒）
        # 把持対象の箱
        "box_size": [0.08, 0.03, 0.06],  # サイズ [x, y, z]
        "box_collision": False,  # 衝突判定（訓練時はオフ）
        "box_fixed": True,  # 固定（動かない）
        # カメラ設定
        "image_resolution": (64, 64),
        "use_rasterizer": True,
        "visualize_camera": False,
    }

    # 報酬スケール
    reward_scales = {
        "keypoints": 1.0,  # キーポイント報酬
    }

    # Franka Panda ロボット設定
    robot_cfg = {
        "ee_link_name": "hand",  # エンドエフェクタリンク名
        "gripper_link_names": ["left_finger", "right_finger"],
        # デフォルト関節角度（ホームポジション）
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],  # 開いた状態
        "ik_method": "dls_ik",  # 逆運動学手法（DLS法）
    }

    return env_cfg, reward_scales, robot_cfg


def load_teacher_policy(env, rl_train_cfg: dict, exp_name: str):
    """
    Stage 1 で訓練した教師ポリシーを読み込む

    Args:
        env: 環境
        rl_train_cfg: RL訓練設定
        exp_name: 実験名

    Returns:
        teacher_policy: 教師ポリシー（推論用）
    """
    log_dir = Path("logs") / f"{exp_name}_rl"

    if not log_dir.exists():
        raise FileNotFoundError(
            f"教師ポリシーが見つかりません: {log_dir}\n"
            "先に Stage 1（RL訓練）を実行してください:\n"
            "  uv run python grasp_train.py --stage rl"
        )

    # 最新のチェックポイントを探す
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {log_dir}"
        )

    # 最後のチェックポイントを使用
    last_ckpt = sorted(checkpoint_files)[-1]

    # ポリシーの読み込み
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt)
    print(f"教師ポリシーを読み込みました: {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


########## メイン ##########


def main():
    parser = argparse.ArgumentParser(
        description="Grasp Policy Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # RL訓練（Stage 1）
  uv run python grasp_train.py --stage rl

  # IL訓練（Stage 2）
  uv run python grasp_train.py --stage il

  # カスタム設定
  uv run python grasp_train.py --stage rl --num_envs 32 --max_iterations 200
        """,
    )
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default="grasp",
        help="実験名（ログディレクトリ名に使用）",
    )
    parser.add_argument(
        "-v", "--vis",
        action="store_true",
        default=False,
        help="ビューワを表示",
    )
    parser.add_argument(
        "-B", "--num_envs",
        type=int,
        default=64,  # macOS 向けにデフォルトを削減（公式: 2048）
        help="並列環境数（macOS: 64推奨、Linux+GPU: 2048）",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=150,  # macOS 向けにデフォルトを削減（公式: 300）
        help="最大イテレーション数",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "il"],
        help="訓練ステージ: rl=強化学習, il=模倣学習",
    )
    args = parser.parse_args()

    # === Genesis 初期化 ===
    # macOS では Metal バックエンドを使用
    gs.init(
        backend=gs.metal,  # macOS: metal, Linux: gpu
        precision="32",
        logging_level="warning",
    )

    # === 設定の読み込み ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg, il_train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # ロボット設定を環境設定に追加
    env_cfg["robot_cfg"] = robot_cfg

    # === ログディレクトリ ===
    log_dir = Path("logs") / f"{args.exp_name}_{args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 設定を保存
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump(
            (env_cfg, reward_scales, robot_cfg, rl_train_cfg, il_train_cfg), f
        )

    # === 環境作成 ===
    # IL は少数の環境で十分（教師から学習するため）
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10

    print(f"環境を作成中... (num_envs={env_cfg['num_envs']})")
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === 訓練 ===
    if args.stage == "il":
        # Stage 2: 模倣学習
        # 教師ポリシーの読み込み
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        il_train_cfg["teacher_policy"] = teacher_policy

        # 模倣学習モジュールのインポート（Stage 2 で使用）
        from imitation_learning import ImitationLearning

        runner = ImitationLearning(
            env, il_train_cfg, teacher_policy, device=gs.device
        )
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=log_dir)
    else:
        # Stage 1: 強化学習（PPO）
        print(f"\n=== Stage 1: RL訓練 ===")
        print(f"  環境数: {env_cfg['num_envs']}")
        print(f"  イテレーション: {args.max_iterations}")
        print(f"  ログ: {log_dir}")
        print()

        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )

    print(f"\n訓練完了! ログ: {log_dir}")


if __name__ == "__main__":
    main()
