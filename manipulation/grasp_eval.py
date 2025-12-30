"""
Grasp Evaluation - Genesis チュートリアル 12

訓練済み把持ポリシーの評価とビデオ録画。

## 使用方法

### RL ポリシー（Stage 1）の評価
```bash
uv run python grasp_eval.py --stage rl
```

### IL ポリシー（Stage 2）の評価
```bash
uv run python grasp_eval.py --stage il
```

### ビデオ録画付き評価
```bash
uv run python grasp_eval.py --stage rl --record
```

## 出力ファイル（--record 時）
- video.mp4: 可視化カメラの映像
- left_cam.mp4: 左ステレオカメラの映像
- right_cam.mp4: 右ステレオカメラの映像

参照: Genesis/examples/manipulation/grasp_eval.py
"""

import argparse
import pickle
import re
from importlib import metadata
from pathlib import Path

import torch

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


########## ポリシー読み込み ##########


def load_rl_policy(env, train_cfg: dict, log_dir: Path):
    """
    RL ポリシー（Stage 1）を読み込む

    Args:
        env: 環境
        train_cfg: 訓練設定
        log_dir: ログディレクトリ

    Returns:
        policy: 推論用ポリシー
    """
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # 最新のチェックポイントを探す
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"チェックポイントが見つかりません: {log_dir}")

    last_ckpt = sorted(checkpoint_files)[-1]
    runner.load(last_ckpt)
    print(f"RL ポリシーを読み込みました: {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_il_policy(env, il_cfg: dict, log_dir: Path):
    """
    IL ポリシー（Stage 2）を読み込む

    Args:
        env: 環境
        il_cfg: 模倣学習設定
        log_dir: ログディレクトリ

    Returns:
        policy: 推論用ポリシー
    """
    from imitation_learning import ImitationLearning

    # ImitationLearning インスタンスを作成（教師なしで読み込み用）
    il_runner = ImitationLearning(env, il_cfg, None, device=gs.device)

    # 最新のチェックポイントを探す（checkpoint_final.pt を優先）
    final_ckpt = log_dir / "checkpoint_final.pt"
    if final_ckpt.exists():
        last_ckpt = final_ckpt
    else:
        checkpoint_files = [
            f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)
        ]
        if not checkpoint_files:
            raise FileNotFoundError(f"チェックポイントが見つかりません: {log_dir}")
        last_ckpt = sorted(checkpoint_files)[-1]
    il_runner.load(last_ckpt)
    print(f"IL ポリシーを読み込みました: {last_ckpt}")

    return il_runner._policy


########## メイン ##########


def main():
    parser = argparse.ArgumentParser(
        description="Grasp Policy Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # RL ポリシー評価
  uv run python grasp_eval.py --stage rl

  # IL ポリシー評価
  uv run python grasp_eval.py --stage il

  # ビデオ録画付き
  uv run python grasp_eval.py --stage rl --record
        """,
    )
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default="grasp",
        help="実験名",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "il"],
        help="評価するポリシー: rl=強化学習, il=模倣学習",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="ビデオを録画する",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=-1,
        help="使用するチェックポイント番号（-1で最新）",
    )
    args = parser.parse_args()

    # PyTorch のデフォルト型を float32 に設定
    torch.set_default_dtype(torch.float32)

    # Genesis 初期化
    gs.init(backend=gs.metal)  # macOS

    # === ログディレクトリ ===
    log_dir = Path("logs") / f"{args.exp_name}_{args.stage}"

    if not log_dir.exists():
        raise FileNotFoundError(
            f"ログディレクトリが見つかりません: {log_dir}\n"
            f"先に訓練を実行してください:\n"
            f"  uv run python grasp_train.py --stage {args.stage}"
        )

    # === 設定の読み込み ===
    cfg_path = log_dir / "cfgs.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {cfg_path}")

    with open(cfg_path, "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, il_train_cfg = pickle.load(f)

    # === 評価用の設定変更 ===
    env_cfg["max_visualize_FPS"] = 60  # 可視化FPS
    env_cfg["box_collision"] = True  # 衝突判定を有効化
    env_cfg["box_fixed"] = False  # 箱を動くようにする
    env_cfg["num_envs"] = 10  # 評価用環境数
    env_cfg["visualize_camera"] = args.record  # 録画時はカメラ有効化
    env_cfg["robot_cfg"] = robot_cfg

    # === 環境作成 ===
    print("環境を作成中...")
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )

    # === ポリシー読み込み ===
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_il_policy(env, il_train_cfg, log_dir)
        policy.eval()  # 評価モードに設定

    # === 評価ループ ===
    obs, _ = env.reset()
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    print(f"\n=== 評価開始 ===")
    print(f"  ステージ: {args.stage.upper()}")
    print(f"  ステップ数: {max_sim_step}")
    print(f"  録画: {'有効' if args.record else '無効'}")
    print()

    with torch.no_grad():
        # 録画開始
        if args.record:
            print("録画を開始...")
            env.vis_cam.start_recording()
            env.left_cam.start_recording()
            env.right_cam.start_recording()

        # シミュレーションループ
        for step in range(max_sim_step):
            if args.stage == "rl":
                # RL: 特権状態からアクションを予測
                actions = policy(obs)
            else:
                # IL: ステレオ画像からアクションを予測
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                actions = policy(rgb_obs, ee_pose)

                # 録画時はカメラをレンダリング
                if args.record:
                    env.vis_cam.render()

            obs, rews, dones, infos = env.step(actions)

            # 進捗表示
            if (step + 1) % 30 == 0:
                print(f"  ステップ: {step + 1}/{max_sim_step}")

        # 把持と持ち上げのデモ
        print("\n把持と持ち上げデモを実行...")
        env.grasp_and_lift_demo()

        # 録画停止
        if args.record:
            print("\n録画を停止・保存...")
            env.vis_cam.stop_recording(
                save_to_filename="video.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
            env.left_cam.stop_recording(
                save_to_filename="left_cam.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
            env.right_cam.stop_recording(
                save_to_filename="right_cam.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
            print("保存完了: video.mp4, left_cam.mp4, right_cam.mp4")

    print("\n評価完了!")


if __name__ == "__main__":
    main()
