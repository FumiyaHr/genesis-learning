"""
Hover Evaluation - Genesis チュートリアル 11

訓練済みのホバリングポリシーを評価。

使用方法:
    uv run python hover_eval.py -e drone-hovering --ckpt 49

参照: Genesis/examples/drone/hover_eval.py
"""

import argparse
import os
import pickle
from importlib import metadata

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

from hover_env import HoverEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=49)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    ########## Genesis 初期化 ##########

    gs.init(backend=gs.metal, precision="32")

    ########## 設定読み込み ##########

    log_dir = f"logs/{args.exp_name}"

    # 訓練時の設定を復元
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"エラー: 設定ファイルが見つかりません: {cfg_path}")
        print("先に hover_train.py を実行してください")
        return

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(cfg_path, "rb")
    )

    # 評価時は報酬計算を無効化
    reward_cfg["reward_scales"] = {}

    # ターゲットを表示
    env_cfg["visualize_target"] = True
    env_cfg["max_visualize_FPS"] = 60

    ########## 環境作成 ##########

    print("=" * 60)
    print("Hover Evaluation - Genesis チュートリアル 11")
    print("=" * 60)
    print(f"  実験名: {args.exp_name}")
    print(f"  チェックポイント: {args.ckpt}")
    print(f"  録画: {args.record}")
    print("=" * 60)

    ########## 録画用カメラ設定 ##########

    camera_cfg = None
    if args.record:
        camera_cfg = {
            "res": (1280, 720),
            "pos": (3.0, 0.0, 2.5),
            "lookat": (0.0, 0.0, 1.0),
            "fov": 40,
        }

    env = HoverEnv(
        num_envs=1,  # 評価は1環境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.record,  # 録画時はビューア非表示
        camera_cfg=camera_cfg,
    )

    if args.record:
        env.camera.start_recording()
        max_steps = 500  # 録画は500ステップ（約5秒）
        print(f"\n録画開始... {max_steps}ステップ")

    ########## ポリシー読み込み ##########

    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(resume_path):
        print(f"エラー: モデルが見つかりません: {resume_path}")
        print(f"利用可能なチェックポイントを確認してください: ls {log_dir}/model_*.pt")
        return

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    print(f"\nモデルを読み込みました: {resume_path}")

    ########## 評価ループ ##########

    obs, _ = env.reset()

    if args.record:
        # 録画モード（指定ステップ数で終了）
        with torch.no_grad():
            for step in range(max_steps):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.camera.render()

        # 録画保存
        output_path = "hover_eval.mp4"
        env.camera.stop_recording(save_to_filename=output_path, fps=60)
        print(f"\n録画を保存しました: {output_path}")
    else:
        # インタラクティブモード（手動終了）
        print("\n評価を開始...")
        print("ビューアを閉じるか Ctrl+C で終了")

        with torch.no_grad():
            try:
                while True:
                    actions = policy(obs)
                    obs, rews, dones, infos = env.step(actions)
            except KeyboardInterrupt:
                print("\n終了しました")


if __name__ == "__main__":
    main()
