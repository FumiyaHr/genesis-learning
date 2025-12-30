"""
Locomotion Evaluation - Genesis チュートリアル 10

訓練済みの歩行ポリシーを評価。

使用方法:
    uv run python locomotion_eval.py -e go2-walking --ckpt 50

参照: Genesis/examples/locomotion/go2_eval.py
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

from locomotion_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=50)
    args = parser.parse_args()

    ########## Genesis 初期化 ##########

    # 評価時は Metal バックエンド（ビューア表示）
    gs.init(backend=gs.metal)

    ########## 設定を読み込み ##########

    log_dir = f"logs/{args.exp_name}"
    cfg_path = f"{log_dir}/cfgs.pkl"

    if not os.path.exists(cfg_path):
        print(f"エラー: 設定ファイルが見つかりません: {cfg_path}")
        print(f"先に locomotion_train.py を実行してください")
        return

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))

    # 評価時は報酬計算を無効化
    reward_cfg["reward_scales"] = {}

    ########## 環境作成 ##########

    print("=" * 60)
    print("Locomotion Evaluation - Genesis チュートリアル 10")
    print("=" * 60)
    print(f"  実験名: {args.exp_name}")
    print(f"  チェックポイント: {args.ckpt}")
    print("=" * 60)

    env = Go2Env(
        num_envs=1,  # 評価時は1環境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,  # ビューア表示
    )

    ########## ポリシー読み込み ##########

    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(resume_path):
        print(f"エラー: モデルが見つかりません: {resume_path}")
        print(f"利用可能なモデル:")
        for f in os.listdir(log_dir):
            if f.startswith("model_") and f.endswith(".pt"):
                print(f"  - {f}")
        return

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    print(f"\nモデルを読み込みました: {resume_path}")
    print("\n評価を開始...")
    print("ビューアを閉じるか Ctrl+C で終了")

    ########## 評価ループ ##########

    obs, _ = env.reset()
    with torch.no_grad():
        try:
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
        except KeyboardInterrupt:
            print("\n評価を終了しました")


if __name__ == "__main__":
    main()
