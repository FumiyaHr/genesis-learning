"""
Imitation Learning (模倣学習) - Genesis チュートリアル 13

教師ポリシーから学生ポリシーへの知識蒸留。

## 概要

模倣学習（Imitation Learning, IL）は、教師の行動を真似することで学習する手法。
本実装では Behavior Cloning という具体的な手法を使用する。

## 主要コンポーネント

1. Policy: CNN + MLP による視覚ベースポリシー
2. ExperienceBuffer: 教師の経験を保存するバッファ
3. ImitationLearning: 学習を管理するクラス

## ネットワーク構造

```
ステレオ画像 (左3ch + 右3ch)
     ↓
  共有 CNN エンコーダ
     ↓
  特徴融合（左 + 右）
     ↓
  ┌────┴────┐
  ↓         ↓
行動ヘッド   姿勢ヘッド（補助タスク）
  ↓         ↓
6DoF行動    物体姿勢
```

## DAgger スタイル学習

分布シフト問題を軽減するため、学生のアクションを使って環境を進め、
教師のアクションを正解として学習する。

参照: Genesis/examples/manipulation/behavior_cloning.py
"""

import os
import time
from collections import deque
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


########## ImitationLearning: 模倣学習クラス ##########


class ImitationLearning:
    """
    模倣学習（Imitation Learning）クラス

    教師ポリシーの行動を模倣して、視覚ベースの学生ポリシーを訓練する。

    マルチタスク学習:
    1. アクション予測（メインタスク）: 教師のアクションを予測
    2. 姿勢予測（補助タスク）: 物体の姿勢を予測（視覚特徴の質を向上）
    """

    def __init__(
        self,
        env,
        cfg: dict,
        teacher: nn.Module,
        device: str = "cpu",
    ):
        """
        初期化

        Args:
            env: 環境（GraspEnv）
            cfg: 設定
            teacher: 教師ポリシー（Stage 1 で訓練済み）
            device: デバイス
        """
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher
        self._num_steps_per_env = cfg["num_steps_per_env"]

        # ステレオRGB画像の形状: 6チャネル（左3ch + 右3ch）
        rgb_shape = (6, env.image_height, env.image_width)
        action_dim = env.num_actions

        # マルチタスクポリシーの作成
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # オプティマイザ
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(), lr=cfg["learning_rate"]
        )

        # 経験バッファ
        self._buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=cfg["buffer_size"],
            img_shape=rgb_shape,
            state_dim=cfg["policy"]["action_head"]["state_obs_dim"],
            action_dim=action_dim,
            device=device,
            dtype=self._policy.dtype,
        )

        # 訓練状態
        self._current_iter = 0

    def learn(self, num_learning_iterations: int, log_dir: str) -> None:
        """
        学習ループ

        Args:
            num_learning_iterations: 学習イテレーション数
            log_dir: ログディレクトリ
        """
        # 報酬記録用バッファ
        self._rewbuffer = deque(maxlen=100)
        self._cur_reward_sum = torch.zeros(
            self._env.num_envs, dtype=torch.float, device=self._device
        )
        self._buffer.clear()

        # TensorBoard ライター
        tf_writer = SummaryWriter(log_dir)

        print(f"\n=== Stage 2: IL訓練（模倣学習） ===")
        print(f"  イテレーション: {num_learning_iterations}")
        print(f"  ログ: {log_dir}")
        print()

        for it in range(num_learning_iterations):
            # === 経験収集フェーズ ===
            start_time = time.time()
            self._collect_with_teacher()
            forward_time = time.time() - start_time

            # === 学習フェーズ ===
            total_action_loss = 0.0
            total_pose_loss = 0.0
            num_batches = 0

            start_time = time.time()
            generator = self._buffer.get_batches(
                self._cfg.get("num_mini_batches", 4), self._cfg["num_epochs"]
            )

            for batch in generator:
                # フォワードパス
                pred_action = self._policy(batch["rgb_obs"], batch["robot_pose"])
                pred_left_pose, pred_right_pose = self._policy.predict_pose(
                    batch["rgb_obs"]
                )

                # アクション予測の損失（メインタスク）
                action_loss = F.mse_loss(pred_action, batch["actions"])

                # 姿勢予測の損失（補助タスク）
                pose_left_loss = self._compute_pose_loss(
                    pred_left_pose, batch["object_poses"]
                )
                pose_right_loss = self._compute_pose_loss(
                    pred_right_pose, batch["object_poses"]
                )
                pose_loss = pose_left_loss + pose_right_loss

                # 合計損失
                total_loss = action_loss + pose_loss

                # バックワードパス
                self._optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self._cfg["max_grad_norm"]
                )
                self._optimizer.step()

                total_action_loss += action_loss.item()
                total_pose_loss += pose_loss.item()
                num_batches += 1

            backward_time = time.time() - start_time

            # 平均損失の計算
            if num_batches == 0:
                raise ValueError("バッチが収集されませんでした")
            avg_action_loss = total_action_loss / num_batches
            avg_pose_loss = total_pose_loss / num_batches

            fps = (self._num_steps_per_env * self._env.num_envs) / forward_time

            # === ログ出力 ===
            if (it + 1) % self._cfg["log_freq"] == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]

                # TensorBoard に記録
                tf_writer.add_scalar("loss/action_loss", avg_action_loss, it)
                tf_writer.add_scalar("loss/pose_loss", avg_pose_loss, it)
                tf_writer.add_scalar(
                    "loss/total_loss", avg_action_loss + avg_pose_loss, it
                )
                tf_writer.add_scalar("lr", current_lr, it)
                tf_writer.add_scalar("buffer_size", self._buffer.size, it)
                tf_writer.add_scalar("speed/fps", int(fps), it)

                if len(self._rewbuffer) > 0:
                    tf_writer.add_scalar("reward/mean", np.mean(self._rewbuffer), it)

                # コンソール出力
                print("--------------------------------")
                print(f" | Iteration:     {it + 1:04d}")
                print(f" | Action Loss:   {avg_action_loss:.6f}")
                print(f" | Pose Loss:     {avg_pose_loss:.6f}")
                print(f" | Total Loss:    {avg_action_loss + avg_pose_loss:.6f}")
                print(f" | Learning Rate: {current_lr:.6f}")
                print(f" | FPS:           {int(fps)}")
                if len(self._rewbuffer) > 0:
                    print(f" | Mean Reward:   {np.mean(self._rewbuffer):.4f}")

            # === チェックポイント保存 ===
            if (it + 1) % self._cfg["save_freq"] == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it + 1:04d}.pt"))

        # 最終チェックポイントを保存
        self.save(os.path.join(log_dir, f"checkpoint_final.pt"))
        tf_writer.close()
        print("\nIL訓練完了!")

    def _compute_pose_loss(
        self, pred_poses: torch.Tensor, target_poses: torch.Tensor
    ) -> torch.Tensor:
        """
        姿勢予測の損失を計算

        損失 = 位置のMSE + 四元数の距離

        Args:
            pred_poses: 予測姿勢 [batch, 7] (x, y, z, qw, qx, qy, qz)
            target_poses: 正解姿勢 [batch, 7]

        Returns:
            loss: 姿勢損失
        """
        # 位置と姿勢に分割
        pred_pos = pred_poses[:, :3]
        pred_quat = pred_poses[:, 3:7]
        target_pos = target_poses[:, :3]
        target_quat = target_poses[:, 3:7]

        # 位置の損失（MSE）
        pos_loss = F.mse_loss(pred_pos, target_pos)

        # 姿勢の損失（四元数の距離）
        # 四元数を正規化
        pred_quat = F.normalize(pred_quat, p=2, dim=1)
        target_quat = F.normalize(target_quat, p=2, dim=1)

        # 四元数の距離: 1 - |dot(q1, q2)|
        # dot が 1 に近いほど同じ姿勢、0 に近いほど90度ずれている
        quat_dot = torch.sum(pred_quat * target_quat, dim=1)
        quat_loss = torch.mean(1.0 - torch.abs(quat_dot))

        return pos_loss + quat_loss

    def _collect_with_teacher(self) -> None:
        """
        教師ポリシーを使って経験を収集

        DAgger スタイルの学習:
        1. 学生がアクションを予測
        2. 教師も「正解」アクションを計算
        3. 学生と教師のアクションの差が小さければ学生のアクションを使用
        4. 差が大きければ教師のアクションを使用（補正）
        5. どちらを使っても、教師のアクションを正解としてバッファに保存
        """
        obs, _ = self._env.get_observations()

        with torch.inference_mode():
            for _ in range(self._num_steps_per_env):
                # ステレオRGB画像を取得
                rgb_obs = self._env.get_stereo_rgb_images(normalize=True)

                # 教師のアクション（正解）
                teacher_action = self._teacher(obs).detach()

                # エンドエフェクタの姿勢
                ee_pose = self._env.robot.ee_pose

                # 物体の姿勢（補助タスクの正解）
                object_pose = torch.cat(
                    [
                        self._env.object.get_pos(),
                        self._env.object.get_quat(),
                    ],
                    dim=-1,
                )

                # バッファに保存
                self._buffer.add(rgb_obs, ee_pose, object_pose, teacher_action)

                # 学生のアクション
                student_action = self._policy(rgb_obs.float(), ee_pose.float())

                # DAgger: 学生と教師のアクションの差を計算
                action_diff = torch.norm(student_action - teacher_action, dim=-1)

                # 差が1.0未満なら学生のアクション、そうでなければ教師のアクション
                condition = (action_diff < 1.0).unsqueeze(-1).expand_as(student_action)
                action = torch.where(condition, student_action, teacher_action)

                # 環境をステップ
                next_obs, reward, done, _ = self._env.step(action)
                self._cur_reward_sum += reward

                obs = next_obs

                # 終了した環境の報酬を記録
                new_ids = (done > 0).nonzero(as_tuple=False)
                self._rewbuffer.extend(
                    self._cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                )
                self._cur_reward_sum[new_ids] = 0

    def save(self, path: str) -> None:
        """チェックポイントを保存"""
        checkpoint = {
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self._current_iter,
            "config": self._cfg,
        }
        torch.save(checkpoint, path)
        print(f"モデルを保存しました: {path}")

    def load(self, path: str) -> None:
        """チェックポイントを読み込み"""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["current_iter"]
        print(f"モデルを読み込みました: {path}")


########## ExperienceBuffer: 経験バッファ ##########


class ExperienceBuffer:
    """
    経験バッファ（FIFO方式）

    教師ポリシーから収集した経験を保存し、
    ミニバッチとして取り出す。

    保存するデータ:
    - RGB画像（ステレオ）
    - ロボットの姿勢
    - 物体の姿勢
    - 教師のアクション
    """

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ):
        """
        初期化

        Args:
            num_envs: 環境数
            max_size: バッファの最大サイズ
            img_shape: 画像の形状 (channels, height, width)
            state_dim: 状態の次元（エンドエフェクタ姿勢）
            action_dim: アクションの次元
            device: デバイス
            dtype: データ型
        """
        self._num_envs = num_envs
        self._max_size = max_size
        self._img_shape = img_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._ptr = 0  # 書き込み位置
        self._size = 0  # 現在のサイズ

        # データバッファを確保
        self._rgb_obs = torch.empty(
            max_size, num_envs, *img_shape, dtype=dtype, device=device
        )
        self._robot_pose = torch.empty(
            max_size, num_envs, state_dim, dtype=dtype, device=device
        )
        self._object_poses = torch.empty(
            max_size, num_envs, 7, dtype=dtype, device=device
        )
        self._actions = torch.empty(
            max_size, num_envs, action_dim, dtype=dtype, device=device
        )

    def add(
        self,
        rgb_obs: torch.Tensor,
        robot_pose: torch.Tensor,
        object_poses: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """経験を追加（FIFO方式で古いデータを上書き）"""
        self._ptr = (self._ptr + 1) % self._max_size
        self._rgb_obs[self._ptr] = rgb_obs
        self._robot_pose[self._ptr] = robot_pose
        self._object_poses[self._ptr] = object_poses
        self._actions[self._ptr] = actions
        self._size = min(self._size + 1, self._max_size)

    def get_batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        ミニバッチを生成

        Args:
            num_mini_batches: ミニバッチ数
            num_epochs: エポック数

        Yields:
            batch: ミニバッチ（辞書形式）
        """
        batch_size = self._size // num_mini_batches

        for _ in range(num_epochs):
            # ランダムにシャッフル
            indices = torch.randperm(self._size)

            for batch_idx in range(0, self._size, batch_size):
                batch_indices = indices[batch_idx : batch_idx + batch_size]

                yield {
                    "rgb_obs": self._rgb_obs[batch_indices].reshape(
                        -1, *self._img_shape
                    ),
                    "robot_pose": self._robot_pose[batch_indices].reshape(
                        -1, self._state_dim
                    ),
                    "object_poses": self._object_poses[batch_indices].reshape(-1, 7),
                    "actions": self._actions[batch_indices].reshape(
                        -1, self._action_dim
                    ),
                }

    def clear(self) -> None:
        """バッファをクリア"""
        self._rgb_obs.zero_()
        self._robot_pose.zero_()
        self._object_poses.zero_()
        self._actions.zero_()
        self._ptr = 0
        self._size = 0

    def is_full(self) -> bool:
        """バッファが満杯かどうか"""
        return self._size == self._max_size

    @property
    def size(self) -> int:
        """現在のバッファサイズ"""
        return self._size


########## Policy: 視覚ベースポリシー ##########


class Policy(nn.Module):
    """
    視覚ベースポリシー

    ステレオカメラからの画像を入力とし、アクションを出力する。

    構造:
    1. 共有CNNエンコーダ（左右カメラで共有）
    2. 特徴融合層（左右の特徴を統合）
    3. アクションヘッド（メインタスク）
    4. 姿勢ヘッド（補助タスク）
    """

    def __init__(self, config: dict, action_dim: int):
        """
        初期化

        Args:
            config: ネットワーク設定
            action_dim: アクションの次元
        """
        super().__init__()

        # 共有CNNエンコーダ
        self.shared_encoder = self._build_cnn(config["vision_encoder"])

        # CNNの出力次元を計算
        vision_encoder_conv_out_channels = config["vision_encoder"]["conv_layers"][-1][
            "out_channels"
        ]
        vision_encoder_output_dim = vision_encoder_conv_out_channels * 4 * 4

        # 特徴融合層（左右のカメラの特徴を統合）
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                vision_encoder_output_dim * 2, vision_encoder_output_dim
            ),  # 2カメラ分
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # アクションヘッド（メインタスク）
        mlp_cfg = config["action_head"].copy()
        self.state_obs_dim = config["action_head"]["state_obs_dim"]
        if self.state_obs_dim is not None:
            mlp_cfg["input_dim"] = vision_encoder_output_dim + self.state_obs_dim
        else:
            mlp_cfg["input_dim"] = vision_encoder_output_dim
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

        # 姿勢ヘッド（補助タスク）
        pose_mlp_cfg = config["pose_head"].copy()
        pose_mlp_cfg["input_dim"] = vision_encoder_output_dim
        pose_mlp_cfg["output_dim"] = 7  # x, y, z, qw, qx, qy, qz
        self.pose_mlp = self._build_mlp(pose_mlp_cfg)

    @property
    def dtype(self):
        """パラメータのデータ型を取得"""
        return next(self.parameters()).dtype

    @staticmethod
    def _build_cnn(config: dict) -> nn.Sequential:
        """
        CNNエンコーダを構築

        畳み込み層 → バッチ正規化 → ReLU の繰り返し
        最後にAdaptive Average Poolingで固定サイズに

        Args:
            config: CNN設定

        Returns:
            Sequential: CNNモジュール
        """
        layers = []

        for conv_config in config["conv_layers"]:
            layers.extend(
                [
                    nn.Conv2d(
                        conv_config["in_channels"],
                        conv_config["out_channels"],
                        kernel_size=conv_config["kernel_size"],
                        stride=conv_config["stride"],
                        padding=conv_config["padding"],
                    ),
                    nn.BatchNorm2d(conv_config["out_channels"]),
                    nn.ReLU(),
                ]
            )

        # Adaptive Average Pooling（任意の入力サイズを固定サイズに）
        if config.get("pooling") == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp(config: dict) -> nn.Sequential:
        """
        MLPを構築

        Args:
            config: MLP設定

        Returns:
            Sequential: MLPモジュール
        """
        mlp_input_dim = config["input_dim"]
        layers = []

        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim

        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def get_features(self, rgb_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ステレオ画像から特徴を抽出

        Args:
            rgb_obs: ステレオRGB画像 [batch, 6, H, W]

        Returns:
            left_features: 左カメラの特徴
            right_features: 右カメラの特徴
        """
        # ステレオ画像を左右に分割
        left_rgb = rgb_obs[:, 0:3]  # 最初の3チャネル（左カメラ）
        right_rgb = rgb_obs[:, 3:6]  # 後の3チャネル（右カメラ）

        # 共有エンコーダで特徴抽出
        left_features = self.shared_encoder(left_rgb).flatten(start_dim=1)
        right_features = self.shared_encoder(right_rgb).flatten(start_dim=1)

        return left_features, right_features

    def forward(
        self, rgb_obs: torch.Tensor, state_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        フォワードパス

        Args:
            rgb_obs: ステレオRGB画像 [batch, 6, H, W]
            state_obs: 状態観測（エンドエフェクタ姿勢） [batch, 7]

        Returns:
            actions: 予測アクション [batch, 6]
        """
        # 左右カメラの特徴を抽出
        left_features, right_features = self.get_features(rgb_obs)

        # 特徴を結合
        combined_features = torch.cat([left_features, right_features], dim=-1)

        # 特徴融合
        fused_features = self.feature_fusion(combined_features)

        # 状態観測を追加（エンドエフェクタの姿勢）
        if state_obs is not None and self.state_obs_dim is not None:
            final_features = torch.cat([fused_features, state_obs], dim=-1)
        else:
            final_features = fused_features

        # アクション予測
        return self.mlp(final_features)

    def predict_pose(
        self, rgb_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        物体の姿勢を予測（補助タスク）

        Args:
            rgb_obs: ステレオRGB画像 [batch, 6, H, W]

        Returns:
            left_pose: 左カメラからの姿勢予測 [batch, 7]
            right_pose: 右カメラからの姿勢予測 [batch, 7]
        """
        left_features, right_features = self.get_features(rgb_obs)
        left_pose = self.pose_mlp(left_features)
        right_pose = self.pose_mlp(right_features)
        return left_pose, right_pose
