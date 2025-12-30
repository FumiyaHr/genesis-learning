"""
Grasp Environment - Genesis チュートリアル 12

Franka Panda ロボットアームによる把持タスク環境。

## 概要
- 7DoF ロボットアーム + 2DoF グリッパー で物体に近づくタスク
- エンドエフェクタ制御（手先の位置・姿勢を指定）
- キーポイント報酬による姿勢アライメント

## 観測空間（14次元）
- 指先 - 物体の相対位置（3次元）
- 指先の姿勢（四元数、4次元）
- 物体の位置（3次元）
- 物体の姿勢（四元数、4次元）

## アクション空間（6次元）
- 位置の変化量（dx, dy, dz）
- 姿勢の変化量（roll, pitch, yaw）

参照: Genesis/examples/manipulation/grasp_env.py
"""

import math
from typing import Literal

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_quat,
)


########## GraspEnv: 把持タスク環境 ##########


class GraspEnv:
    """
    把持（Grasping）タスク環境

    ロボットアームが箱に近づき、把持可能な姿勢をとることを学習する。
    """

    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        """
        環境の初期化

        Args:
            env_cfg: 環境設定（環境数、観測次元、アクション次元など）
            reward_cfg: 報酬設定（報酬関数のスケール）
            robot_cfg: ロボット設定（リンク名、デフォルト関節角度など）
            show_viewer: ビューワを表示するか
        """
        # === 基本パラメータ ===
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]  # 観測次元: 14
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # アクション次元: 6
        self.device = gs.device

        # === 画像設定（ステレオカメラ用） ===
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)

        # === 時間設定 ===
        self.ctrl_dt = env_cfg["ctrl_dt"]  # 制御周期（秒）
        self.max_episode_length = math.ceil(
            env_cfg["episode_length_s"] / self.ctrl_dt
        )

        # === 設定の保存 ===
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg

        # アクションスケール: 生のアクション（-1〜1）を実際の変化量に変換
        self.action_scales = torch.tensor(
            env_cfg["action_scales"], device=self.device
        )

        # === シーン構築 ===
        self._build_scene(show_viewer)

        # === 報酬関数の準備 ===
        self._setup_rewards()

        # === キーポイントオフセットの初期化 ===
        # キーポイント報酬で使用する7点の相対位置
        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs,
            device=self.device,
            unit_length=0.5,
        )

        # === バッファ初期化とリセット ===
        self._init_buffers()
        self.reset()

    def _build_scene(self, show_viewer: bool) -> None:
        """
        シーンの構築

        シーンに以下を追加:
        1. 地面
        2. Franka Panda ロボットアーム
        3. 把持対象の箱
        4. ステレオカメラ（左右2台）
        """
        # シーン作成
        # Note: macOS では substeps を増やして安定化
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt,
                substeps=4,  # 2 → 4 に増やして安定化
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs))),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            # BatchRenderer は Linux のみサポート
            # macOS では通常のレンダラーを使用
            # renderer=gs.options.renderers.BatchRenderer(
            #     use_rasterizer=self.env_cfg["use_rasterizer"],
            # ),
            show_viewer=show_viewer,
        )

        # 地面の追加
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # ロボットの追加
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=self.env_cfg["robot_cfg"],
            device=gs.device,
        )

        # 把持対象（赤い箱）の追加
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=self.env_cfg["box_size"],  # 箱のサイズ [x, y, z]
                fixed=self.env_cfg["box_fixed"],  # 固定するか
                collision=self.env_cfg["box_collision"],  # 衝突判定
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),  # 赤色
                ),
            ),
        )

        # 可視化用カメラ（オプション）
        if self.env_cfg.get("visualize_camera", False):
            self.vis_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=True,
                debug=True,
            )

        # ステレオカメラ（左右2台）
        # Stage 2 の模倣学習で視覚入力として使用
        self.left_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(1.25, 0.3, 0.3),  # 左側から見る
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=False,
        )
        self.right_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(1.25, -0.3, 0.3),  # 右側から見る
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=False,
        )

        # シーンビルド
        self.scene.build(n_envs=self.num_envs)

        # PD制御ゲインの設定（ビルド後に呼び出す必要がある）
        self.robot.set_pd_gains()

    def _setup_rewards(self) -> None:
        """報酬関数のセットアップ"""
        self.reward_functions = {}
        self.episode_sums = {}

        for name in self.reward_scales.keys():
            # 報酬スケールに制御周期を掛ける（時間正規化）
            self.reward_scales[name] *= self.ctrl_dt
            # 報酬関数を登録（例: "keypoints" → self._reward_keypoints）
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            # エピソード累積報酬の初期化
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

    def _init_buffers(self) -> None:
        """内部バッファの初期化"""
        # エピソード長カウンタ
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        # リセットフラグ
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=gs.device
        )
        # 目標姿勢（物体の位置・姿勢）
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        # 追加情報
        self.extras = {"observations": {}}

    ########## リセット ##########

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """
        指定した環境をリセット

        Args:
            envs_idx: リセットする環境のインデックス
        """
        if len(envs_idx) == 0:
            return

        self.episode_length_buf[envs_idx] = 0

        # ロボットをホームポジションにリセット
        self.robot.reset(envs_idx)

        # 物体をランダムな位置・姿勢にリセット
        num_reset = len(envs_idx)

        # ランダムな位置（ロボットの前方）
        random_x = torch.rand(num_reset, device=self.device) * 0.4 + 0.2  # 0.2〜0.6
        random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.5  # -0.25〜0.25
        random_z = torch.ones(num_reset, device=self.device) * 0.025  # 地面すれすれ
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        # ランダムな姿勢（ヨー回転のみ）
        # 基本姿勢: 下向き（グリッパーが上から掴む姿勢）
        q_downward = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(num_reset, 1)

        # ランダムなヨー回転（±45度）
        random_yaw = (
            torch.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        ) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self.device),
                torch.zeros(num_reset, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        # 目標姿勢の保存と物体の配置
        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        # エピソード統計の記録
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        """全環境をリセット"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        obs, self.extras = self.get_observations()
        return obs, self.extras

    ########## ステップ ##########

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        1ステップ実行

        Args:
            actions: アクション [num_envs, 6]
                     (dx, dy, dz, droll, dpitch, dyaw)

        Returns:
            obs: 観測
            reward: 報酬
            reset_buf: リセットフラグ
            extras: 追加情報
        """
        # エピソード長をインクリメント
        self.episode_length_buf += 1

        # アクションをスケーリング
        actions = self.rescale_action(actions)

        # ロボットにアクションを適用
        # open_gripper=True: グリッパーは開いたまま（位置決めフェーズ）
        self.robot.apply_action(actions, open_gripper=True)

        # シミュレーション実行
        self.scene.step()

        # エピソード終了判定
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # 報酬計算
        reward = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # 観測取得
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        """アクションをスケーリング（-1〜1 → 実際の変化量）"""
        return action * self.action_scales

    def is_episode_complete(self) -> torch.Tensor:
        """エピソード終了判定"""
        # 時間切れ判定
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = time_out_buf

        # タイムアウト情報の記録（価値関数のブートストラップ用）
        time_out_idx = time_out_buf.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        return self.reset_buf.nonzero(as_tuple=True)[0]

    ########## 観測 ##########

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """
        観測の取得

        観測（14次元）:
        - 指先と物体の相対位置（3次元）
        - 指先の姿勢（四元数、4次元）
        - 物体の位置（3次元）
        - 物体の姿勢（四元数、4次元）

        Note:
            物体の位置・姿勢は「特権状態」と呼ばれる。
            シミュレーションでは直接取得できるが、
            実世界ではカメラなどのセンサーから推定する必要がある。
        """
        # 指先（グリッパー中心）の位置と姿勢
        finger_pos = self.robot.center_finger_pose[:, :3]
        finger_quat = self.robot.center_finger_pose[:, 3:7]

        # 物体の位置と姿勢
        obj_pos = self.object.get_pos()
        obj_quat = self.object.get_quat()

        # 観測の構築
        obs_components = [
            finger_pos - obj_pos,  # 相対位置（指先から見た物体の方向）
            finger_quat,  # 指先の姿勢
            obj_pos,  # 物体の位置（特権状態）
            obj_quat,  # 物体の姿勢（特権状態）
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)

        # クリティック用の観測も保存
        self.extras["observations"]["critic"] = obs_tensor

        return obs_tensor, self.extras

    def get_privileged_observations(self) -> None:
        """特権観測の取得（使用しない）"""
        return None

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        """
        ステレオカメラからRGB画像を取得

        Stage 2 の模倣学習で使用する視覚入力。

        Args:
            normalize: 0〜255 を 0〜1 に正規化するか

        Returns:
            stereo_rgb: [num_envs, 6, H, W]
                        左カメラ3ch + 右カメラ3ch

        Note:
            macOS では BatchRenderer が使えないため、カメラの出力は (H, W, C) 形式。
            これを (num_envs, C, H, W) 形式に変換して返す。
        """
        # 左右カメラからレンダリング
        rgb_left, _, _, _ = self.left_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_right, _, _, _ = self.right_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )

        # numpy array の場合は torch tensor に変換
        if isinstance(rgb_left, np.ndarray):
            rgb_left = torch.from_numpy(rgb_left.copy()).to(self.device)
        if isinstance(rgb_right, np.ndarray):
            rgb_right = torch.from_numpy(rgb_right.copy()).to(self.device)

        # macOS: (H, W, C) → (1, C, H, W) → (num_envs, C, H, W)
        # Linux (BatchRenderer): (B, H, W, C) → (B, C, H, W)
        if rgb_left.dim() == 3:
            # macOS: バッチ次元がない場合
            rgb_left = rgb_left.permute(2, 0, 1)[:3].float()  # (C, H, W)
            rgb_right = rgb_right.permute(2, 0, 1)[:3].float()
            # 全環境で同じ画像を複製（macOS の制限）
            # .contiguous() は expand 後のテンソルをメモリ上で連続させる
            # （conv2d などの操作に必要）
            rgb_left = rgb_left.unsqueeze(0).expand(self.num_envs, -1, -1, -1).contiguous()
            rgb_right = rgb_right.unsqueeze(0).expand(self.num_envs, -1, -1, -1).contiguous()
        else:
            # Linux: バッチ次元がある場合
            rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3].float()
            rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3].float()

        # 正規化
        if normalize:
            rgb_left = torch.clamp(rgb_left, min=0.0, max=255.0) / 255.0
            rgb_right = torch.clamp(rgb_right, min=0.0, max=255.0) / 255.0

        # 左右を結合: [B, 6, H, W]
        stereo_rgb = torch.cat([rgb_left, rgb_right], dim=1)
        return stereo_rgb

    ########## 報酬関数 ##########

    def _reward_keypoints(self) -> torch.Tensor:
        """
        キーポイント報酬

        キーポイント報酬とは:
            単純な距離報酬では「位置は近いが姿勢がずれている」状態でも
            高い報酬が得られてしまう。

            キーポイント報酬は、グリッパーと物体それぞれに7つの参照点を設定し、
            対応する点同士の距離で評価する。これにより位置と姿勢の両方を考慮できる。

        キーポイントの配置:
            原点 + 6軸方向（±x, ±y, ±z）の7点

        報酬計算:
            reward = exp(-distance_sum)
            距離が0に近いほど報酬は1に近づく
        """
        keypoints_offset = self.keypoints_offset

        # 指先のオフセット（フレームの原点から実際の指先へ）
        finger_tip_z_offset = torch.tensor(
            [0.0, 0.0, -0.06],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)

        # 指先のキーポイント位置（ワールド座標）
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3] + finger_tip_z_offset,
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )

        # 物体のキーポイント位置（ワールド座標）
        object_pos_keypoints = self._to_world_frame(
            self.object.get_pos(),
            self.object.get_quat(),
            keypoints_offset,
        )

        # キーポイント間の距離の合計
        dist = torch.norm(
            finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1
        ).sum(-1)

        # 指数報酬: 距離が小さいほど報酬が大きい
        return torch.exp(-dist)

    def _to_world_frame(
        self,
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, 7, 3]
    ) -> torch.Tensor:
        """ローカル座標のキーポイントをワールド座標に変換"""
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(
                keypoints_offset[:, k], quaternion
            )
        return world

    @staticmethod
    def get_keypoint_offsets(
        batch_size: int, device: str, unit_length: float = 0.5
    ) -> torch.Tensor:
        """
        キーポイントのオフセットを取得

        7つのキーポイント:
        - 原点
        - x軸方向（±）
        - y軸方向（±）
        - z軸方向（±）
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # 原点
                    [-1.0, 0, 0],  # x-
                    [1.0, 0, 0],  # x+
                    [0, -1.0, 0],  # y-
                    [0, 1.0, 0],  # y+
                    [0, 0, -1.0],  # z-
                    [0, 0, 1.0],  # z+
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    ########## デモ ##########

    def grasp_and_lift_demo(self) -> None:
        """
        把持と持ち上げのデモ

        訓練したポリシーで物体に近づいた後、
        このメソッドで実際に掴んで持ち上げることができる。
        """
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()

        # 持ち上げ姿勢
        lift_height = 0.3
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height

        # 最終姿勢（テーブルの上）
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4

        # ホームポジション
        reset_pose = torch.tensor(
            [0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        for i in range(total_steps):
            if i < total_steps / 4:  # 把持
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:  # 持ち上げ
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:  # 移動
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:  # リセット
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()


########## Manipulator: ロボットアームクラス ##########


class Manipulator:
    """
    Franka Panda ロボットアームのラッパークラス

    Franka Panda:
        - 7自由度（DoF）のロボットアーム
        - 2自由度のパラレルジョーグリッパー
        - 産業用ロボットとして広く使用されている
    """

    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: dict,
        device: str = "cpu",
    ):
        """
        初期化

        Args:
            num_envs: 環境数
            scene: Genesis シーン
            args: ロボット設定
            device: デバイス
        """
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # ロボットをシーンに追加
        # MJCF形式（MuJoCoのモデル形式）を使用
        self._robot_entity = scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # グリッパーの開閉位置
        self._gripper_open_dof = 0.04  # 開いた状態
        self._gripper_close_dof = 0.00  # 閉じた状態

        # 逆運動学（IK）の手法
        self._ik_method: Literal["gs_ik", "dls_ik"] = args["ik_method"]

        # 内部バッファの初期化
        self._init()

    def _init(self) -> None:
        """内部バッファの初期化"""
        # アーム関節数: 全関節 - グリッパー2関節 = 7
        self._arm_dof_dim = self._robot_entity.n_dofs - 2
        self._gripper_dim = 2

        # 関節インデックス
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]

        # リンクの取得
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(
            self._args["gripper_link_names"][0]
        )
        self._right_finger_link = self._robot_entity.get_link(
            self._args["gripper_link_names"][1]
        )

        # デフォルト関節角度
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]

    def set_pd_gains(self) -> None:
        """
        PD制御ゲインの設定

        PD制御とは:
            位置制御の一種。目標位置への追従を制御する。
            - Kp（比例ゲイン）: 目標との差に比例した力
            - Kv（微分ゲイン）: 速度に比例した減衰力

        Note:
            ロボットごとに適切なゲインは異なる。
            高品質なURDF/MJCFファイルにはゲイン情報が含まれていることもある。
        """
        # 各関節のPゲイン（位置ゲイン）
        self._robot_entity.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        # 各関節のDゲイン（速度ゲイン）
        self._robot_entity.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        # 各関節のトルク範囲
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def reset(self, envs_idx: torch.IntTensor) -> None:
        """指定した環境のロボットをリセット"""
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None) -> None:
        """ホームポジションにリセット"""
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)

        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)

        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """
        アクションを適用

        Args:
            action: 6次元アクション (dx, dy, dz, droll, dpitch, dyaw)
            open_gripper: グリッパーを開くか
        """
        # 現在の関節角度を取得
        q_pos = self._robot_entity.get_qpos()

        # 逆運動学で目標関節角度を計算
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid IK method: {self._ik_method}")

        # グリッパーの制御
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof

        # 位置制御コマンドを送信
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis 逆運動学

        Genesis 組み込みのIKソルバーを使用。
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # 目標位置 = 現在位置 + 変化量
        target_position = delta_position + self._ee_link.get_pos()

        # 目標姿勢 = 現在姿勢 × 変化量
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(
            quat_rel, self._ee_link.get_quat()
        )

        # IKで関節角度を計算
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped Least Squares (DLS) 逆運動学

        DLS法とは:
            通常の最小二乗法IKは、特異点（関節が伸びきった状態など）で
            関節速度が発散してしまう問題がある。

            DLS法はこれを防ぐため、ダンピング項（λ）を追加する。
            λが大きいほど安定するが、精度は下がる。

        数式:
            Δq = J^T (J J^T + λ²I)^(-1) Δx

            J: ヤコビアン（関節速度と手先速度の関係）
            Δx: 手先の移動量
            Δq: 関節の移動量
            λ: ダンピング係数
        """
        delta_pose = action[:, :6]
        lambda_val = 0.01  # ダンピング係数

        # ヤコビアンの取得
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)

        # ダンピング行列
        lambda_matrix = (lambda_val**2) * torch.eye(
            n=jacobian.shape[1], device=self._device
        )

        # DLS逆運動学
        delta_joint_pos = (
            jacobian_T
            @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)

        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True) -> None:
        """
        目標姿勢に移動

        Args:
            goal_pose: 目標姿勢 [x, y, z, qw, qx, qy, qz]
            open_gripper: グリッパーを開くか
        """
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )

        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    ########## プロパティ ##########

    @property
    def base_pos(self) -> torch.Tensor:
        """ロボットベースの位置"""
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        エンドエフェクタの姿勢

        Returns:
            [x, y, z, qw, qx, qy, qz]
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        """左指の姿勢"""
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        """右指の姿勢"""
        pos, quat = self._right_finger_link.get_pos(), self._right_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        グリッパー中心の姿勢

        左右の指の中間点を返す。
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)
