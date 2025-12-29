"""
Hover Environment - Genesis チュートリアル 11

ドローンのホバリング制御環境。
PPO で訓練して、ランダムなターゲット位置に到達してホバリングを維持する。

参照: Genesis/examples/drone/hover_env.py
"""

import copy
import math

import torch

import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)


########## ユーティリティ関数 ##########


def gs_rand_float(lower, upper, shape, device):
    """指定範囲のランダムな浮動小数点数を生成"""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


########## HoverEnv クラス ##########


class HoverEnv:
    """
    ドローンホバリング環境

    Gym スタイルのインターフェース:
    - reset(): 環境をリセット
    - step(action): 1ステップ実行

    locomotion との違い:
    - 制御対象: 4つのプロペラ（RPM制御）
    - 観測: 17次元（相対位置、姿勢、速度など）
    - 行動: 4次元（各プロペラのRPM調整）
    """

    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, camera_cfg=None
    ):
        """
        環境を初期化

        Args:
            num_envs: 並列環境数
            env_cfg: 環境設定
            obs_cfg: 観測設定
            reward_cfg: 報酬設定
            command_cfg: コマンド（ターゲット）設定
            show_viewer: ビューア表示
            camera_cfg: 録画用カメラ設定（オプション）
        """
        ########## 基本設定 ##########

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # 制御周波数: 100Hz（locomotion は 50Hz）
        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # 設定を保存
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        ########## シーン作成 ##########

        # レンダリングする環境数（パフォーマンスのため制限）
        rendered_env_num = min(1, self.num_envs)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        ########## エンティティ追加 ##########

        # 地面
        self.scene.add_entity(gs.morphs.Plane())

        # ターゲット（可視化用の球体）
        if env_cfg.get("visualize_target", True):
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),  # 赤い球
                    ),
                ),
            )
        else:
            self.target = None

        # ドローン（Crazyflie 2.X）
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # ホバリングRPM（この値でドローンが静止する）
        self.hover_rpm = 14468.429183500699

        ########## 録画用カメラ（オプション） ##########

        self.camera = None
        if camera_cfg is not None:
            self.camera = self.scene.add_camera(
                res=camera_cfg.get("res", (1280, 720)),
                pos=camera_cfg.get("pos", (3.0, 0.0, 2.5)),
                lookat=camera_cfg.get("lookat", (0.0, 0.0, 1.0)),
                fov=camera_cfg.get("fov", 40),
                GUI=False,
            )

        ########## シーンビルド ##########

        self.scene.build(n_envs=num_envs)

        ########## 報酬関数の準備 ##########

        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            # dt を掛けて時間正規化
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        ########## バッファ初期化 ##########

        # 観測、報酬、リセットバッファ
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )

        # コマンド（ターゲット位置）
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float
        )

        # アクション
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        # ドローン状態
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # 相対位置（ターゲットとの距離）
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        # オイラー角（終了判定用）
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        # クラッシュ条件
        self.crash_condition = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )

        # ログ用
        self.extras = {}
        self.extras["observations"] = {}

    ########## コマンド生成 ##########

    def _resample_commands(self, envs_idx):
        """
        新しいターゲット位置をサンプリング

        Args:
            envs_idx: リサンプルする環境のインデックス
        """
        if len(envs_idx) == 0:
            return

        # ランダムなターゲット位置を生成
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device
        )

    def _at_target(self):
        """
        ターゲットに到達した環境のインデックスを返す
        """
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    ########## ステップ実行 ##########

    def step(self, actions):
        """
        1ステップ実行

        Args:
            actions: ポリシーからのアクション [num_envs, 4]
                     各値は [-1, 1] の範囲で、ホバリングRPMからの調整量

        Returns:
            obs: 観測 [num_envs, 17]
            rew: 報酬 [num_envs]
            reset: リセットフラグ [num_envs]
            extras: 追加情報
        """
        ########## アクション処理 ##########

        # アクションをクリップ
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )

        # RPMに変換して設定
        # action=0 → ホバリング状態
        # action>0 → 上昇、action<0 → 下降
        rpm = (1 + self.actions * 0.8) * self.hover_rpm
        self.drone.set_propellels_rpm(rpm)

        # ターゲット位置を更新（可視化用）
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)

        # シミュレーション実行
        self.scene.step()

        ########## 状態更新 ##########

        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()

        # 相対位置（ターゲットとの距離）
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

        # 姿勢（四元数）
        self.base_quat[:] = self.drone.get_quat()

        # オイラー角（度）に変換
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )

        # ボディ座標系での速度
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        ########## ターゲット到達判定 ##########

        # ターゲットに到達したら新しいターゲットをサンプリング
        envs_at_target = self._at_target()
        self._resample_commands(envs_at_target)

        ########## 終了判定 ##########

        # クラッシュ条件
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )

        # リセット条件: エピソード長超過 または クラッシュ
        self.reset_buf = (
            (self.episode_length_buf > self.max_episode_length) | self.crash_condition
        )

        # タイムアウト情報
        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        ########## 環境リセット ##########

        self._reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        ########## 報酬計算 ##########

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        ########## 観測計算 ##########

        self.obs_buf = torch.cat(
            [
                # 相対位置（クリップして正規化）
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                # 姿勢（四元数）
                self.base_quat,
                # 線速度（クリップして正規化）
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                # 角速度（クリップして正規化）
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                # 前回のアクション
                self.last_actions,
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    ########## リセット ##########

    def _reset_idx(self, envs_idx):
        """指定した環境をリセット"""
        if len(envs_idx) == 0:
            return

        # ドローン位置・姿勢をリセット
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        self.drone.set_pos(
            self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.drone.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )

        # 速度をリセット
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # バッファをリセット
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # エピソード報酬をログに記録
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # 新しいターゲットをサンプリング
        self._resample_commands(envs_idx)

        # 相対位置を更新
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

    def reset(self):
        """全環境をリセット"""
        self.reset_buf[:] = True
        self._reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def get_observations(self):
        """観測を取得"""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """特権観測（なし）"""
        return None

    ########## 報酬関数 ##########

    def _reward_target(self):
        """
        ターゲット報酬

        前ステップからの距離変化を報酬にする。
        ターゲットに近づくと正、離れると負。
        """
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(
            torch.square(self.rel_pos), dim=1
        )
        return target_rew

    def _reward_smooth(self):
        """
        スムーズ報酬

        急激なアクション変化をペナルティ。
        """
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        """
        ヨー角報酬

        ヨー角（Z軸回転）が0に近いほど高い報酬。
        """
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        """
        角速度ペナルティ

        角速度が大きいほどペナルティ。
        """
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        """
        クラッシュペナルティ

        墜落条件に該当したら大きなペナルティ。
        """
        crash_rew = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        crash_rew[self.crash_condition] = 1
        return crash_rew
