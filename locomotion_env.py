"""
Locomotion Environment - Genesis チュートリアル 10

Go2 四足歩行ロボットの強化学習環境。
公式サンプル go2_env.py をベースに、詳細なコメントを追加。

Gym スタイルの環境インターフェース:
- reset(): 環境をリセットして初期観測を返す
- step(action): アクションを実行して (観測, 報酬, 終了, 情報) を返す

参照: Genesis/examples/locomotion/go2_env.py
"""

import math

import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    """指定範囲内のランダムテンソルを生成"""
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class Go2Env:
    """
    Go2 四足歩行ロボットの強化学習環境

    この環境は以下を提供:
    1. 並列シミュレーション（複数環境を同時実行）
    2. 観測空間（角速度、重力、関節位置など）
    3. 報酬関数（速度追従、姿勢維持など）
    4. 終了条件（転倒検出）
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        """
        環境の初期化

        Args:
            num_envs: 並列環境数
            env_cfg: 環境設定（関節角度、PD ゲインなど）
            obs_cfg: 観測設定（スケールなど）
            reward_cfg: 報酬設定（報酬関数の重み）
            command_cfg: コマンド設定（速度範囲）
            show_viewer: ビューア表示
        """
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # 実ロボットには1ステップの遅延がある
        self.simulate_action_latency = True
        # 制御周波数: 50Hz（実ロボットと同じ）
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # 設定を保存
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        ########## シーン作成 ##########

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,  # 安定性のためのサブステップ
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,  # 自己衝突なし
                tolerance=1e-5,
                # 歩行ポリシーでは衝突ペアは少ない
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            # 1つの環境のみレンダリング（パフォーマンス向上）
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        ########## 床を追加 ##########

        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        ########## ロボットを追加 ##########

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        ########## シーンビルド（並列環境） ##########

        self.scene.build(n_envs=num_envs)

        ########## モーターの DOF インデックスを取得 ##########

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        ########## PD コントローラ設定 ##########

        # kp: 比例ゲイン（位置誤差への反応）
        # kd: 微分ゲイン（速度誤差への反応）
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        ########## 重力方向ベクトル ##########

        # 初期化用（単一ベクトル）
        gravity_single = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        # 実行時用（num_envs 分に拡張）
        self.global_gravity = gravity_single.unsqueeze(0).expand(num_envs, -1).clone()

        ########## 初期状態 ##########

        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        # 初期化用は単一ベクトルで計算してから拡張
        init_proj_grav_single = transform_by_quat(gravity_single.unsqueeze(0), self.inv_base_init_quat.unsqueeze(0))
        self.init_projected_gravity = init_proj_grav_single.squeeze(0)

        ########## バッファ初期化 ##########

        # 状態バッファ
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)

        # 観測・報酬バッファ
        self.obs_buf = torch.empty((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # コマンドバッファ（目標速度）
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        ]

        # アクションバッファ
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)

        # 関節状態バッファ
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        # ベース状態バッファ
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)

        # デフォルト関節位置
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )

        # 追加情報
        self.extras = dict()
        self.extras["observations"] = dict()

        ########## 報酬関数の準備 ##########

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            # 報酬スケールを dt で調整
            self.reward_scales[name] *= self.dt
            # 報酬関数を登録
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            # エピソード報酬の合計
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def _resample_commands(self, envs_idx):
        """コマンド（目標速度）を再サンプリング"""
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        """
        環境を1ステップ進める

        Args:
            actions: ポリシーからのアクション [num_envs, num_actions]

        Returns:
            obs_buf: 観測 [num_envs, num_obs]
            rew_buf: 報酬 [num_envs]
            reset_buf: 終了フラグ [num_envs]
            extras: 追加情報
        """
        # アクションをクリップ
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # アクション遅延をシミュレート（実ロボットの特性）
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # 目標関節位置を計算
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # PD コントローラで関節位置を制御
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))

        # シミュレーション1ステップ
        self.scene.step()

        ########## 状態更新 ##########

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        ########## 報酬計算 ##########

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        ########## コマンド再サンプリング ##########

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        ########## 終了条件チェック ##########

        # エピソード長超過
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # ピッチ角度が大きすぎる（転倒）
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        # ロール角度が大きすぎる（転倒）
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # タイムアウト情報
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        ########## 必要なら環境リセット ##########

        self._reset_idx(self.reset_buf)

        ########## 観測更新 ##########

        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """現在の観測を取得"""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """特権観測（訓練時のみ使用）を取得"""
        return None

    def _reset_idx(self, envs_idx=None):
        """指定した環境をリセット"""
        # 状態をリセット
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # バッファをリセット
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_pos.copy_(self.init_base_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # エピソード報酬を記録
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(envs_idx, 0.0)

        # コマンド再サンプリング
        self._resample_commands(envs_idx)

    def _update_observation(self):
        """観測を更新"""
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3: 角速度
                self.projected_gravity,  # 3: 投影重力
                self.commands * self.commands_scale,  # 3: コマンド
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12: 関節位置
                self.dof_vel * self.obs_scales["dof_vel"],  # 12: 関節速度
                self.actions,  # 12: 前回のアクション
            ),
            dim=-1,
        )

    def reset(self):
        """全環境をリセット"""
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    ########## 報酬関数 ##########

    def _reward_tracking_lin_vel(self):
        """線形速度追従報酬（xy軸）"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """角速度追従報酬（yaw）"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        """z軸速度ペナルティ（上下動を抑制）"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        """アクション変化ペナルティ（滑らかな動きを促進）"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """デフォルト姿勢維持報酬"""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """ベース高さ維持報酬"""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
