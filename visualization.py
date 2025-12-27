"""
Visualization & Rendering - Genesis チュートリアル 02

このスクリプトでは Genesis の可視化機能を学びます：
1. Viewer（インタラクティブビューア）の設定
2. Camera（カメラ）の追加と画像レンダリング
3. ビデオ録画機能
"""

import numpy as np

import genesis as gs

########## 初期化 ##########

gs.init(backend=gs.cpu)

########## シーン作成（Viewer と可視化オプション設定） ##########

scene = gs.Scene(
    # Viewer を表示
    show_viewer=True,

    # Viewer の設定
    # res: 解像度 (幅, 高さ)
    # camera_pos: カメラ位置 (x, y, z)
    # camera_lookat: カメラが見る点 (x, y, z)
    # camera_fov: 視野角（Field of View）度数
    # max_FPS: 最大フレームレート
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),

    # 可視化オプション（Viewer と Camera 共通）
    # show_world_frame: 原点に座標軸を表示
    # world_frame_size: 座標軸のサイズ（メートル）
    # show_link_frame: 各リンクの座標軸を表示
    # show_cameras: カメラの位置・視錐台を表示
    # plane_reflection: 床の反射を有効化
    # ambient_light: 環境光の色 (R, G, B) 0.0〜1.0
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),

    # レンダラー（Rasterizer = 高速な標準描画）
    renderer=gs.renderers.Rasterizer(),
)

########## Entity 追加 ##########

# 床（平面）
plane = scene.add_entity(gs.morphs.Plane())

# Franka ロボットアーム
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########## カメラ追加 ##########

# シーンにカメラを追加
# カメラは Viewer とは独立して動作し、画像をレンダリングできる
# res: 画像解像度
# pos: カメラ位置
# lookat: カメラが見る点
# fov: 視野角
# GUI: True にすると OpenCV ウィンドウで画像を表示
cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False,  # OpenCV ウィンドウは使わない
)

########## シーンビルド ##########

scene.build()

########## カメラで画像をレンダリング ##########

# render() で各種画像を取得できる
# rgb: RGB カラー画像
# depth: 深度画像（カメラからの距離）
# segmentation: セグメンテーション画像（物体ごとに色分け）
# normal: 法線マップ（表面の向きを色で表現）
print("画像をレンダリング中...")
rgb, depth, segmentation, normal = cam.render(
    depth=True,
    segmentation=True,
    normal=True
)

print(f"RGB 画像サイズ: {rgb.shape}")
print(f"Depth 画像サイズ: {depth.shape}")
print(f"Segmentation 画像サイズ: {segmentation.shape}")
print(f"Normal 画像サイズ: {normal.shape}")

########## ビデオ録画 ##########

print("\nビデオ録画を開始...")

# 録画開始（以降の render() で取得した RGB 画像が記録される）
cam.start_recording()

# 120 ステップのシミュレーションを実行しながらカメラを動かす
for i in range(120):
    scene.step()

    # カメラを円軌道で移動させる
    # np.sin/cos で x, y 座標を計算（半径 3.0m の円）
    cam.set_pose(
        pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat=(0, 0, 0.5),
    )

    # 画像をレンダリング（録画中は自動的に記録される）
    cam.render()

# 録画停止してビデオを保存
cam.stop_recording(save_to_filename='video.mp4', fps=60)

print("ビデオを 'video.mp4' に保存しました")
print("\n完了！")
