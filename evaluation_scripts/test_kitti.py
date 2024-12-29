# import sys
# sys.path.append('droid_slam')

# from tqdm import tqdm
# import numpy as np
# import torch
# import lietorch
# import cv2
# import os
# import glob 
# import time
# import argparse

# import torch.nn.functional as F
# from droid import Droid

# def show_image(image):
#     image = image.permute(1, 2, 0).cpu().numpy()
#     cv2.imshow('image', image / 255.0)
#     cv2.waitKey(1)

# def image_stream_kitti(datapath, image_size=[376, 1241]):
#     """ Image generator for KITTI dataset """

#     # Calibration parameters for KITTI
#     fx, fy, cx, cy = 718.856, 718.856, 607.1928, 185.2157

#     K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3, 3)
#     d_l = np.zeros(5)  # KITTI typically assumes zero distortion

#     # Read all png images in folder
#     images_list = sorted(glob.glob(os.path.join(datapath, 'image_2', '*.png')))

#     for t, imfile in enumerate(images_list):
#         image = cv2.imread(imfile)
#         ht0, wd0, _ = image.shape
#         image = cv2.undistort(image, K_l, d_l)
#         image = torch.from_numpy(image).permute(2, 0, 1)

#         intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda()
#         intrinsics[0] *= image.shape[2] / float(image_size[1])
#         intrinsics[1] *= image.shape[1] / float(image_size[0])
#         intrinsics[2] *= image.shape[2] / float(image_size[1])
#         intrinsics[3] *= image.shape[1] / float(image_size[0])
        

#         yield t, image[None], intrinsics
        
        
# def image_stream_kitti(datapath):
#     """ Image generator with pre-defined intrinsic parameters """

#     # Create the intrinsic matrix K
#     # K = np.eye(3)
#     # K[0, 0] = 718.856
#     # K[0, 2] = 718.856
#     # K[1, 1] = 607.1928
#     # K[1, 2] = 185.2157
#     fx = 718.856
#     fy = 718.856
#     cx = 607.1928
#     cy = 185.2157

#     # Read and sort image files from the directory
#     image_list = sorted(glob.glob(os.path.join(datapath, 'image_2', '*.png')))
    
#     target_height = 376  # Fixed target height
#     target_width = 1241  # Fixed target width

#     for t, imfile in enumerate(image_list):
#         image = cv2.imread(imfile)
        
#         # Distortion correction (assumes no distortion if calib vector not provided)
#         # If distortion is expected, modify or pass additional arguments to handle it

#         h0, w0, _ = image.shape
        
#         # # Scale to fit into 384x512 while maintaining aspect ratio
#         # h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
#         # w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

#         # # Resize and crop to multiples of 8
#         # image = cv2.resize(image, (w1, h1))
#         # image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        
#         # Resize the image to exactly match target size
#         image = cv2.resize(image, (target_width, target_height))

#         # Convert image to torch tensor and adjust dimensions
#         image = torch.as_tensor(image).permute(2, 0, 1)

#         # Scale intrinsics to match the resized image dimensions
#         intrinsics = torch.as_tensor([fx, fy, cx, cy])
#         # intrinsics[0::2] *= (w1 / w0)  # Adjust fx, cx for new width
#         # intrinsics[1::2] *= (h1 / h0)  # Adjust fy, cy for new height
#         intrinsics[0::2] *= (target_width / w0)  # Adjust fx, cx for new width
#         intrinsics[1::2] *= (target_height / h0)  # Adjust fy, cy for new height

#         yield t, image[None], intrinsics  # Return timestamp, image, and intrinsics

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--datapath")
#     parser.add_argument("--weights", default="droid.pth")
#     parser.add_argument("--buffer", type=int, default=512)
#     parser.add_argument("--image_size", default=[376, 1241])
#     parser.add_argument("--disable_vis", action="store_true")

#     parser.add_argument("--beta", type=float, default=0.6)
#     parser.add_argument("--filter_thresh", type=float, default=1.75)
#     parser.add_argument("--warmup", type=int, default=12)
#     parser.add_argument("--keyframe_thresh", type=float, default=2.25)
#     parser.add_argument("--frontend_thresh", type=float, default=12.0)
#     parser.add_argument("--frontend_window", type=int, default=25)
#     parser.add_argument("--frontend_radius", type=int, default=2)
#     parser.add_argument("--frontend_nms", type=int, default=1)

#     parser.add_argument("--backend_thresh", type=float, default=15.0)
#     parser.add_argument("--backend_radius", type=int, default=2)
#     parser.add_argument("--backend_nms", type=int, default=3)
    
#     parser.add_argument("--upsample", action="store_true")
#     parser.add_argument("--reconstruction_path", help="path to saved reconstruction") #再構築データの保存

#     args = parser.parse_args()

#     args.stereo = False
#     torch.multiprocessing.set_start_method('spawn')

#     print("Running evaluation on {}".format(args.datapath))
#     print(args)

#     droid = Droid(args)
#     time.sleep(5)

#     for (t, image, intrinsics) in tqdm(image_stream_kitti(args.datapath)):
#         if not args.disable_vis:
#             show_image(image)
#         droid.track(t, image, intrinsics=intrinsics)

#     # After tracking, get the estimated trajectory
#     traj_est = droid.terminate(image_stream_kitti(args.datapath))

#     ### Evaluate trajectory ###

#     print("#" * 20 + " Results...")

#     import evo
#     from evo.core.trajectory import PoseTrajectory3D
#     from evo.tools import file_interface
#     from evo.core import sync
#     import evo.main_ape as main_ape
#     from evo.core.metrics import PoseRelation

#     # Load KITTI ground truth trajectory
#     gt_file = os.path.join(args.datapath, '../../poses/07.txt')
#     traj_ref = file_interface.read_kitti_trajectory_file(gt_file)

#     # Convert estimated trajectory to KITTI format
#     traj_est = PoseTrajectory3D(
#         positions_xyz=traj_est[:, :3],
#         orientations_quat_wxyz=traj_est[:, 3:],
#     )

#     # Synchronize ground truth and estimated trajectories
#     traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

#     # Compute Absolute Pose Error (APE)
#     result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
#                           pose_relation=PoseRelation.translation_part, 
#                           align=True, correct_scale=True)

#     print(result)

import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F

#windowに画像表示
def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

#キャリブレーションなど
def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4: #画像の歪み補正
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics #画像と内部パラメータを生成

# 再構築データを保存することを定義
def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

# DROID-SLAMvideoオブジェクトからこれまでに処理されたタイムスタンプと画像を取得
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
# 再構築データを.npyファイルとして保存し、後で簡単にアクセスできるようにする
    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    
    
# 軌跡をKITTI形式で保存する
# def save_traj_kitti(traj_est, output_file):
#     """
#     Save trajectory in KITTI format.
#     """
    
#     from evo.tools.file_interface import write_kitti_poses_file
    
#     # positions_xyz と orientations_quat_wxyz を torch.Tensor に変換
#     positions_xyz = torch.tensor(traj_est.positions_xyz, dtype=torch.float32, device="cuda")
#     orientations_quat_wxyz = torch.tensor(traj_est.orientations_quat_wxyz, dtype=torch.float32, device="cuda")
    
#     # 軌跡を変換
#     poses = []
#     for i in range(len(traj_est.positions_xyz)):
#         # クォータニオンを取得
#         quat = orientations_quat_wxyz[i]
        
#         # SE3オブジェクトの作成（バッチ次元は直接変換不要）
#         se3 = lietorch.SE3(quat.unsqueeze(0))  # バッチ次元を追加
#         #R = lietorch.SE3(traj_est.orientations_quat_wxyz[i]).rotation().matrix().cpu().numpy()
#         R = se3.matrix().squeeze(0).cpu().numpy()[:3, :3]  # 回転行列を抽出
#         t = positions_xyz[i].cpu().numpy()
#         pose = np.eye(4)
#         pose[:3, :3] = R
#         pose[:3, 3] = t
#         poses.append(pose)
    
#     # KITTI形式で書き出し
#     write_kitti_poses_file(output_file, poses)
#     print(f"Trajectory saved to {output_file}")

# コマンドライン引数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame") #開始フレーム
    parser.add_argument("--stride", default=3, type=int, help="frame stride") #フレーム間隔
    
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction") #再構築データの保存
    args = parser.parse_args()

    args.stereo = False

    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths　再構築パスが指定されている場合にアップサンプリングを有効
    if args.reconstruction_path is not None:
        args.upsample = True

    #メイン処理
    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue
# 視覚化が有効になっている場合は、画像を表示
        if not args.disable_vis:
            show_image(image[0])
# droidが作成されていない場合,初期化を行う
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args) #droid.pyへ移動
# 現在の画像のトラッキングを行う        
        droid.track(t, image, intrinsics=intrinsics) #ここでエラー
# 再構築結果の保存
    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
# 軌道推定の完了
    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    print("Finish!!!!!!!!!!!!!!!!")
    
     ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    from evo.tools.file_interface import write_kitti_poses_file
    
    
    
    image_path = os.path.join(args.imagedir)
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]
    
    
    
    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))
    
    
    # 保存先ファイルパスを指定
    # output_file = "traj_est_kitti.txt"
    # write_kitti_poses_file(output_file, traj_est)
    #save_traj_kitti(traj_est, output_file)
    
    
    # if not isinstance(traj_est, PoseTrajectory3D):
    #     raise TypeError(f"Expected traj_est to be of type PoseTrajectory3D, but got {type(traj_est)}")
    
    #gt_file = os.path.join('datasets/kitti/dataset/poses/04.txt')
    # traj_ref = file_interface.read_kitti_poses_file(gt_file)
    traj_ref = file_interface.read_kitti_poses_file('datasets/kitti/dataset/poses/07.txt')
    
    # if not isinstance(traj_ref, PoseTrajectory3D):
    #     raise TypeError(f"Expected traj_ref to be of type PoseTrajectory3D, but got {type(traj_ref)}")

    
    #traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)


    print(result)
