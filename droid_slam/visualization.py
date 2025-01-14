import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d
import os
import re
#python高速化のため
from numba import jit
#要素のカウント
from collections import Counter

import time

from lietorch import SE3
import geom.projective_ops as pops

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    ##追記##
    droid_visualization.masks = {}
    droid_visualization.next_colors = {}
    droid_visualization.count = 0
    ##########
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.005

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True
            
            
    def display_frame_at_t(video, t):
    # """
    # 指定した時刻 t のフレームを画像として表示する
    # """
        # tのフレーム画像を取得
        frame = video.images[t].cpu().numpy()  # Tensor を NumPy 配列に変換
        frame = frame.transpose(1, 2, 0)       # (C, H, W) → (H, W, C)

        # 表示
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV 用に変換
        cv2.imshow(f"Frame at t={t}", frame_bgr)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        return
    
    @jit(nopython=True)
    def calc_distance(point, pre_point):
        distances = np.empty(len(pre_point))
        # for i in range(len(pre_point)):
            # diff = point - pre_point[i]
            # distance = np.sqrt(np.dot(diff.T, diff))
            # distances[i] = distance
        distances = [np.sqrt(np.dot(point - pre_point[i].T, point - pre_point[i])) for i in range(len(pre_point))]
        return distances

              
    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        
        intrinsics = cam.intrinsic
        #print(f"Intrinsic width: {intrinsics.width}, height: {intrinsics.height}")
        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value
                dirty_index, = torch.where(video.dirty.clone()) # video.dirty が True のフレームを dirty_index として取得
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False
            
            # 指定されたフレーム（dirty_index）の姿勢と深度を取得
            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()
            
            #入力画像で色を塗りたいときは以下を使う
            #images = torch.index_select(video.images, 0, dirty_index)
            

            ##########segmentationの画像を取得し、三次元点群に色を付ける処理##########
            
            # 入力画像群をソート h,w = 328.0, 584.0
            image_files = sorted(os.listdir('mask2former_cityscapes_full/'))  
            
            #print(f"video.images.shape: {video.images.shape}")
            # # for i, idx in enumerate(dirty_index):
            # # # ここで現在のフレームに対応する画像を取得する
            # #     #print(f"i: {i}")
            # #     tstamp_index = int(video.tstamp[idx])  # 必要に応じて整数にキャスト
            # #     current_image = torch.from_numpy(cv2.imread(f'mask2former_cityscapes_full/{image_files[tstamp_index]}')).permute(2, 0, 1).to(device="cuda", dtype=torch.uint8)
            # #     image_tensor = current_image.unsqueeze(0)  # バッチ次元を追加 (1, 3, H, W)
    
            # #     # i 番目のフレームに current_image を適用
            # #     images[i] = image_tensor
            # #     print(f"image_files[video.tstamp[i]]: {image_files[tstamp_index]}")
            # #     #print(f"tstamp_index: {tstamp_index}")
            
            
            
            
            semantic_images = torch.zeros((1024,3,328,584), dtype=torch.uint8)  # 4次元テンソル
            semantic_images = semantic_images.to(device="cuda")
            #print(f"dirty_index: {dirty_index}")
            for idx in dirty_index:
            # ここで現在のフレームに対応する画像を取得する
                tstamp_index = int(video.tstamp[idx])  # 必要に応じて整数にキャスト
                semantic_images[idx] = torch.from_numpy(cv2.imread(f'mask2former_cityscapes_full/{image_files[tstamp_index]}')).permute(2, 0, 1)
                
                # # 画像を表示
                # cv2.imshow(f"Image {idx}", cv2.imread(f'mask2former_cityscapes_full/{image_files[tstamp_index]}'))
                # cv2.waitKey(10000)  # キー入力待ち（ウィンドウを閉じるため）
                # display_frame_at_t(video, idx)
             
            images = torch.index_select(semantic_images, 0, dirty_index)
            
            ######################################################################
            
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            
            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))
            
            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                # フレーム ix に対応する古いカメラオブジェクトを削除
                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                #フレーム ix に対応する古い点群データを削除
                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ## add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                #vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                #print(f"images[i].shape: {images[i].shape}")    
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                # clr[:] = 1.0
                #print(f"pts: {pts}")
                
                
                
                #########################################
                ## 一個後のフレームの点群と比較 ###
                # if i == 0:
                #     next_mask = masks[i+1].reshape(-1)
                #     next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                #     next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    

                #     # 現在のフレームの点ごとにチェック
                #     for j, point in enumerate(pts):
                #         x, y, z = point
                #         next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                #         next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)

                #         # 条件に一致する点があるか確認
                #         next_in_range = next_x_range & next_y_range
                #         if np.any(next_in_range):   # 範囲内の点が存在
                #             next_close_pts = next_pts[next_in_range]
                #             next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得

                #             # # 三次元距離を計算
                #             next_distances = calc_distance(point,next_close_pts)
                #             close_clr_list = []
                #             close_clr_list = [next_close_clr[k] for k, dist in enumerate(next_distances) if dist < 0.02]
                #             # 最も頻出する色を決定
                #             if len(close_clr_list) > 0:
                #                 # most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                #                 close_clr_list.append(clr[j])
                #                 most_common_color = Counter(map(tuple, close_clr_list)).most_common(1)[0][0]
                #                 clr[j] = most_common_color 
                
                if i ==0 and len(dirty_index) > 10:
                    next_mask = masks[i+1].reshape(-1)
                    next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next1_mask = masks[i+2].reshape(-1)
                    next1_pts = points[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next1_clr = images[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next2_mask = masks[i+3].reshape(-1)
                    next2_pts = points[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                    next2_clr = images[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                   
                    # 現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                        next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)
                        next1_x_range = (next1_pts[:, 0] >= x - 0.1) & (next1_pts[:, 0] <= x + 0.1)
                        next1_y_range = (next1_pts[:, 1] >= y - 0.1) & (next1_pts[:, 1] <= y + 0.1)
                        next2_x_range = (next2_pts[:, 0] >= x - 0.1) & (next2_pts[:, 0] <= x + 0.1)
                        next2_y_range = (next2_pts[:, 1] >= y - 0.1) & (next2_pts[:, 1] <= y + 0.1)

                        # 条件に一致する点があるか確認
                        next_in_range = next_x_range & next_y_range
                        next1_in_range = next1_x_range & next1_y_range
                        next2_in_range = next2_x_range & next2_y_range
                        # if np.any(next_in_range):   # 範囲内の点が存在
                        next_close_pts = next_pts[next_in_range]
                        next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        next1_close_pts = next1_pts[next1_in_range]
                        next1_close_clr = next1_clr[next1_in_range]  # 後フレームの色を取得
                        next2_close_pts = next2_pts[next2_in_range]
                        next2_close_clr = next2_clr[next2_in_range]  # 後フレームの色を取得

                        # # 三次元距離を計算
                        next_distances = calc_distance(point,next_close_pts)
                        next1_distances = calc_distance(point,next1_close_pts)
                        next2_distances = calc_distance(point,next2_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next1_close_clr[k]) for k, dist in enumerate(next1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next2_close_clr[k]) for k, dist in enumerate(next2_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color
                            
                if i ==1 and len(dirty_index) > 10:
                    pre_mask = masks[i-1].reshape(-1)
                    pre_pts = points[i-1].reshape(-1, 3)[next_mask].cpu().numpy()
                    pre_clr = images[i-1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_mask = masks[i+1].reshape(-1)
                    next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next1_mask = masks[i+2].reshape(-1)
                    next1_pts = points[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next1_clr = images[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next2_mask = masks[i+3].reshape(-1)
                    next2_pts = points[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                    next2_clr = images[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                   
                    # 現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        pre_x_range = (pre_pts[:, 0] >= x - 0.1) & (pre_pts[:, 0] <= x + 0.1)
                        pre_y_range = (pre_pts[:, 1] >= y - 0.1) & (pre_pts[:, 1] <= y + 0.1)
                        next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                        next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)
                        next1_x_range = (next1_pts[:, 0] >= x - 0.1) & (next1_pts[:, 0] <= x + 0.1)
                        next1_y_range = (next1_pts[:, 1] >= y - 0.1) & (next1_pts[:, 1] <= y + 0.1)
                        next2_x_range = (next2_pts[:, 0] >= x - 0.1) & (next2_pts[:, 0] <= x + 0.1)
                        next2_y_range = (next2_pts[:, 1] >= y - 0.1) & (next2_pts[:, 1] <= y + 0.1)

                        # 条件に一致する点があるか確認
                        pre_in_range = pre_x_range & pre_y_range
                        next_in_range = next_x_range & next_y_range
                        next1_in_range = next1_x_range & next1_y_range
                        next2_in_range = next2_x_range & next2_y_range
                        # if np.any(next_in_range):   # 範囲内の点が存在
                        pre_close_pts = pre_pts[pre_in_range]
                        pre_close_clr = pre_clr[pre_in_range]  # 前フレームの色を取得
                        next_close_pts = next_pts[next_in_range]
                        next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        next1_close_pts = next1_pts[next1_in_range]
                        next1_close_clr = next1_clr[next1_in_range]  # 後フレームの色を取得
                        next2_close_pts = next2_pts[next2_in_range]
                        next2_close_clr = next2_clr[next2_in_range]  # 後フレームの色を取得

                        # # 三次元距離を計算
                        pre_distances = calc_distance(point,pre_close_pts)
                        next_distances = calc_distance(point,next_close_pts)
                        next1_distances = calc_distance(point,next1_close_pts)
                        next2_distances = calc_distance(point,next2_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next1_close_clr[k]) for k, dist in enumerate(next1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next2_close_clr[k]) for k, dist in enumerate(next2_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color 
                            
                            
                            
                if i ==2 and len(dirty_index) > 10:
                    pre1_mask = masks[i-2].reshape(-1)
                    pre1_pts = points[i-2].reshape(-1, 3)[next_mask].cpu().numpy()
                    pre1_clr = images[i-2].reshape(-1, 3)[next_mask].cpu().numpy()
                    pre_mask = masks[i-1].reshape(-1)
                    pre_pts = points[i-1].reshape(-1, 3)[next_mask].cpu().numpy()
                    pre_clr = images[i-1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_mask = masks[i+1].reshape(-1)
                    next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next1_mask = masks[i+2].reshape(-1)
                    next1_pts = points[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next1_clr = images[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next2_mask = masks[i+3].reshape(-1)
                    next2_pts = points[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                    next2_clr = images[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                   
                    # 現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        pre1_x_range = (pre1_pts[:, 0] >= x - 0.1) & (pre1_pts[:, 0] <= x + 0.1)
                        pre1_y_range = (pre1_pts[:, 1] >= y - 0.1) & (pre1_pts[:, 1] <= y + 0.1)
                        pre_x_range = (pre_pts[:, 0] >= x - 0.1) & (pre_pts[:, 0] <= x + 0.1)
                        pre_y_range = (pre_pts[:, 1] >= y - 0.1) & (pre_pts[:, 1] <= y + 0.1)
                        next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                        next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)
                        next1_x_range = (next1_pts[:, 0] >= x - 0.1) & (next1_pts[:, 0] <= x + 0.1)
                        next1_y_range = (next1_pts[:, 1] >= y - 0.1) & (next1_pts[:, 1] <= y + 0.1)
                        next2_x_range = (next2_pts[:, 0] >= x - 0.1) & (next2_pts[:, 0] <= x + 0.1)
                        next2_y_range = (next2_pts[:, 1] >= y - 0.1) & (next2_pts[:, 1] <= y + 0.1)

                        # 条件に一致する点があるか確認
                        pre1_in_range = pre1_x_range & pre1_y_range
                        pre_in_range = pre_x_range & pre_y_range
                        next_in_range = next_x_range & next_y_range
                        next1_in_range = next1_x_range & next1_y_range
                        next2_in_range = next2_x_range & next2_y_range
                        # if np.any(next_in_range):   # 範囲内の点が存在
                        pre1_close_pts = pre1_pts[pre1_in_range]
                        pre1_close_clr = pre1_clr[pre1_in_range]  # 前フレームの色を取得
                        pre_close_pts = pre_pts[pre_in_range]
                        pre_close_clr = pre_clr[pre_in_range]  # 前フレームの色を取得
                        next_close_pts = next_pts[next_in_range]
                        next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        next1_close_pts = next1_pts[next1_in_range]
                        next1_close_clr = next1_clr[next1_in_range]  # 後フレームの色を取得
                        next2_close_pts = next2_pts[next2_in_range]
                        next2_close_clr = next2_clr[next2_in_range]  # 後フレームの色を取得

                        # # 三次元距離を計算
                        pre1_distances = calc_distance(point,pre1_close_pts)
                        pre_distances = calc_distance(point,pre_close_pts)
                        next_distances = calc_distance(point,next_close_pts)
                        next1_distances = calc_distance(point,next1_close_pts)
                        next2_distances = calc_distance(point,next2_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(pre1_close_clr[k]) for k, dist in enumerate(pre1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next1_close_clr[k]) for k, dist in enumerate(next1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next2_close_clr[k]) for k, dist in enumerate(next2_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color 
                
                #if i > 0 and len(dirty_index)-1 > i:  # 最初のフレームは比較対象がない
                if i > 2 and len(dirty_index) > 10 and len(dirty_index)-3 > i:  # 最初のフレームは比較対象がない
                    prev2_mask = masks[i-3].reshape(-1)
                    prev2_pts = points[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev2_clr = images[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev1_mask = masks[i-2].reshape(-1)
                    prev1_pts = points[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev1_clr = images[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev_mask = masks[i-1].reshape(-1)
                    prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    next_mask = masks[i+1].reshape(-1)
                    next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next1_mask = masks[i+2].reshape(-1)
                    next1_pts = points[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next1_clr = images[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next2_mask = masks[i+3].reshape(-1)
                    next2_pts = points[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                    next2_clr = images[i+3].reshape(-1, 3)[next2_mask].cpu().numpy()
                    
                    

                    #現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        pre2_x_range = (prev2_pts[:, 0] >= x - 0.1) & (prev2_pts[:, 0] <= x + 0.1)
                        pre2_y_range = (prev2_pts[:, 1] >= y - 0.1) & (prev2_pts[:, 1] <= y + 0.1)
                        pre1_x_range = (prev1_pts[:, 0] >= x - 0.1) & (prev1_pts[:, 0] <= x + 0.1)
                        pre1_y_range = (prev1_pts[:, 1] >= y - 0.1) & (prev1_pts[:, 1] <= y + 0.1)
                        pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                        pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)
                        next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                        next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)
                        next1_x_range = (next1_pts[:, 0] >= x - 0.1) & (next1_pts[:, 0] <= x + 0.1)
                        next1_y_range = (next1_pts[:, 1] >= y - 0.1) & (next1_pts[:, 1] <= y + 0.1)
                        next2_x_range = (next2_pts[:, 0] >= x - 0.1) & (next2_pts[:, 0] <= x + 0.1)
                        next2_y_range = (next2_pts[:, 1] >= y - 0.1) & (next2_pts[:, 1] <= y + 0.1)

                        # 条件に一致する点があるか確認
                        pre2_in_range = pre2_x_range & pre2_y_range
                        pre1_in_range = pre1_x_range & pre1_y_range
                        pre_in_range = pre_x_range & pre_y_range
                        next_in_range = next_x_range & next_y_range
                        next1_in_range = next1_x_range & next1_y_range
                        next2_in_range = next2_x_range & next2_y_range
                        
                        
                        #if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在  ##修正必要
                        pre2_close_pts = prev2_pts[pre2_in_range]
                        pre2_close_clr = prev2_clr[pre2_in_range]  # 前フレームの色を取得
                        pre1_close_pts = prev1_pts[pre1_in_range]
                        pre1_close_clr = prev1_clr[pre1_in_range]  # 前フレームの色を取得
                        pre_close_pts = prev_pts[pre_in_range]
                        pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得
                        next_close_pts = next_pts[next_in_range]
                        next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        next1_close_pts = next1_pts[next1_in_range]
                        next1_close_clr = next1_clr[next1_in_range]  # 後フレームの色を取得
                        next2_close_pts = next2_pts[next2_in_range]
                        next2_close_clr = next2_clr[next2_in_range]  # 後フレームの色を取得

                        # # 三次元距離を計算
                        pre2_distances = calc_distance(point,pre2_close_pts)
                        pre1_distances = calc_distance(point,pre1_close_pts)
                        pre_distances = calc_distance(point,pre_close_pts)
                        next_distances = calc_distance(point,next_close_pts)
                        next1_distances = calc_distance(point,next1_close_pts)
                        next2_distances = calc_distance(point,next2_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(pre2_close_clr[k]) for k, dist in enumerate(pre2_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre1_close_clr[k]) for k, dist in enumerate(pre1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next1_close_clr[k]) for k, dist in enumerate(next1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next2_close_clr[k]) for k, dist in enumerate(next2_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color
                        
                        
                        
                        
                        #######検出された点の可視化##########################################################    
                    #     #if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在
                    #     # pre_close_pts = prev_pts[pre_in_range]
                    #     # pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得
                    #     next_close_pts = next_pts[next_in_range]
                    #     next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        
                    #     next_in_range_indices = np.where(next_in_range)[0]

                    #     # # 三次元距離を計算
                    #     # pre_distances = calc_distance(point,pre_close_pts)
                    #     next_distances = calc_distance(point,next_close_pts)
                    #     close_clr_list = []
                    #     # close_clr_list = [pre_close_clr[k] for k, dist in enumerate(pre_distances) if dist < 0.1]
                    #     close_clr_list = [k for k, dist in enumerate(next_distances) if dist < 0.02]
                        
                            
                                
                    #     if j == 100:
                    #         if np.any(next_in_range):   # 範囲内の点が存在
                    #             next_clr[:] = 1.0
                                
                    #             ######選択範囲の可視化######
                    #             next_clr[next_in_range] = [0.5,1.0,0.5]  # 後フレームの色を取得
                                
                    #             ######選出された点の可視化######
                    #             for k in close_clr_list:
                    #                 next_clr[next_in_range_indices[k]] = [0.0,0.0,1.0]
                                
                    #             ######対象とする三次元点の可視化######
                    #             if i==7:
                    #                 clr[:] = 1.0
                    #                 clr[j] = [1.0,0.0,0.0]
                                
                    #             #if droid_visualization.next_colors is not None and droid_visualization.count == i and i  == 8:
                    #             if droid_visualization.next_colors is not None and droid_visualization.count == i:
                    #                 if j < len(droid_visualization.next_colors):
                    #                     clr = droid_visualization.next_colors
                                        
                    #             droid_visualization.next_colors = next_clr
                    #             droid_visualization.count = i+1
                                
                    # if i != 8 and i != 7:
                    #     clr[:] = 1.0
                        #####################################################################
                                        
                            
                                
                                
                     
                     
                     
                # if i == len(dirty_index)-1:
                #     prev_mask = masks[i-1].reshape(-1)
                #     prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                #     prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    
                #     # 現在のフレームの点ごとにチェック
                #     for j, point in enumerate(pts):
                #         x, y, z = point
                #         pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                #         pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)

                #         # 条件に一致する点があるか確認
                #         pre_in_range = pre_x_range & pre_y_range
                #         if np.any(pre_in_range):   # 範囲内の点が存在
                #             pre_close_pts = prev_pts[pre_in_range]
                #             pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得

                #             # # 三次元距離を計算
                #             pre_distances = calc_distance(point,pre_close_pts)
                #             close_clr_list = []
                #             close_clr_list = [pre_close_clr[k] for k, dist in enumerate(pre_distances) if dist < 0.1]
                #             # 最も頻出する色を決定
                #             if len(close_clr_list) > 0:
                #                 # most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                #                 close_clr_list.append(clr[j])
                #                 most_common_color = Counter(map(tuple, close_clr_list)).most_common(1)[0][0]
                #                 clr[j] = most_common_colork, dist in enumerate(distances):
                #             #     if dist < 0.1:
                #             #         close_clr_list.append(pre_close_clr[k])
                #             #close_clr_list = [np.where(dist < 0.1, pre_close_clr[k]) for k, dist in enumerate(distances)]
                #             close_clr_list = [pre_close_clr[k] for k, dist in enumerate(pre_distances) if dist < 0.1]
                #             close_clr_list = [next_close_clr[k] for k, dist in enumerate(next_distances) if dist < 0.1]
                #             # 最も頻出する色を決定
                #             if len(close_clr_list) > 0:
                #                 # most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                #                 close_clr_list.append(clr[j])
                #                 most_common_color = Counter(map(tuple, close_clr_list)).most_common(1)[0][0]
                #                 clr[j] = most_common_color
                     
                     
                     
                # if i == len(dirty_index)-1:
                #     prev_mask = masks[i-1].reshape(-1)
                #     prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                #     prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    
                #     # 現在のフレームの点ごとにチェック
                #     for j, point in enumerate(pts):
                #         x, y, z = point
                #         pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                #         pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)

                #         # 条件に一致する点があるか確認
                #         pre_in_range = pre_x_range & pre_y_range
                #         # if np.any(pre_in_range):   # 範囲内の点が存在
                #         pre_close_pts = prev_pts[pre_in_range]
                #         pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得

                #         # # 三次元距離を計算
                #         pre_distances = calc_distance(point,pre_close_pts)
                #         close_clr_list = []
                #         close_clr_list = [pre_close_clr[k] for k, dist in enumerate(pre_distances) if dist < 0.02]
                #         # 最も頻出する色を決定
                #         if len(close_clr_list) > 0:
                #             # most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                #             close_clr_list.append(clr[j])
                #             most_common_color = Counter(map(tuple, close_clr_list)).most_common(1)[0][0]
                #             clr[j] = most_common_color
                
                
                
                
                if i == len(dirty_index)-3 and len(dirty_index) > 10:
                    prev2_mask = masks[i-3].reshape(-1)
                    prev2_pts = points[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev2_clr = images[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev1_mask = masks[i-2].reshape(-1)
                    prev1_pts = points[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev1_clr = images[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev_mask = masks[i-1].reshape(-1)
                    prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    next_mask = masks[i+1].reshape(-1)
                    next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next1_mask = masks[i+2].reshape(-1)
                    next1_pts = points[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()
                    next1_clr = images[i+2].reshape(-1, 3)[next1_mask].cpu().numpy()

                    #現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        pre2_x_range = (prev2_pts[:, 0] >= x - 0.1) & (prev2_pts[:, 0] <= x + 0.1)
                        pre2_y_range = (prev2_pts[:, 1] >= y - 0.1) & (prev2_pts[:, 1] <= y + 0.1)
                        pre1_x_range = (prev1_pts[:, 0] >= x - 0.1) & (prev1_pts[:, 0] <= x + 0.1)
                        pre1_y_range = (prev1_pts[:, 1] >= y - 0.1) & (prev1_pts[:, 1] <= y + 0.1)
                        pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                        pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)
                        next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                        next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)
                        next1_x_range = (next1_pts[:, 0] >= x - 0.1) & (next1_pts[:, 0] <= x + 0.1)
                        next1_y_range = (next1_pts[:, 1] >= y - 0.1) & (next1_pts[:, 1] <= y + 0.1)

                        # 条件に一致する点があるか確認
                        pre2_in_range = pre2_x_range & pre2_y_range
                        pre1_in_range = pre1_x_range & pre1_y_range
                        pre_in_range = pre_x_range & pre_y_range
                        next_in_range = next_x_range & next_y_range
                        next1_in_range = next1_x_range & next1_y_range
                        #if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在  ##修正必要
                        pre2_close_pts = prev2_pts[pre2_in_range]
                        pre2_close_clr = prev2_clr[pre2_in_range]  # 前フレームの色を取得
                        pre1_close_pts = prev1_pts[pre1_in_range]
                        pre1_close_clr = prev1_clr[pre1_in_range]  # 前フレームの色を取得
                        pre_close_pts = prev_pts[pre_in_range]
                        pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得
                        next_close_pts = next_pts[next_in_range]
                        next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        next1_close_pts = next1_pts[next1_in_range]
                        next1_close_clr = next1_clr[next1_in_range]  # 後フレームの色を取得

                        # # 三次元距離を計算
                        pre2_distances = calc_distance(point,pre2_close_pts)
                        pre1_distances = calc_distance(point,pre1_close_pts)
                        pre_distances = calc_distance(point,pre_close_pts)
                        next_distances = calc_distance(point,next_close_pts)
                        next1_distances = calc_distance(point,next1_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(pre2_close_clr[k]) for k, dist in enumerate(pre2_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre1_close_clr[k]) for k, dist in enumerate(pre1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next1_close_clr[k]) for k, dist in enumerate(next1_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color
                            
                
                if i == len(dirty_index)-2 and len(dirty_index) > 10:
                    prev2_mask = masks[i-3].reshape(-1)
                    prev2_pts = points[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev2_clr = images[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev1_mask = masks[i-2].reshape(-1)
                    prev1_pts = points[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev1_clr = images[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev_mask = masks[i-1].reshape(-1)
                    prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    next_mask = masks[i+1].reshape(-1)
                    next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                   
                    #現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        pre2_x_range = (prev2_pts[:, 0] >= x - 0.1) & (prev2_pts[:, 0] <= x + 0.1)
                        pre2_y_range = (prev2_pts[:, 1] >= y - 0.1) & (prev2_pts[:, 1] <= y + 0.1)
                        pre1_x_range = (prev1_pts[:, 0] >= x - 0.1) & (prev1_pts[:, 0] <= x + 0.1)
                        pre1_y_range = (prev1_pts[:, 1] >= y - 0.1) & (prev1_pts[:, 1] <= y + 0.1)
                        pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                        pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)
                        next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                        next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)

                        # 条件に一致する点があるか確認
                        pre2_in_range = pre2_x_range & pre2_y_range
                        pre1_in_range = pre1_x_range & pre1_y_range
                        pre_in_range = pre_x_range & pre_y_range
                        next_in_range = next_x_range & next_y_range
                        #if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在  ##修正必要
                        pre2_close_pts = prev2_pts[pre2_in_range]
                        pre2_close_clr = prev2_clr[pre2_in_range]  # 前フレームの色を取得
                        pre1_close_pts = prev1_pts[pre1_in_range]
                        pre1_close_clr = prev1_clr[pre1_in_range]  # 前フレームの色を取得
                        pre_close_pts = prev_pts[pre_in_range]
                        pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得
                        next_close_pts = next_pts[next_in_range]
                        next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得
                        # # 三次元距離を計算
                        pre2_distances = calc_distance(point,pre2_close_pts)
                        pre1_distances = calc_distance(point,pre1_close_pts)
                        pre_distances = calc_distance(point,pre_close_pts)
                        next_distances = calc_distance(point,next_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(pre2_close_clr[k]) for k, dist in enumerate(pre2_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre1_close_clr[k]) for k, dist in enumerate(pre1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color
                            
                            
                if i == len(dirty_index)-1 and len(dirty_index) > 10:
                    prev2_mask = masks[i-3].reshape(-1)
                    prev2_pts = points[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev2_clr = images[i-3].reshape(-1, 3)[prev2_mask].cpu().numpy()
                    prev1_mask = masks[i-2].reshape(-1)
                    prev1_pts = points[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev1_clr = images[i-2].reshape(-1, 3)[prev1_mask].cpu().numpy()
                    prev_mask = masks[i-1].reshape(-1)
                    prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                    prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                   
                    #現在のフレームの点ごとにチェック
                    for j, point in enumerate(pts):
                        x, y, z = point
                        pre2_x_range = (prev2_pts[:, 0] >= x - 0.1) & (prev2_pts[:, 0] <= x + 0.1)
                        pre2_y_range = (prev2_pts[:, 1] >= y - 0.1) & (prev2_pts[:, 1] <= y + 0.1)
                        pre1_x_range = (prev1_pts[:, 0] >= x - 0.1) & (prev1_pts[:, 0] <= x + 0.1)
                        pre1_y_range = (prev1_pts[:, 1] >= y - 0.1) & (prev1_pts[:, 1] <= y + 0.1)
                        pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                        pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)
                     
                        # 条件に一致する点があるか確認
                        pre2_in_range = pre2_x_range & pre2_y_range
                        pre1_in_range = pre1_x_range & pre1_y_range
                        pre_in_range = pre_x_range & pre_y_range
                        #if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在  ##修正必要
                        pre2_close_pts = prev2_pts[pre2_in_range]
                        pre2_close_clr = prev2_clr[pre2_in_range]  # 前フレームの色を取得
                        pre1_close_pts = prev1_pts[pre1_in_range]
                        pre1_close_clr = prev1_clr[pre1_in_range]  # 前フレームの色を取得
                        pre_close_pts = prev_pts[pre_in_range]
                        pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得

                        # # 三次元距離を計算
                        pre2_distances = calc_distance(point,pre2_close_pts)
                        pre1_distances = calc_distance(point,pre1_close_pts)
                        pre_distances = calc_distance(point,pre_close_pts)
                        close_clr_list = []
                        close_clr_list.extend([tuple(pre2_close_clr[k]) for k, dist in enumerate(pre2_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre1_close_clr[k]) for k, dist in enumerate(pre1_distances) if dist < 0.02])
                        close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.02])
                        # 最も頻出する色を決定
                        if len(close_clr_list) > 0:
                            close_clr_list.append(tuple(clr[j]))
                            most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                            clr[j] = most_common_color
                
                #########################################
                ## 一個前のフレームの点群と比較 ###
                
                # if i > 0 and len(dirty_index)-1 > i:  # 最初のフレームは比較対象がない
                #     prev_mask = masks[i-1].reshape(-1)
                #     prev_pts = points[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                #     prev_clr = images[i-1].reshape(-1, 3)[prev_mask].cpu().numpy()
                #     next_mask = masks[i+1].reshape(-1)
                #     next_pts = points[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                #     next_clr = images[i+1].reshape(-1, 3)[next_mask].cpu().numpy()
                    

                #     # 現在のフレームの点ごとにチェック
                #     for j, point in enumerate(pts):
                #         x, y, z = point
                #         pre_x_range = (prev_pts[:, 0] >= x - 0.1) & (prev_pts[:, 0] <= x + 0.1)
                #         pre_y_range = (prev_pts[:, 1] >= y - 0.1) & (prev_pts[:, 1] <= y + 0.1)
                #         next_x_range = (next_pts[:, 0] >= x - 0.1) & (next_pts[:, 0] <= x + 0.1)
                #         next_y_range = (next_pts[:, 1] >= y - 0.1) & (next_pts[:, 1] <= y + 0.1)

                #         # 条件に一致する点があるか確認
                #         pre_in_range = pre_x_range & pre_y_range
                #         next_in_range = next_x_range & next_y_range
                #         # if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在
                #         #     pre_close_pts = prev_pts[pre_in_range]
                #         #     pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得
                #         #     next_close_pts = next_pts[next_in_range]
                #         #     next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得

                #         #     # # 三次元距離を計算
                #         #     pre_distances = calc_distance(point,pre_close_pts)
                #         #     next_distances = calc_distance(point,next_close_pts)
                #         #     close_clr_list = []
                #         #     close_clr_list1 = []
                #         #     # for k, dist in enumerate(distances):
                #         #     #     if dist < 0.1:
                #         #     #         close_clr_list.append(pre_close_clr[k])
                #         #     #close_clr_list = [np.where(dist < 0.1, pre_close_clr[k]) for k, dist in enumerate(distances)]
                #         #     close_clr_list = [pre_close_clr[k] for k, dist in enumerate(pre_distances) if dist < 0.05]
                #         #     close_clr_list1 = [next_close_clr[k] for k, dist in enumerate(next_distances) if dist < 0.05] 
                #         #     close_clr_list.append([next_close_clr[k] for k, dist in enumerate(next_distances) if dist < 0.05])
                #         #     # 最も頻出する色を決定
                #         #     if len(close_clr_list) > 0:
                #         #         # most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                #         #         close_clr_list.append(clr[j])
                #         #         most_common_color = Counter(map(tuple, close_clr_list)).most_common(1)[0][0]
                #         #         clr[j] = most_common_color
                        
                #     if np.any(pre_in_range) and np.any(next_in_range):   # 範囲内の点が存在
                #         pre_close_pts = prev_pts[pre_in_range]
                #         pre_close_clr = prev_clr[pre_in_range]  # 前フレームの色を取得
                #         next_close_pts = next_pts[next_in_range]
                #         next_close_clr = next_clr[next_in_range]  # 後フレームの色を取得

                #         # 三次元距離を計算
                #         pre_distances = calc_distance(point, pre_close_pts)
                #         next_distances = calc_distance(point, next_close_pts)
                #         close_clr_list = []

                #         close_clr_list.extend([tuple(pre_close_clr[k]) for k, dist in enumerate(pre_distances) if dist < 0.05])
                #         close_clr_list.extend([tuple(next_close_clr[k]) for k, dist in enumerate(next_distances) if dist < 0.05])

                #         # 最も頻出する色を決定
                #         if len(close_clr_list) > 0:
                #             close_clr_list.append(tuple(clr[j]))
                #             most_common_color = Counter(close_clr_list).most_common(1)[0][0]
                #             clr[j] = most_common_color
                                
                                    
                #########################################
                
                
                #########三次元点群の距離を計算して、近い点を赤色にする処理#########
                #clr[:] = 0.5  # 全ての要素を 0.5 に設定
                # with open('pixel_clr', "w") as f:
                #     for clr_value in clr:
                #         f.write(f"{clr_value[0]:.6f}, {clr_value[1]:.6f}, {clr_value[2]:.6f}\n")
                #     f.write("\n")
                
                
                # if i > 2:
                #     start = time.perf_counter()
                #     # for x, point in enumerate(pts):
                #         # for y, pre_point in enumerate(points[i-1].reshape(-1, 3)[mask].cpu().numpy()):
                #             # if np.linalg.norm(point - pre_point) < 0.001:
                #         # if calc_distance(point,points[i-1].reshape(-1, 3)[mask].cpu().numpy()) < 0.01:
                #     list1 = [x for x, point in enumerate(pts) for y, pre_point in enumerate(points[i-1].reshape(-1, 3)[mask].cpu().numpy()) if calc_distance1(point,pre_point) < 0.1]
                #     if np.any(list1):
                #         for x in list1:
                #             clr[x] = [0.5,1.0,0.5]
                #                 # pre_data = np.array([[i-1,pre_point,clr[y]]],dtype=object)
                #                 # now_data = np.array([[i,point,clr[x]]],dtype=object)
                #                 # near_point = np.vstack((pre_data,now_data),dtype=object)
                #                 # if all(clr[x]==clr[y])==True:
                #                 #     print(f"same color")
                #                 #print(f"clr[x]: {clr[x]}")
                #                 #print(f"near!!!!!!!point: {near_point}")
                #                 # print(f"near!!!!!!!point: {point}")
                #                 # print(f"")
                #         with open('near_point_sum', "a") as f:
                #             f.write(f"{len(list1)}\n")
                #             print(f"list1: {len(list1)}")
                #     end = time.perf_counter()
                #     print('{:.4f}'.format((end-start)))
                        
                #     print('{:.3f}'.form                print(f"images[i].shape: {images[i].shape}")
                            
                
                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            #Open3D のイベントを処理し、ビジュアライザを再描画
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()
