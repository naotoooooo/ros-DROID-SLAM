#画像フレームに基づいてカメラの位置や動きを推定するために最適化処理を行う
import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
import record
import cv2


# DROID-SLAM のバックエンド最適化タスクを処理
class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample                # 深度マップの解像度を高めるためにアップサンプリングを適用するかどうかを決定
        self.beta = args.beta                        # 重み係数 最適化による変換と回転の影響を制御するために使用
        self.backend_thresh = args.backend_thresh    # フレーム間の接続を含めるしきい値
        self.backend_radius = args.backend_radius    # 因子グラフで近接接続を確立するための半径
        self.backend_nms = args.backend_nms          # 非最大抑制パラメータ 冗長エッジを削減する
    
    def display_frame_at_t(self,video, t=100):
    # """
    # 指定した時刻 t のフレームを画像として表示する
    # """
        # tのフレーム画像を取得
        frame = video.images[t].cpu().numpy()  # Tensor を NumPy 配列に変換
        frame = frame.transpose(1, 2, 0)       # (C, H, W) → (H, W, C)

        # 表示
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV 用に変換
        cv2.imshow(f"Frame at t={t}", frame_bgr)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        #return
        
   
        
    @torch.no_grad() # 勾配計算を無効にしてメモリ使用量を最適化するデコレータ
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value  # ビデオから現在のフレーム数を取得
        if not self.video.stereo and not torch.any(self.video.disps_sens): # ビデオデータの正規化
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t, upsample=self.upsample) # factorgraphオブジェクトを作成

        # バックエンドパラメータに基づいて、ファクターグラフに近接ファクター(接続)を追加
        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)
        #edges_sum = graph.add_proximity_factors(rad=self.backend_radius, 
        #                                        nms=self.backend_nms, 
        #                                        thresh=self.backend_thresh, 
        #                                        beta=self.beta)
        # print("\nedges_sum", edges_sum)
        #filename = "edges_sum.txt"
        #record.save_number_to_file(edges_sum, filename)
        #self.display_frame_at_t(self.video, t=50)
                
        graph.update_lowmem(steps=steps)
        graph.clear_edges() # 最適化後に因子グラフ内のすべてのエッジをクリアしてリソースを解放
        self.video.dirty[:t] = True
