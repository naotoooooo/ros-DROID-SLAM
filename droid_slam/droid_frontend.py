import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius


    #各フレームを処理し、因子グラフの更新やキーフレームの管理を行う
    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None: # 相関値があれば、古くなった因子をグラフから削除、メモリと計算効率のため
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        #現在のフレームの近隣フレーム間に新しい近接因子を追加 
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        #深度推定を更新、センサーから有効なデータが得られれば使用し、なければ既存の推定値を保持 task:視差について
        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        #因子グラフの局所最適化を複数回実行
        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        #フレーム間の距離を計算し、動きが十分かを評価
        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        #距離が閾値未満の場合、直近のキーフレームを削除。
        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        #動きが十分な場合、追加の最適化を実行
        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        #次のフレームの初期姿勢と深度を設定
        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        #可視化用の更新フラグを設定
        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True


    # 因子グラフを構築して最適化
    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        # 時系列的に近いフレームを接続し、システムの初期化や安定化に使用
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        # 因子グラフの局所最適化を複数回実行 update operator:fullBA
        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # 最初のフレームに近接因子を追加
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        # 再度、因子グラフを最適化 update operator:fullBA
        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # 最後のフレームから初期姿勢と深度を推定 task:どのように初期化しているか調べるのあり
        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # 初期化が完了したフラグを設定し、最後のフレームの状態を保存
        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        # スレッドセーフな方法でビデオの状態を更新
        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        # ウォームアップフェーズ中に作られた因子を削除
        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization 因子グラフを構築して最適化 
        if not self.is_initialized and self.video.counter.value == self.warmup:  #warmup==8
            self.__initialize()
            
        # do update 因子グラフの更新やキーフレームの管理 キーフレームが追加されたときにupdateを実行
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
