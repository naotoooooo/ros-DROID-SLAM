import torch
import lietorch
import numpy as np

# import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops
import cv2

class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    # 既存のエッジや非アクティブエッジと重複するエッジを削除
    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        # keepはすべての値がFalseに初期化されたテンソル
        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        # iiとjjの各エッジについて、esetに存在しない場合にkeepをTrueに設定
        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        # keepがTrueのエッジのみを返す
        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    # @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges 
        # 既存のエッジや非アクティブエッジと重複するエッジを削除
        ii, jj = self.__filter_repeated_edges(ii, jj)

        # 追加するエッジがない場合、何もせず関数を終了
        if ii.shape[0] == 0:
            return

        # place limit on number of factors 最大エッジ数の制限   ii.shape[0]はエッジの数
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True) # 最も古いエッジを削除=非アクティブ因子に保存

        # 指定されたフレームペア ii のネットワーク特徴量を取得
        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        # with torch.cuda.amp.autocast(enabled=False):
        target, _ = self.video.reproject(ii, jj)
        weight = torch.zeros_like(target)

        # add edges to factor graph
        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    # @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        # 削除する因子を非アクティブ因子に保存
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        #因子の削除
        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        # 相関データ 特徴量データ p*ij 重みの削除
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    # @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """


        with self.video.get_lock():
            self.video.tstamp[ix] = self.video.tstamp[ix+1]
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]
            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        # 非アクティブ因子をすべて削除
        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    #因子グラフを更新し、カメラポーズや深度の最適化を行う update_op~DBA
    # @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        # with torch.cuda.amp.autocast(enabled=False):
        #グラフ内のエッジ（フレームペア）に基づいて再投影座標を計算 coords1:pij   mask:有効な3Dポイントをマスク
        coords1, mask = self.video.reproject(self.ii, self.jj)
        #フレーム間の相対的な動きを計算 task:target-coords1をする意味がわからない
        motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
        motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
        
        # correlation features 画像座標系pijに関する相関値
        corr = self.corr(coords1)

        #ConvGRU
        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)  #UpdateModuleのforwardメソッドを呼び出す
            
        # 現在の因子グラフ内で最も古い（最小の）始点フレームインデックスを取得
        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        # target=p*ij
        # with torch.cuda.amp.autocast(enabled=False):
        self.target = coords1 + delta.to(dtype=torch.float)
        self.weight = weight.to(dtype=torch.float)

        ht, wd = self.coords0.shape[0:2]
        self.damping[torch.unique(self.ii)] = damping

        # 非アクティブな因子を含める場合 task:非アクティブ因子とは？
        if use_inactive:  
            m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3) # 開始フレームより三個前の非アクティブ因子を含むフレームを採用
            ii = torch.cat([self.ii_inac[m], self.ii], 0)
            jj = torch.cat([self.jj_inac[m], self.jj], 0)
            target = torch.cat([self.target_inac[:,m], self.target], 1)
            weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

        else:
            ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


        damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

        target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
        weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

        # dense bundle adjustment
        self.video.ba(target, weight, damping, ii, jj, t0, t1, 
            itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
        if self.upsample:
            self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1

        
    # エッジ情報をtxtファイルに保存するスクリプト
    def save_edge_pairs_to_file(self,ii, jj, filename="edges.txt"):
    # """
    # エッジが繋がれたフレームペアをファイルに保存
    # 左側にフレーム番号、右側にペア番号を列挙
    # """
        from collections import defaultdict

        # フレーム番号ごとにペア番号をまとめる
        edges_dict = defaultdict(list)
        for start, end in zip(ii.cpu().numpy(), jj.cpu().numpy()):
            edges_dict[start].append(end)
            edges_dict[end].append(start)  # 双方向エッジも記録

        # 重複を排除してソート
        for key in edges_dict:
            edges_dict[key] = sorted(set(edges_dict[key]))

        # ファイルに保存
        with open(filename, "w") as f:
            for frame, pairs in sorted(edges_dict.items()):
                pairs_str = " ".join(map(str, pairs))
                f.write(f"{frame}: {pairs_str}\n")

        print(f"Edges saved to {filename}")
        
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
        return

    # 使用例
    # display_frame_at_t(self.video, t=100)



    # @torch.cuda.amp.autocast(enabled=False)
    # メモリ効率の高いアプローチを使用しながら因子グラフを更新する関数 大規模なデータセットや限られたGPUメモリで作業する場合に役立つ
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value # 現在のフレーム数

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        # グローバルバンドル調整を実行するために、指定されたステップ数(steps)のループを開始  各ステップでは、グラフ接続を改良することでポーズ推定値が向上
        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            # with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 8
            print("self.jj.max()", self.jj.max())
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                # with torch.cuda.amp.autocast(enabled=True):
                 
                net, delta, weight, damping, upmask = \
                    self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                if self.upsample:
                    self.video.upsample(torch.unique(iis), upmask)
                
                #型を揃えた
                net = net.half()  # FP16に変換
                self.net = self.net.half()  # 宛先もFP16に変換


                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True
            # フレームペアの番号を出力
            # for start, end in zip(self.ii.cpu().numpy(), self.jj.cpu().numpy()):
            #     print(f"Edge between frames: {start} -> {end}")
            # 使用例
        # self.save_edge_pairs_to_file(self.ii, self.jj, filename="edges.txt")
            
        


    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        # フレームペアの生成
        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        # 半径 r 以内の隣接フレームのみ選択
        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    # 因子グラフにおける近接ベースのエッジの追加
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)
        # ii,jj フレームペア生成
        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        self.save_edge_pairs_to_file(ii, jj, filename="edges_all.txt")
        # self.display_frame_at_t(self.video, t)
        
        
        D =self.video.distance(ii, jj, beta=beta).cpu().numpy()
        filename = "distances.txt"
        # ファイルに保存
        with open(filename, "w") as f:
            for i, j, dist in zip(ii.cpu().numpy(), jj.cpu().numpy(), D):
                f.write(f"Frame Pair ({i}, {j}): Distance = {dist}\n")

        print(f"Distances saved to {filename}")
        
        d = self.video.distance(ii, jj, beta=beta) # フレームペア間の距離を計算
        
        d[ii - rad < jj] = np.inf # radより近いフレームは無効
        d[d > 100] = np.inf # 距離が100より離れているフレームは無効

        # アクティブ、不良、非アクティブな接続のインデックスをii1,jj1に統合
        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)

        # 非最大値抑制(NMS)を適用　冗長なペアを削除
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = [] # エッジを保存するための空のリスト
        for i in range(t0, t):
            if self.video.stereo: # ステレオモードが有効になっている場合は、フレームにステレオエッジを追加
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i): # 時間的に近いすべてのペアにエッジを追加
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        # 距離の値が小さい順に反復処理して、しきい値の距離までのエッジを追加
        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue
            
            # 十分なエッジが追加されると接続の追加が停止
            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            # ローカルエリアでさらに非最大抑制(NMS)を適用
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        #print("\nes", len(es))
        # エッジリスト(es)を因子グラフに追加し、指定されたフレーム範囲の近接接続を確定                     
        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)

        self.add_factors(ii, jj, remove)
        self.save_edge_pairs_to_file(self.ii, self.jj, filename="edges_thresh50.txt")
        
        
        return len(es)
