# from topoLoss import *
import torch

from TDFMain_pytorch import *
from betti_compute import betti_number

n_fix = 0
n_remove = 0
pers_thd_lh = 0.03
pers_thd_gt = 0.03


def getPers(likelihood, groundtruth):
    pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(likelihood)

    if (pd_lh.shape[0] > 0):
        lh_pers = pd_lh[:, 1] - pd_lh[:, 0]
        lh_pers_valid = lh_pers[np.where(lh_pers > pers_thd_lh)]
    else:
        lh_pers = np.array([])
        lh_pers_valid = np.array([])

    pd_gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom_gt(groundtruth)

    if (pd_gt.shape[0] > 0):  # number of critical points (n, 2)
        gt_pers = pd_gt[:, 1] - pd_gt[:, 0]
        gt_pers_valid = gt_pers[np.where(gt_pers > pers_thd_gt)]
    else:
        gt_pers = np.array([])
        gt_pers_valid = np.array([])

    return pd_lh, bcp_lh, dcp_lh, pd_gt, bcp_gt, dcp_gt, lh_pers, lh_pers_valid, gt_pers, gt_pers_valid


def get_critical_points(likelihoodMap, label, predict, args):
    topo_size = args.topo_size
    gt_dmap = label.to(args.device)
    et_dmap = likelihoodMap

    for y in range(0, gt_dmap.shape[0], topo_size):
        for x in range(0, gt_dmap.shape[1], topo_size):
            likelihood = et_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])]
            groundtruth = gt_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                          x:min(x + topo_size, gt_dmap.shape[1])]
            binary = predict[y:min(y + topo_size, gt_dmap.shape[0]),
                     x:min(x + topo_size, gt_dmap.shape[1])]

            # print('likelihood', likelihood.shape, 'groundtruth', groundtruth.shape, 'binaryPredict', binary.shape)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)

            if (torch.min(likelihood) == 1 or torch.max(likelihood) == 0): continue
            if (torch.min(groundtruth) == 1 or torch.max(groundtruth) == 0): continue
            if groundtruth_betti_number == 0: continue
            if (abs(predict_betti_number - groundtruth_betti_number) / groundtruth_betti_number) < 0.4:
                continue
            if (len(likelihood.shape) < 2 or len(groundtruth.shape) < 2):
                continue
            print('row: ', y, 'col: ', x)

            pd_lh, bcp_lh, dcp_lh, pd_gt, bcp_gt, dcp_gt, lh_pers, lh_pers_valid, gt_pers, gt_pers_valid = getPers(
                likelihood, groundtruth)
            print(pd_lh, bcp_lh, dcp_lh)
            ss


def convert_topo(likelihoodMaps, labels, predict_all, args):
    for i, likelihoodMap in enumerate(likelihoodMaps):
        cp = get_critical_points(likelihoodMap, labels[i], predict_all[i], args)
