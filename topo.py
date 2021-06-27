import logging
import time
from PIL import Image
from TDFMain_pytorch import *
from betti_compute import betti_number
from pre_processing import *
import torch
from save_history import *

# logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

torch.set_printoptions(edgeitems=60)
np.set_printoptions(edgeitems=60)

n_fix = 0
n_remove = 0
pers_thd_lh = 0.01
pers_thd_gt = 0.03


def filterBcp(bcp, pers):
    bcp = bcp[np.where(pers > pers_thd_lh)]

    return bcp


def getPers(likelihood):
    pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(likelihood)

    if (pd_lh.shape[0] > 0):
        lh_pers = pd_lh[:, 1] - pd_lh[:, 0]
        lh_pers_valid = lh_pers[np.where(lh_pers > pers_thd_lh)]
    else:
        lh_pers = np.array([])
        lh_pers_valid = np.array([])

    bcp_lh = filterBcp(bcp_lh, lh_pers)  # comment this line to keep all critical poins

    return pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid


def getTopoFilter(criticPointMap, bcp_lh, args):
    # print('bcp_lh.shape', bcp_lh.shape)
    for coor in bcp_lh:
        if any(coor < 0) or any(coor >= criticPointMap.shape[0]) or any(coor >= criticPointMap.shape[1]): continue
        criticPointMap[int(coor[0])][int(coor[1])] = 1

    criticPointMap = addGaussianFilter(criticPointMap, args)
    criticPointMap = (criticPointMap - torch.min(criticPointMap)) / (
                torch.max(criticPointMap) - torch.min(criticPointMap)) if torch.max(
        criticPointMap) != 0 else criticPointMap

    return criticPointMap


def get_critical_points(likelihoodMap, label, groundtruth, args):
    criticPointMap = torch.zeros(likelihoodMap.shape).to(args.device)
    pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid = getPers(likelihoodMap)
    criticPointMap = getTopoFilter(criticPointMap, bcp_lh, args)
    return criticPointMap


def get_critical_points_patch(likelihoodMap, label, predict, args):
    topo_size = args.topo_size
    gt_dmap = label.to(args.device)
    et_dmap = likelihoodMap
    criticPointMapAll = torch.zeros(likelihoodMap.shape).to(args.device)
    for y in range(0, et_dmap.shape[0], topo_size):
        for x in range(0, et_dmap.shape[1], topo_size):
            likelihood = et_dmap[y:min(y + topo_size, et_dmap.shape[0]),
                         x:min(x + topo_size, et_dmap.shape[1])]
            groundtruth = gt_dmap[y:min(y + topo_size, et_dmap.shape[0]),
                          x:min(x + topo_size, et_dmap.shape[1])]
            binary = predict[y:min(y + topo_size, et_dmap.shape[0]),
                     x:min(x + topo_size, et_dmap.shape[1])]
            criticPointMap = torch.zeros(likelihood.shape).to(args.device)
            # print('likelihood', likelihood, 'groundtruth', groundtruth, 'binaryPredict', binary)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)

            if (torch.min(likelihood) == 1 or torch.max(likelihood) == 0):
                continue
            if (torch.min(groundtruth) == 1 or torch.max(groundtruth) == 0):
                continue
            if groundtruth_betti_number == 0:
                continue
            if (abs(predict_betti_number - groundtruth_betti_number) / groundtruth_betti_number) < 0.3:
                continue
            if (len(likelihood.shape) < 2 or len(groundtruth.shape) < 2):
                continue
            print('row: ', y, 'col: ', x)
            pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid = getPers(likelihood)
            criticPointMap = getTopoFilter(criticPointMap, bcp_lh, args)
            criticPointMapAll[y:min(y + topo_size, et_dmap.shape[0]),
            x:min(x + topo_size, et_dmap.shape[1])] = criticPointMap
    return criticPointMapAll


def saveForTest(img, i, type):
    img[img > 1] = 1
    img[img < 0] = 0

    img = img * 255
    img = img.detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(img)
    img.save('paper/' + str(i) + type + '.png')


def convert_topo(likelihoodMaps, labels, predict_all, args):
    dataSet = []
    for i, likelihoodMap in enumerate(likelihoodMaps):
        cp = get_critical_points(likelihoodMap, labels[i], predict_all[i], args)
        newData = torch.cat((cp.unsqueeze(0), likelihoodMap.unsqueeze(0)), dim=0)
        dataSet.append(newData)

    dataSet = torch.stack(dataSet)  # (n, 2, size, size)
    train = (dataSet, labels)
    torch.save(train, args.topo_dataset_cache)

    return train


def downsampling(likelihoodMap, args, times=2):
    if 'ISBI' in args.database:
        avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    else:
        avgpool = nn.AvgPool2d(kernel_size=3, stride=2)
    for i in range(times):
        likelihoodMap = avgpool(likelihoodMap)

    return likelihoodMap


def upsampling(likelihoodMap, times=2):
    up = [nn.Upsample(scale_factor=2, mode="bilinear"), nn.Upsample(size=(625, 625), mode="bilinear")]

    for i in range(times):
        likelihoodMap = up[i](likelihoodMap)
        # print(likelihoodMap.shape)
    return likelihoodMap


def temporalAttention(attention, iter_attention, batch, epoch):
    if epoch == 0:
        iter_attention.append(attention.detach())
    else:
        attention = attention + iter_attention[batch] * 0.1
        iter_attention[batch] = attention.detach()
    return attention, iter_attention


def oneImageAttention(query, key0, key1, key2, value):
    w, h = query.shape[1], query.shape[2]
    proj_query = query.squeeze(0).view(-1, w * h).permute(1, 0).cpu()
    i = 0
    for key in [key0, key1, key2]:
        proj_key = key.squeeze(0).view(-1, w * h).cpu()
        energy = torch.mm(proj_query, proj_key)
        softmax1D = torch.nn.Softmax(dim=-1)
        similarity = softmax1D(energy)
        # if i == 1: print(similarity)

        proj_value = value.squeeze(0).view(-1, w * h).cpu()
        score = torch.mm(proj_value, similarity.permute(1, 0))
        if i == 0: print(score, 'hhhhh')
        attention = score.view(w, h)

        attention = (attention - torch.min(attention)) / (torch.max(attention) - torch.min(attention))

        saveForTest(attention, i, 'att.png')
        i += 1


def get_attention_map(output, labels, iter_attention, args, batch, epoch, valid):
    keys = [[] for j in range(args.step_size)]
    num_batch = len(output[0])
    for i in range(num_batch):
        cp_group = []
        for step, lh in enumerate(output):
            pred = lh >= 0.5
            cp = get_critical_points_patch(lh[i], labels[i][step], pred[i], args)  # (size, size) 
            keys[step].append(cp)
            cp_group.append(cp)
        if valid:
            save_attention_features(cp_group, batch, epoch, args)
    keys = [torch.stack(key, 0).unsqueeze(1) for key in keys]
    keyGroup = torch.cat(keys, dim=1)
    mid = int(args.step_size / 2)
    query = torch.cat([keys[mid] for i in range(args.step_size)], dim=1)
    value = torch.cat([output[mid].unsqueeze(1) for i in range(args.step_size)], dim=1)
    #    key0, key1, key2 = torch.stack(key0, 0), torch.stack(key1, 0), torch.stack(key2, 0)
    #    keyGroup = torch.cat((key0.unsqueeze(1), key1.unsqueeze(1), key2.unsqueeze(1)), dim = 1)
    #    query = torch.cat((key1.unsqueeze(1), key1.unsqueeze(1), key1.unsqueeze(1)), dim = 1)
    #    value = torch.cat((lh1.unsqueeze(1), lh1.unsqueeze(1), lh1.unsqueeze(1)), dim = 1)

    batchSize = query.shape[0]
    channelSize = query.shape[1]
    w, h = query.shape[2], query.shape[3]
    proj_query = query.view(batchSize, channelSize, w * h).permute(0, 2, 1).cpu()  # batch, 156 * 156, 3

    # oneImageAttention(query, key0, key1, key2, value)

    proj_key = keyGroup.view(batchSize, channelSize, w * h).cpu()  # batch, 3, 156 * 156

    energy = torch.bmm(proj_query, proj_key)  # batch, 156*156, 156*156
    # softmax1D = torch.nn.Softmax(dim=-1)
    # similarity = softmax1D(energy)

    proj_value = value.view(batchSize, channelSize, w * h).cpu() # batch, 3, 156 * 156
    score = torch.bmm(proj_value, energy)  # .permute(0, 2, 1))
    attention_down = score.view(batchSize, channelSize, w, h).to(args.device)
    # print(attention_down.shape) #10, 3, 156, 156

    attention_b = attention_down.view(batchSize * channelSize, -1)
    attention_norm = (attention_b - attention_b.min(1, keepdim=True)[0]) / (
                attention_b.max(1, keepdim=True)[0] - attention_b.min(1, keepdim=True)[0] + 0.00000000001)

    attention_norm = attention_norm.view(batchSize, channelSize, w, h)
    if valid:
        for c in range(0, channelSize):
            save_attention_score(attention_norm[:, c], batch, epoch, args, c)
    # attention = (out[0] + out[1] + out[2]) / 3
    attention = torch.mean(attention_norm, dim=1)  # batch, 156, 156

    if args.topo_iteration and not valid:
        attention, iter_attention = temporalAttention(attention, iter_attention, batch, epoch)

    if valid:
        save_attention(attention, batch, epoch, args)

    return attention, True, iter_attention


def topo_attention(output, labels, iter_attention, args, batch=0, epoch=0, valid=False):
    # [batch, 2 * step_size, size, size], [batch, step_size, size, size]
    torch.autograd.set_detect_anomaly(True)
    start = time.time()
    lh_down = []
    for step in range(args.step_size):
        lh_down.append(output[:, step * 2: (step + 1) * 2][:, 1])
    mid = int(args.step_size / 2)
    value = lh_down[mid]

    attention, hasCriticalPoints, iter_attention = get_attention_map(lh_down, labels, iter_attention, args, batch,
                                                                     epoch, valid)

    if hasCriticalPoints:
        result = attention + (1 - value)
    else:
        result = 1 - value

    saveResult = result.clone()
    saveResult[saveResult > 1] = 1
    saveResult[saveResult < 0] = 0

    if valid:
        save_likelihood([value, (1 - saveResult)], batch, epoch, args)

    print('topo-attention running time: ', time.time() - start)

    final = torch.stack((result, (1 - result)), dim=1)
    softmax2D = nn.Softmax2d()
    final = softmax2D(final)
    # img = final[-1,-1].clone()
    # saveForTest(img, 0, 'resultForTopo')
    label = labels[:, mid].long().to(args.device)  # labels_down[:,1].long().to(args.device)

    return final, label, iter_attention


if __name__ == "__main__":
    imgPath = Image.open('results_clstm/CREMI_5step/valid_1/saved_images/epoch_350/0-0lh.png')

    for i, img_as_img in enumerate(ImageSequence.Iterator(imgPath)):
        img_as_np = np.asarray(img_as_img)
    img_as_tensor = torch.from_numpy(img_as_np).float()
    img_as_tensor = (img_as_tensor - torch.min(img_as_tensor)) / (torch.max(img_as_tensor) - torch.min(img_as_tensor))
    print(img_as_tensor.shape)
    predicts = torch.stack([1- img_as_tensor, img_as_tensor], dim =0)
    print(predicts.shape)
    # pred_class = torch.argmax(predicts[:, step * 2: (step + 1) * 2], dim=1)
    # pred_class_all.append(pred_class)
    # acc = accuracy_check(labels[:, step].cpu(), pred_class.cpu())
    # total_acc += acc
    # likelihoodMap = predicts[:, step * 2: (step + 1) * 2][:, 1, :, :]
    # likelihoodMaps.append(likelihoodMap)
    ss
    # img_as_tensor = img_as_tensor[:,:,0]
    img_as_tensor = (img_as_tensor - torch.min(img_as_tensor)) / (torch.max(img_as_tensor) - torch.min(img_as_tensor))
    pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid = getPers(img_as_tensor)

    criticPointMap = torch.zeros(img_as_tensor.shape)
    print(criticPointMap.shape)
    for coor in bcp_lh:
        if any(coor < 0) or any(coor >= criticPointMap.shape[0]) or any(coor >= criticPointMap.shape[1]): continue
        criticPointMap[int(coor[0])][int(coor[1])] = 1

    filter = get_gaussian_kernel()
    img = filter(criticPointMap.unsqueeze(0).unsqueeze(0))
    img = img.squeeze(0).squeeze(0)
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    saveForTest(img, 0, type='gau')

    w, h = img.shape[0], img.shape[1]
    proj_query = img.view(-1, w * h).permute(1, 0).cpu()
    proj_key = img.view(-1, w * h).cpu()

    energy = torch.mm(proj_query, proj_key)
    print('here', energy.shape)
    softmax1D = torch.nn.Softmax(dim=-1)
    attention = softmax1D(energy)
    print(attention.shape)
    # saveForTest(energy, 0, type='energy')
    # saveForTest(attention, 0, type='attention')

    proj_value = img_as_tensor.view(-1, w * h).cpu()
    out = torch.mm(proj_value, attention.permute(1, 0))
    out = out.view(w, h)
    out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
    # saveForTest(out, 0, type='out')
    result = 1 - out + 1 - img_as_tensor
    result[result > 1] = 1
    result[result < 0] = 0
    saveForTest(1 - result, 0, type='result')
    # result = 1- (img + 1 - img_as_tensor)

    # saveForTest(img_as_tensor, 0, type='origin')
    # saveForTest(result, 0, type='result')
