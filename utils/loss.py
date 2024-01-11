# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
    
        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  #m.nl为3, balance仍未[4.0, 1.0, 0.4]
        self.ssi = list(m.stride).index(16) if autobalance else 0  # det.stride是[ 8., 16., 32.]， self.ssi表示stride为16的索引，当autobalance为true时，self.ssi为1.
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors  #[3,3,2]
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        #p 是每个预测头输出的结果
        #    [p[0].shape： torch.Size([16, 3, 80, 80, 85])  [batchsize,anchor box数量,特征图大小,特征图大小,80+4+1]
        #     p[1].shape： torch.Size([16, 3, 40, 40, 85])
        #     p[2].shape： torch.Size([16, 3, 20, 20, 85])
        #    ]
 
        # targets: gt box信息，维度是(n, 6)，其中n是整个batch的图片里gt box的数量，以下都以gt box数量为190来举例。
        # 6的每一个维度为(图片在batch中的索引， 目标类别， x, y, w, h)
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # 四个参数什么意思见最下面
        #indices里的是坐标，tbox里的是偏移量 
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image index,anchor index，预测该gt box的网格y坐标，预测该gt box的网格x坐标。
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            #tobj shape: torch.Size([16, 3, 80, 80])
            #pi shape： torch.Size([16, 3, 80, 80, 85])
            n = b.shape[0]  # number of targets 下文中的712
            if n:
                #pi[b, a, gj, gi] shape 712,85
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                #pxy=pi[:, :2]
                #pwh=pi[:, 2:4]
                #_=pi[:, 4:5] 置信度
                #pcls = 剩下80个分类

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5 #将预测的中心点坐标变换到-0.5到1.5之间
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i] #表示把预测框的宽和高限制在4倍的anchors内
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #计算808个预测框与gt box的iou loss
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype) #detach函数使得iou不可反向传播， clamp将小于0的iou裁剪为0
                if self.sort_obj_iou:
                    j = iou.argsort() #返回的是排序后的score_iou中的元素在原始score_iou中的位置。
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio
                #tobj:(16, 3, 80, 80,808),
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    #pcls shape:(808,80)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp   #构造独热码
                    lcls += self.BCEcls(pcls, t)  # BCE
                    #self.cn和self.cp分别是标签平滑的负样本平滑标签和正样本平滑标签

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # 该函数主要是处理gt box，先介绍一下gt box的整体处理策略：
        # 1、将gt box复制3份，原因是有三种长宽的anchor， 每种anchor都有gt box与其对应，也就是在筛选之前，一个gt box有三种anchor与其对应。
        # 2、过滤掉gt box的w和h与anchor的w和h的比值大于设置的超参数anchor_t的gt box。
        # 3、剩余的gt box，每个gt box使用至少三个方格来预测，一个是gt box中心点所在方格，另两个是中心点离的最近的两个方格，如下图
        na, nt = self.na, targets.shape[0]  # number of anchors一般是3, nt是这个bs中所有的gt box数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # 7个数，前6个数对应targets的第二维度6 normalized to gridspace gain
        #ai:anchor的索引，shape为(3, gt box的数量)， 3行里，第一行全是0， 第2行全是1， 第三行全是2，表示每个gt box都对应到3个anchor上。
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # 加上anchor的索引，把target重复三边 (3,nt,7),targets[0][0]和target[1][0]都属
                                                                           #长度为7的张量，他们的差别只有最后一位 anchor的索引不同
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets 乘了个g，所以都是0.5

        for i in range(self.nl):    #对每个检测头进行遍历
            anchors, shape = self.anchors[i], p[i].shape     #p[i]:[16,3,80,80,85] anchors:[3,3,2]
            #shape  [batchsize,anchor box数量,特征图大小,特征图大小,80+4+1]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy 重复shape的第三二位

            # Match targets to anchors
            #2：6代表target里的xywh,因为是归一化后的,所以需要乘上xyxy来恢复原先的尺度
            t = targets * gain  # target shape(3,n,7)
            if nt:  #nt是gt box的数量
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # shape为[3,nt,2] 2是gt box的w和h与anchor的w和h的比值 anchors[:, None] 的形状为3,1,2
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  #当gt box的w和h与anchor的w和h的比值比设置的超参数anchor_t大时，则此gt box去除
                # j的形状是(3,nt),里面的值均为true或false
                t = t[j]  # 过滤掉不合适的gtbox

                # Offsets
                # t的每一个维度为(图片在batch中的索引， 目标类别， x, y, w, h,anchor的索引)
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # 将以图像左上角为原点的坐标变换为以图像右下角为原点的坐标 gain[2,3]是特征图的xy长度
                #g是0.5
                #下面是寻找另外两个负责该gt的gird
                # 以图像左上角为原点的坐标，取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false。
                # j和k的shape都是(239)，true的位置分别表示靠近方格左边的gt box和靠近方格上方的gt box。 
                j, k = ((gxy % 1 < g) & (gxy > 1)).T 
                #以图像右下角为原点的坐标，取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false。
                #l和m的shape都是(239)，true的位置分别表示靠近方格右边的gt box和靠近方格下方的gt box。
                l, m = ((gxi % 1 < g) & (gxi > 1)).T   #大于1是为了防止超出边界
                j = torch.stack((torch.ones_like(j), j, k, l, m)) #j的shape为(5, 239)
                t = t.repeat((5, 1, 1))[j] #将t复制五遍，用j过滤
                #之前的shape为(239, 7)， 这里将t复制5个，大小变成了[5, 239, 7]然后使用j来过滤，
                # 假设过滤后的t shape为(712,7)
                #第一个t是保留所有的gt box，因为上一步里面增加了一个全为true的维度，
                #第二个t保留了靠近方格左边的gt box，
                #第三个t保留了靠近方格上方的gt box，
                #第四个t保留了靠近方格右边的gt box，
                #第五个t保留了靠近方格下边的gt box，
               
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # 第一个t保留所有的gt box偏移量为[0, 0], 即不做偏移
                # 第二个t保留的靠近方格左边的gt box，偏移为[0.5, 0]，即向左偏移0.5(后面代码是用gxy - offsets，所以正0.5表示向左偏移)，则偏移到左边方格，表示用左边的方格来预测
                # 第三个t保留的靠近方格上方的gt box，偏移为[0, 0.5]，即向上偏移0.5，则偏移到上边方格，表示用上边的方格来预测
                # 第四个t保留的靠近方格右边的gt box，偏移为[-0.5, 0]，即向右偏移0.5，则偏移到右边方格，表示用右边的方格来预测
                # 第五个t保留的靠近方格下边的gt box，偏移为[0, 0.5]，即向下偏移0.5，则偏移到下边方格，表示用下边的方格来预测
                 #offsets的shape为(712, 2), 表示保留下来的712个gt box的x, y对应的偏移，
                # 一个gt box的中心点x坐标要么是靠近方格左边，要么是靠近方格右边，y坐标要么是靠近方格上边，要么是靠近方格下边，
                # 所以一个gt box在以上五个t里面，会有三个t是true。
                # 也即一个gt box有三个方格来预测，一个是中心点所在方格，另两个是离的最近的两个方格。
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors:(712,1),表示anchor的索引, image, class
            gij = (gxy - offsets).long()  #转换成了整形，是预测该gtbox的网格坐标，这样下面gxy - gij得到的不是offsets，而是相对该网格的偏移量
            gi, gj = gij.T  # grid indices

            # Append 一共三次循环，一次循环append一个最高维度
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # indices的shape为(3, ([712], [712], [712], [712])), 
            # 4个808分别表示每个gt box(包括偏移后的gt box)在batch中的image index， anchor index， 预测该gt box的网格y坐标， 预测该gt box的网格x坐标。
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # 假如tbox的shape为(3, ([712, 4]))， 表示3个检测头对应的gt box的xywh， 其中x和y已经减去了预测方格的整数坐标，
            # 比如原始的gt box的中心坐标是(51.7, 44.8)，则该gt box由方格(51, 44)，以及离中心点最近的两个方格(51, 45)和(52, 44)来预测(见build_targets函数里的解析),
            # 换句话说这三个方格预测的gt box是同一个，其中心点是(51.7, 44.8)，但tbox保存这三个方格预测的gt box的xy时，保存的是针对这三个方格的偏移量,
            # 分别是：
            #     (51.7 - 51 = 0.7, 44.8 - 44 = 0.8)
            #     (51.7 - 51 = 0.7, 44.8 - 45 = -0.2)
            #     (51.7 - 52 = -0.3, 44.8 - 44 = 0.8)
            anch.append(anchors[a])  # shape为(3, ([712, 2]))， 表示每个检测头对应的712个gt box所对应的anchor。
            tcls.append(c)  # shape为(3, 712), 表示3个检测头对应的gt box的类别。 

        return tcls, tbox, indices, anch
