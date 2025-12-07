import torch.nn.functional as F
from .SoftmaxLoss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .TripletLoss import TripletLoss
import ipdb
import torch 

def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, camids):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS) / len(score)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(score)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    return loss
                else:
                    
                    reference_vect_cam = camids[target]
                    mask = reference_vect_cam == target_cam
                    mask = ~mask  # 取反

                    softmax_predictions = F.softmax(score[0], dim=1)
    
                    # 使用负对数似然损失计算交叉熵
                    # cross_entropy_loss = -torch.sum(targets * torch.log(softmax_predictions), dim=1).mean()

                    if isinstance(score, list):
                        try:
#                       这里的的loss没写对！！ 不行就自己手动重新写一下交叉熵的代码。
                            target1 = F.one_hot(target[mask], 1041).float()
                            ID_LOSS1 = -torch.sum(target1 * torch.log(softmax_predictions[mask]), dim=1).mean()
                            target2 = F.one_hot(target[~mask], 1041).float()
                            ID_LOSS2 = -torch.sum(target2 * torch.log(softmax_predictions[~mask]), dim=1).mean()
                            # ID_LOSS3 = F.cross_entropy(score[1], target)
                        except:
                            ipdb.set_trace()
                        ID_LOSS = 0.9 * ID_LOSS1 + 0.1 * ID_LOSS2
                    else:
                        ID_LOSS = F.cross_entropy(score[0][mask], target[mask])

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)/ len(feat)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func