import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule, RepDepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean
import random

class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule =   RepDepthwiseConvModule if conv_type == "RepDWConv" else DepthwiseConvModule
        #self.ConvModule2 = DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1),
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                ) 
            )
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def forward(self, feats):
        #print('==============',sum([torch.numel(i) for i in self.backbone.parameters()] + [torch.numel(i) for i in self.fpn.parameters()] + [torch.numel(i) for i in self.head.parameters()])/10**6)
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        device = preds.device
        batch_size = preds.shape[0]
        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        # todo 损失计算

        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res,0
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res,0
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign,f):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )
        #print(decoded_bboxes.shape, '%%%%%%%%%%%%%%%')




        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)
        #print(pos_inds.shape,'￥$￥$￥$￥$￥$',decoded_bboxes[pos_inds,:].shape)

        bs = 84
        if f and cls_preds.shape[0] % bs ==0:
            IOU = torch.zeros_like(label_scores,device='cuda')
            Iou = iou(decoded_bboxes[pos_inds,:], bbox_targets[pos_inds,:])
            #print(torch.max(Iou), torch.max(label_scores[pos_inds]), '%%%%%%%%%%%%%%%')
            n,_ = cls_preds .shape

            Cls_preds        = cls_preds.reshape(bs,n//bs,-1)
            Labels                 =  labels.reshape(bs,n//bs)
            Label_scores  =  label_scores.reshape(bs,n//bs)
            IOU[pos_inds] = Iou
            IOU  = IOU.reshape(bs,n//bs)

            
            #print(cls_preds.shape,labels.shape,label_scores.shape,IOU.shape,pos_inds.shape)
            loss_extra = 0
            for i in range(0,bs,4):
                Pos_inds = torch.nonzero(   (Labels[i,:] >= 0) & (Labels[i,:] < self.num_classes), as_tuple=False).squeeze(1)
                #print(Pos_inds.shape,pos_inds.shape)
                loss_extra += extra_loss(Cls_preds[i,:,:], (Labels[i,:], Label_scores[i,:],IOU[i,:]), Pos_inds, bs)
            loss_extra/=bs
        else:
            loss_extra = 0 

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        #print(loss,'￥$￥$￥$￥$￥$￥$￥$￥')
        if f :
            loss+=loss_extra
            loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl,loss_extra=loss_extra)
        else:
            loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)
        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        # No target
        if num_gts == 0:
            return labels, label_scores, bbox_targets, dist_targets, 0

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
        return (
            labels,
            label_scores,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)
import random

def extra_loss(pred, target,labelBool,batchSize):
    pred = pred.sigmoid()

    label, score ,iou= target
    
    n,c = pred.shape

    
    pred = torch.cat([pred,torch.zeros([n,1],device='cuda')],dim=1)
    allC = []
    #labelBool = label!=c
    # labelBool0 = label==c
    label0 = label[labelBool]
    num = label0.shape[0]
    #print(num)
    Loss = torch.tensor(0.0,device='cuda') #torch.zeros(n).cuda()

    #pred = pred[:,label]
    #print(pred.shape,score.shape,'￥$￥$￥$￥$￥$￥')

    #Loss[labelBool] = (score[labelBool] - 0.9)*torch.log( torch.max(1+pred[labelBool]-score[labelBool] ,torch.tensor(0.001,device='cuda')))

    if np.random.rand()>0.7:

        low = 0
        high = num//3
    elif np.random.rand()>0.3:
        low = num//3
        high = num//3*2
    else:
        low = num //3*2
        high = num


    for i in range(num):
        if  label0[i] not in allC:
            allC.append(label0[i])
            idx = label == label0[i]
            scorc = score[idx]
            iouc = iou[idx]
            predc = pred[:,label0[i]]
            labelc = label[idx]
            predc = predc[idx]
            Max,index = torch.max(predc,dim=0)
            #scorc = scorc - 0.9#scorc[index]
            # loss = ((scorc-predc)**2)*(iouc-0.95)*torch.log( torch.min(1+predc-scorc,torch.tensor(1,device='cuda')))
            #loss =  -(iouc - iouc[index]) * torch.log( torch.min(1+predc-scorc , torch.tensor(1,device='cuda')))
            loss =  (iouc-0.9)*torch.log( torch.min(1+predc-scorc,torch.tensor(1,device='cuda')))
            # 0.95
            # loss = loss[loss>0]
            if iouc.size(0) > 0:
                Loss += torch.mean(loss)  #Loss[idx] = loss#todo 每一类loss直接mean，然后按照类别相加
    #bol = Loss >0
    #print(torch.max(iou[labelBool]), Loss[bol])

    #print(torch.max(Loss))
    #Loss[bol] = 0
    #Loss[labelBool0] = 0
    #print(0.1*Loss[Loss!=0])
    #print(allC)
    #todo 开始的3个epoch用整个num计算损失，然后才low ,high
    return 1*Loss/max(len(allC),1) #torch.tensor(0.0,device='cuda')  if Loss[bol].size(0)==0 else  0.5* torch.mean(Loss[bol])




def iou(pred,gt):
    zero = torch.tensor(0,device='cuda')
    aog = torch.abs(gt[:, 2] - gt[:, 0] +
                    1) * torch.abs(gt[:, 3] - gt[:, 1] + 1)
    aop = torch.abs(pred[:,2] - pred[:, 0] +
                    1) * torch.abs(pred[:, 3] - pred[:, 1] + 1)

    iw = torch.min(pred[:,2], gt[:,2]) - torch.max(
        pred[:, 0], gt[:,0]) + 1
    ih = torch.min(pred[:,3], gt[:,3]) - torch.max(
        pred[:, 1], gt[:,1]) + 1
    inter = torch.max(iw,zero) * torch.max(ih,zero)

    union = aog + aop - inter
    # print(union.shape,inter.shape,self.t_zero)
    iou = torch.max(inter / union,zero)
    return iou