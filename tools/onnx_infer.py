import sys
sys.path.append('/home/ubuntu/nanodet')

import cv2
import onnxruntime as ort
import os.path as osp
import os
import time
import math
import numpy as np
from tqdm import tqdm
import shutil
import argparse
from nanodet.util import load_config,cfg


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


class onnx_cv2_eval():
    '''
       onnx推理,opencv读图
    '''

    def __init__(self, model_path='',
                img_size=(302,320),
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                keep_ratio=False):

        session = ort.InferenceSession(model_path,providers=['CUDAExecutionProvider']) 
        #      ## provider有如下几种,选择其一即可：'CPUExecutionProvider'，'CUDAExecutionProvider','TensorrtExecutionProvider'
        inp_name = session.get_inputs()[0].name
        oup_name = session.get_outputs()[0].name
        self.session = session
        self.inp_name = inp_name
        self.oup_name = oup_name
        self.img_size=img_size
        self.mean=mean
        self.std=std
        self.keep_ratio=keep_ratio

    def run(self, image, cfg, need_preprocess=False, need_nms=False):
        
        # 如果是未经处理的图片， 则需进行预处理
        if need_preprocess:
            
            image = cv2.resize(image, self.img_size)
            # resize to mobile_net tensor size
            image = self.preprocess(image,self.mean,self.std)
            # preprocess it
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)
            image = np.expand_dims(image, 0)
        # 确定推理样本id， 主要是为了减少计算量， 加快训练速度

        ret = self.session.run([self.oup_name], {self.inp_name: image})[0]
        
        if need_nms:
            ret = self.postprocess(ret,cfg)
        
        # print(ret,'\n',ret.shape)
        # ret = ret.argmax(axis=1)[0]

        return ret
    
    def preprocess(self,img,mean,std):
        mean = np.array(mean, dtype=np.float64).reshape(1, -1)
        stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        return img
    
    
    def postprocess(self,preds,cfg):

        class_names = cfg.class_names
        reg_max = cfg.model.arch.head.reg_max

        cls_scores,bbox_preds = np.split(preds,[len(class_names)],-1)

        input_shape = self.img_size
        input_height, input_width = input_shape
        strides = cfg.model.arch.head.strides

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in strides
        ]
        # get grid cells of one image, set onnx_batch = 1
        onnx_batch = 1
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                onnx_batch,
                featmap_sizes[i],
                stride,
            )
            for i, stride in enumerate(strides)
        ]
        

        center_priors = np.concatenate(mlvl_center_priors, 1)

        dis_preds = self.distribution_project(reg_max, bbox_preds) * center_priors[..., 2, None]
        bboxes = self.distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_scores #self.sigmoid(cls_scores)
        result_list = []
        for i in range(onnx_batch):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
    
            padding = np.zeros([score.shape[0], 1])
            score = np.concatenate([score, padding], 1)
       

            results = self.multiclass_nms(
                score,
                bbox,
                score_thr=0.1,
                iou_threshold=0.6,
                max_num=100,
                class_names=class_names,
            )
            result_list.append(results)
        return result_list

    
    def get_single_level_center_priors(self, batch_size, featmap_size, stride,):
        h, w = featmap_size
        x_range = (np.arange(w, dtype=np.float32)) * stride
        y_range = (np.arange(h, dtype=np.float32)) * stride
        y, x = np.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = np.zeros_like(x) + stride
        proiors = np.stack([x, y, strides, strides], -1)
        proiors = np.expand_dims(proiors, 0)
        return proiors.repeat(batch_size, 0) 
    
    def distribution_project(self,reg_max,x):
        project = np.linspace(0, reg_max, reg_max + 1, dtype=np.float32)
        project = np.expand_dims(project,1)
  
        shape = x.shape
        x = self.softmax(x.reshape(*shape[:-1], 4, reg_max + 1), dim=-1)
        x = np.dot(x, project).reshape(*shape[:-1], 4)
   
        return x

    def distance2bbox(self,points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            np: Decoded bboxes.
        """
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            x1 = x1.clip(min=0, max=max_shape[1])
            y1 = y1.clip(min=0, max=max_shape[0])
            x2 = x2.clip(min=0, max=max_shape[1])
            y2 = y2.clip(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], -1)

    def multiclass_nms(self,scores,bboxes, score_thr=0.5, iou_threshold=0.6, max_num=100,class_names=[]):
        results = {}
        cls_id = np.argmax(scores,-1).astype(dtype='int64')
        for i in range(cls_id.shape[0]):
            if results.get(cls_id[i]):
                results[cls_id[i]].append(list(bboxes[i].astype(dtype='int64')) + [scores[i][cls_id[i]]])
            else:
                results[cls_id[i]]= [list(bboxes[i].astype(dtype='int64')) + [scores[i][cls_id[i]]]]
        for k,v in results.items():
            v_ = self.oneclass_nms(v)
            results[k] = v_
        return results
            
    
    def oneclass_nms(self,score_bbox, score_thr=0.5, iou_threshold=0.6):

        score_bbox.sort(key = lambda usr:usr[-1])
        res = [score_bbox.pop()]
        while score_bbox!=[]:
            x1,y1,x2,y2,_ = res[-1]
            tmp = []
            n = len(score_bbox)-1
            while n>=0:
                x3,y3,x4,y4,_=score_bbox[n]
                inter =   max(0,min(x2,x4) - max(x1,x3)) * max(0, min(y2,y4)-max(y1,y3))
                union =   max(0,x2-x1)*max(0,y2-y1) + max(0,x3-x3)*max(0,y4-y3) - inter
                iou = inter / (union+1.0e-8)
                if iou < iou_threshold:
                    tmp.append(score_bbox[n])
                n-=1
            score_bbox = tmp
            if score_bbox!=[]:
                res.append(score_bbox.pop())
        return res


    def softmax(self,x,dim):
        x -= np.max(x, axis= dim, keepdims=True)
        f_x = np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)
        return f_x

    def sigmoid(self,x):
        x = 1/(1+np.exp(-x))
        return x




def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
     
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description='onnx model inference for nanodet')
    parser.add_argument("--config", help='The config file path')
    parser.add_argument("--model_path", help="onnx model path")
    args = parser.parse_args()
    return args


def main(args):

    load_config(cfg, args.config)

    #onnx_infer = onnx_cv2_eval(model_path=args.model_path)

    img_path   = cfg.data.val.img_path
    input_size = cfg.data.val.input_size
    norm_mean  = cfg.data.val.pipeline.normalize[0] 
    norm_std   = cfg.data.val.pipeline.normalize[0] 
    keep_ratio = cfg.data.val.keep_ratio


    try:
        onnx_infer = onnx_cv2_eval(model_path=args.model_path,
                                img_size=input_size,
                                mean=norm_mean,
                                std=norm_std,
                                keep_ratio=keep_ratio)
    except FileNotFoundError:
        raise ('onnx file is not found')
    
    imgs = os.listdir(img_path)
    save_dir = img_path+'_det_result'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    total_time = 0
    for img_name in tqdm(imgs):

        if cv2.imread(osp.join(img_path,img_name)) is not None:
            raw_img = cv2.imread(osp.join(img_path,img_name))
            raw_img = cv2.resize(raw_img, input_size)
        
        img = cv2.imread(osp.join(img_path,img_name))
        img = img[...,::-1]
        ret = onnx_infer.run(img,cfg,need_preprocess=True)

        det_img = overlay_bbox_cv(raw_img,ret[0],cfg.class_names,0.55)
        cv2.imwrite(os.path.join(save_dir,img_name),det_img)
        

    """
    input_size: [320,320] #[w,h]
    keep_ratio: False
    pipeline:
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
    """



if __name__ == '__main__':
    args = parse_args()
    main(args)
    
