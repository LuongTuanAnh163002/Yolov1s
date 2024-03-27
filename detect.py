import cv2 as cv
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import yaml

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from models.yolo import model_tiny_yolov1
from utils.general import increment_path, xywh2minmax, colorstr
from utils.datasets import LoadImages
from utils.metric import iou

class Tiny_Yolov1(object):
  def __init__(self, weights_path, class_name):
      self.weights_path = weights_path
      self.classes_name = class_name

  def predict(self, image):
      input_shape = (1, 448, 448, 3)
      inputs = Input(input_shape[1:4])
      outputs = model_tiny_yolov1(inputs)
      model = Model(inputs=inputs, outputs=outputs)
      model.load_weights(self.weights_path, by_name=True)
      y = model.predict(image, batch_size=1)

      return y

def yolo_head_v1(feats):
  # Dynamic implementation of conv dims for fully convolutional model.
  conv_dims = np.shape(feats)[0:2]  # assuming channels last
  # print(conv_dims)
  # In YOLO the height index is the inner most iteration.
  conv_height_index = np.arange(0, stop=conv_dims[0])
  # print(conv_height_index)
  conv_width_index = np.arange(0, stop=conv_dims[1])
  # print(conv_width_index)
  conv_height_index = np.tile(conv_height_index, [conv_dims[1]])
  # print(conv_height_index)

  # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
  conv_width_index = np.tile(np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
  # print(conv_width_index)
  conv_width_index = np.reshape(np.transpose(conv_width_index), [conv_dims[0] * conv_dims[1]])
  # print(conv_width_index)
  conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
  conv_index = np.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])

  conv_dims = np.reshape(conv_dims, [1, 1, 1, 2])
  # print(conv_dims)
  box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
  box_wh = feats[..., 2:4] * 448

  return box_xy, box_wh

def detect(opt):
  source, weight = opt.source, opt.weight
  img_size, save_dir = opt.img_size, opt.save_dir
  conf_thresh = opt.conf_thres
  Path(save_dir).mkdir(parents=True, exist_ok=True)

  model_weight = weight + "/best.hdf5"
  model_cfg_path = weight + "/model_cfg.yaml"
  with open(model_cfg_path) as f:
      model_cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
  class_name = model_cfg_dict["class_name"]
  n_class = model_cfg_dict['n_class']
  model = Tiny_Yolov1(model_weight, class_name)

  vid_path, vid_writer = None, None
  dataset = LoadImages(source, img_size=img_size)
  t0 = time.time()
  for path, im, im0s, vid_cap in dataset:
    name_img = Path(path).name
    input_shape = (1, 448, 448, 3)
    image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    image = cv.resize(image, input_shape[1:3])
    image = np.reshape(image, input_shape)
    image = image / 255.

    prediction = model.predict(image)
    predict_class = prediction[..., :20]  # 1 * 7 * 7 * 20
    predict_trust = prediction[..., 20:22]  # 1 * 7 * 7 * 2
    predict_box = prediction[..., 22:]  # 1 * 7 * 7 * 8

    predict_class = np.reshape(predict_class, [7, 7, 1, 20])
    predict_trust = np.reshape(predict_trust, [7, 7, 2, 1])
    predict_box = np.reshape(predict_box, [7, 7, 2, 4])

    predict_scores = predict_class * predict_trust  # 7 * 7 * 2 * 20

    box_classes = np.argmax(predict_scores, axis=-1)  # 7 * 7 * 2
    box_class_scores = np.max(predict_scores, axis=-1)  # 7 * 7 * 2
    best_box_class_scores = np.max(box_class_scores, axis=-1, keepdims=True)  # 7 * 7 * 1

    box_mask = box_class_scores >= best_box_class_scores  # ? * 7 * 7 * 2

    filter_mask = box_class_scores >= conf_thresh  # 7 * 7 * 2
    filter_mask *= box_mask  # 7 * 7 * 2

    filter_mask = np.expand_dims(filter_mask, axis=-1)  # 7 * 7 * 2 * 1

    predict_scores *= filter_mask  # 7 * 7 * 2 * 20
    predict_box *= filter_mask  # 7 * 7 * 2 * 4

    box_classes = np.expand_dims(box_classes, axis=-1)
    box_classes *= filter_mask  # 7 * 7 * 2 * 1

    box_xy, box_wh = yolo_head_v1(predict_box)  # 7 * 7 * 2 * 2
    box_xy_min, box_xy_max = xywh2minmax(box_xy, box_wh)  # 7 * 7 * 2 * 2, 7 * 7 * 2 * 2

    predict_trust *= filter_mask  # 7 * 7 * 2 * 1
    nms_mask = np.zeros_like(filter_mask)  # 7 * 7 * 2 * 1
    predict_trust_max = np.max(predict_trust)
    max_i = max_j = max_k = 0
    while predict_trust_max > 0:
      for i in range(nms_mask.shape[0]):
          for j in range(nms_mask.shape[1]):
              for k in range(nms_mask.shape[2]):
                  if predict_trust[i, j, k, 0] == predict_trust_max:
                      nms_mask[i, j, k, 0] = 1
                      filter_mask[i, j, k, 0] = 0
                      max_i = i
                      max_j = j
                      max_k = k
      for i in range(nms_mask.shape[0]):
          for j in range(nms_mask.shape[1]):
              for k in range(nms_mask.shape[2]):
                  if filter_mask[i, j, k, 0] == 1:
                      iou_score = iou(box_xy_min[max_i, max_j, max_k, :],
                                      box_xy_max[max_i, max_j, max_k, :],
                                      box_xy_min[i, j, k, :],
                                      box_xy_max[i, j, k, :])
                      if iou_score > 0.1:
                          filter_mask[i, j, k, 0] = 0
      predict_trust *= filter_mask  # 7 * 7 * 2 * 1
      predict_trust_max = np.max(predict_trust)

    box_xy_min *= nms_mask
    box_xy_max *= nms_mask

    origin_shape = im0s.shape[0:2]
    im0s = cv.resize(im0s, (img_size, img_size))
    detect_shape = filter_mask.shape

    for i in range(detect_shape[0]):
      for j in range(detect_shape[1]):
          for k in range(detect_shape[2]):
              if nms_mask[i, j, k, 0]:
                  cv.rectangle(im0s, (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                (int(box_xy_max[i, j, k, 0]), int(box_xy_max[i, j, k, 1])),
                                (255, 0, 0))
                  cv.putText(im0s, class_name[box_classes[i, j, k, 0]],
                              (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                              1, 1, (255, 0, 0))
    im0s = cv.resize(im0s, (origin_shape[1], origin_shape[0]))
    cv2.imwrite(save_dir + "/" + name_img, im0s)

  print(colorstr("green", f'Done. ({time.time() - t0:.3f}s)'))
  print("All result save in: ", save_dir)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, default='image, video', help='data.yaml path')
  parser.add_argument('--weight', type=str, default='weight of yolov1', help='data.yaml path')
  parser.add_argument('--conf_thres', type=float, default=0.1, help='confident threshold')
  parser.add_argument('--img_size', type=int, default=448, help='[train, test] image sizes')
  parser.add_argument('--project', default='runs/detect', help='save to project/name')
  parser.add_argument('--name', default='exp', help='save to project/name')
  opt = parser.parse_args()
  opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
  detect(opt)


  

