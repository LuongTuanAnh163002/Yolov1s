from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf

import time
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
import os
import numpy as np
import warnings
import argparse
import yaml
import platform

from utils.general import increment_path, colorstr, date_modified
from models.yolo import model_tiny_yolov1
from utils.datasets import SequenceData
from utils.loss import yolo_loss

warnings.filterwarnings("ignore") #remove warning

def train(opt):
  s = f'Yolov1 🚀 {date_modified()} tensorflow {tf.__version__} '
  print(colorstr("red", s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s))
  data, epochs, batch_size, save_dir = opt.data, opt.epochs, opt.batch_size, opt.save_dir
  input_shape = opt.img_size

  save_dir = increment_path(save_dir, exist_ok=False)
  Path(save_dir).mkdir(parents=True, exist_ok=True)
  with open(opt.data) as f:
      data_dict = yaml.load(f, Loader=yaml.SafeLoader)
  
  datasets_path = data_dict["datasets_path"]
  nc = data_dict["nc"]
  class_name = data_dict['names']

  with open(save_dir + '/model_cfg.yaml', 'w') as file:
    yaml.dump({'n_class': nc, 'class_name': class_name}, file)
  #-------------------Load model----------------------------
  print(colorstr("Loading model..."))
  inputs = Input(input_shape)
  yolo_outputs = model_tiny_yolov1(inputs)
  model = Model(inputs=inputs, outputs=yolo_outputs)
  print(colorstr("Model structure:"))
  print(model.summary())
  model.compile(loss=yolo_loss, optimizer='adam')
  weights_path = os.path.join(save_dir, 'best.hdf5')
  checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                save_weights_only=True, save_best_only=True)
  early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
  print(colorstr("green", "Done load model..."))
  
  #-------------------Load model----------------------------

  print(colorstr('Loading dataset from: ') + str(datasets_path))
  #-----------------Load dataset---------------------------
  train_generator = SequenceData(
          'train', datasets_path, input_shape, batch_size)
  validation_generator = SequenceData(
      'val', datasets_path, input_shape, batch_size)
  
  print(colorstr("green", 'Done load dataset'))
  #-----------------Load dataset---------------------------

  #------------------Training------------------------------
  print(f"Start training for {epochs} epochs")
  t0 = time.time()
  tensorboard_callback = TensorBoard(log_dir= save_dir + "/logs")
  history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        # use_multiprocessing=True,
        workers=4,
        callbacks=[checkpoint, early_stopping, tensorboard_callback]
    )
  print(colorstr("green", f'Done. ({time.time() - t0:.3f}s)'))
  #------------------End training------------------------------

  #------------------Save result-------------------------------
  val_loss_history =  history.history['val_loss']
  loss_history = history.history['loss']
  plt.plot(np.arange(len(loss_history)), loss_history, label="train_loss")
  plt.plot(np.arange(len(val_loss_history)), val_loss_history, label="val_loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.grid()
  plt.legend(loc="upper right")
  plt.savefig(os.path.join(save_dir, "result.png"))
  #------------------Save result-------------------------------

  #------------------Save last weight--------------------------
  last_point = os.path.join(save_dir, 'last.hdf5')
  model.save_weights(last_point)
  #------------------Save last weight--------------------------
  dict_opt = {'data': opt.data, 'epochs': opt.epochs, 'weight_ckpt': opt.weight_ckpt,
              'batch_size': opt.batch_size, 'img_size': opt.img_size, 'resume': opt.resume,
              'workers': opt.workers, 'project': opt.project, 'name': opt.name}
  
  with open(save_dir + '/opt.yaml', 'w') as file:
    yaml.dump(dict_opt, file)
  text_script_run = "python train.py"
  for key in dict_opt.keys():
    if key == "weight_ckpt" and dict_opt[key] != '':
      text_script_run += f" --weight_ckpt {dict_opt[key]}"
    
    elif key in ["data", "epochs", "batch_size"]:
      text_script_run += f" --{key} {dict_opt[key]}"
    
    elif key == "resume" and dict_opt[key] != False:
       text_script_run += f" --resume {dict_opt[key]}"
    
    elif key == "img_size" and dict_opt[key] != [448, 448, 3]:
       text_script_run += f" --img_size {dict_opt[key]}"
    
    elif key == "workers" and dict_opt[key] != 2:
       text_script_run += f" --workers {dict_opt[key]}"

    elif key == "project" and dict_opt[key] != 'runs/train':
       text_script_run += f" --project {dict_opt[key]}"
    
    elif key == "name" and dict_opt[key] != 'exp':
       text_script_run += f" --name {dict_opt[key]}"
  
  with open(save_dir + '/script.txt', 'w') as file:
    file.write(text_script_run)
      
  print("All result save in:", save_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='data/1class_flag.yaml', help='data.yaml path')
  parser.add_argument('--epochs', type=int, default=300)
  parser.add_argument('--weight_ckpt', type=str, default='', help='weight checkpoint to resume')
  parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
  parser.add_argument('--img_size', nargs='+', type=int, default=[448, 448, 3], help='[train, test] image sizes')
  parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
  parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
  parser.add_argument('--project', default='runs/train', help='save to project/name')
  parser.add_argument('--name', default='exp', help='save to project/name')
  opt = parser.parse_args()
  opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
  print(colorstr("red", f"Start with 'tensorboard --logdir {opt.save_dir}, view at http://localhost:6006/"))
  print(colorstr(opt))
  train(opt)


