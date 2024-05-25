import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

from src.dataset.cityscapes import Cityscapes
from config import CS_ROOT, cs_coco_roots, params
from src.model_utils import inference
from src.imageaugmentations import Compose, Normalize, ToTensor

from evalPixelLevelSemanticLabeling import main as run_eval

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-train_num", "--TRAIN_NUM", nargs="?", type=str, required=True)
  train_num = vars(parser.parse_args())["TRAIN_NUM"]
  assert train_num is not None

  valset_predictions_dir = os.path.join(CS_ROOT, "valset_predictions")
  valset_predictions_results_dir = os.path.join(CS_ROOT, "valset_predictions_results", train_num)

  os.environ["CITYSCAPES_DATASET"] = CS_ROOT
  
  if not os.path.exists(valset_predictions_dir): 
    print(f"Create directory: {valset_predictions_dir}", flush=True)
    os.makedirs(valset_predictions_dir)
  os.environ["CITYSCAPES_RESULTS"] = valset_predictions_dir
  
  if not os.path.exists(valset_predictions_results_dir): 
    print(f"Create directory: {valset_predictions_results_dir}", flush=True)
    os.makedirs(valset_predictions_results_dir)
  os.environ["CITYSCAPES_EXPORT_DIR"] = valset_predictions_results_dir
 
  roots = cs_coco_roots()
  p = params()
  transform = Compose([ToTensor(), Normalize(Cityscapes.mean, Cityscapes.std)])
  dataset = Cityscapes(CS_ROOT, transform=transform)
  inf = inference(params, roots, dataset, dataset.num_train_ids)
  
  for i in range(len(dataset)):
    probs, gt_train, gt_label, im_path = inf.prob_gt_calc(i)

    ret = torch.argmax(torch.tensor(probs), dim=0).squeeze(0).cpu().numpy()

    new_ret = np.zeros(ret.shape)

    for label in Cityscapes.train_id2label.keys():
        new_ret[ret == label] = Cityscapes.train_id2label[label].id
    
    mask = Image.fromarray(new_ret).convert("L")

    mask.save(os.path.join(valset_predictions_dir, im_path.split(os.sep)[-1]))
    print(f"{i+1}/{len(dataset)}: {os.path.join(valset_predictions_dir, im_path.split(os.sep)[-1])}", flush=True)
  
  sys.argv = [sys.argv[0]] # 1337
  run_eval()
