import os
import sys
import time
import numpy as np
import json
import pickle
sys.path.append(os.path.dirname(sys.path[0]))

from PIL import Image
from config import cs_coco_roots
from src.dataset.coco import COCO
from pycocotools.coco import COCO as coco_tools


def main():
    start = time.time()
    root = cs_coco_roots.coco_root
    split = "train"
    year = 2017
    id_in = COCO.train_id_in
    id_out = COCO.train_id_out
    annotation_file = '{}/annotations/instances_{}.json'.format(root, split+str(year))
    images_dir = '{}/{}'.format(root, split+str(year))
    tools = coco_tools(annotation_file)
    save_dir = '{}/annotations/ood_seg_{}'.format(root, split+str(year))
    print("\nPrepare COCO{} {} split for OoD training".format(str(year), split))

    # Names of classes that are excluded - these are Cityscapes classes also available in COCO
    exclude_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']

    # Fetch all image ids that does not include instance from classes defined in "exclude_classes"
    exclude_cat_Ids = tools.getCatIds(catNms=exclude_classes)
    exclude_img_Ids = []
    for cat_Id in exclude_cat_Ids:
        exclude_img_Ids += tools.getImgIds(catIds=cat_Id)
    exclude_img_Ids = set(exclude_img_Ids)
    img_Ids = [int(image[:-4]) for image in os.listdir(images_dir) if int(image[:-4]) not in exclude_img_Ids]

    num_masks = 0
    # Process each image
    print("Ground truth segmentation mask will be saved in:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created save directory:", save_dir)
     
    object_stats = {}
    pixel_stats = {}
    modified_stats = {"less_than_0.2" : {"object_stats" : {}, "pixel_stats" : {}}, 
                      "greater_than_0.8" : {"object_stats" : {}, "pixel_stats" : {}}}

    with open('coco_categories.json', 'r') as f:
      cats = json.load(f)

    for i, img_Id in enumerate(img_Ids):
        img = tools.loadImgs(img_Id)[0]
        h, w = img['height'], img['width']

        ann_Ids = tools.getAnnIds(imgIds=img['id'], iscrowd=None)
        annotations = tools.loadAnns(ann_Ids)
        
        # Generate binary segmentation mask
        mask = np.ones((h, w), dtype="uint8") * id_in
        for j in range(len(annotations)):
          mask = np.maximum(tools.annToMask(annotations[j])*id_out, mask)
          
          category_name = tools.loadCats(annotations[j]["category_id"])[0]["name"]
          number_of_pixels = np.sum(tools.annToMask(annotations[j]) == 1)
          
          for cat in cats:
            if cat["name"] == category_name:
              category_name = cat["supercategory"]

          if category_name not in object_stats:
            object_stats[category_name] = 1
            pixel_stats[category_name] = number_of_pixels
          else:
            object_stats[category_name] += 1
            pixel_stats[category_name] += number_of_pixels
        
        # Save segmentation mask
        Image.fromarray(mask).save(os.path.join(save_dir, "{:012d}.png".format(img_Id)))
        
        ood_perc = (np.sum(np.array(mask) == id_out)) / (mask.shape[0]*mask.shape[1])
        if ood_perc <= 0.2:
          path = os.path.join(root, "annotations", "less_than_0.2")
          if not os.path.exists(path):
              os.makedirs(path)
          Image.fromarray(mask).save(os.path.join(path, "{:012d}.png".format(img_Id)))
          
          if category_name not in modified_stats["less_than_0.2"]["object_stats"]:
            modified_stats["less_than_0.2"]["object_stats"][category_name] = 1
            modified_stats["less_than_0.2"]["pixel_stats"][category_name] = number_of_pixels
          else:
            modified_stats["less_than_0.2"]["object_stats"][category_name] += 1
            modified_stats["less_than_0.2"]["pixel_stats"][category_name] += number_of_pixels

        elif ood_perc >= 0.8:
          path = os.path.join(root, "annotations", "greater_than_0.8")
          if not os.path.exists(path):
              os.makedirs(path)
          Image.fromarray(mask).save(os.path.join(path, "{:012d}.png".format(img_Id)))
          
          if category_name not in modified_stats["greater_than_0.8"]["object_stats"]:
            modified_stats["greater_than_0.8"]["object_stats"][category_name] = 1
            modified_stats["greater_than_0.8"]["pixel_stats"][category_name] = number_of_pixels
          else:
            modified_stats["greater_than_0.8"]["object_stats"][category_name] += 1
            modified_stats["greater_than_0.8"]["pixel_stats"][category_name] += number_of_pixels

        num_masks += 1
        print("\rImages Processed: {}/{}".format(i + 1, len(img_Ids)), end=' ')
        sys.stdout.flush()

    with open("coco_objects_stats.json", "w") as f:
      json.dump(object_stats, f)
    
    with open("modified_stats.p", "wb") as f:
      pickle.dump(modified_stats, f)
    
    with open("coco_pixel_stats.p", "wb") as f:
      pickle.dump(pixel_stats, f)

    # Print summary
    print("\nNumber of created segmentation masks: ", num_masks)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == '__main__':
    main()
