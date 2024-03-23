import os
from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple
from src.dataset.cityscapes import Cityscapes
from config import CS_ROOT, FS_ROOT

class Fishyscapes(Dataset):

    FishyscapesClass = namedtuple('FishyscapesClass', ['name', 'id', 'train_id', 'hasinstances',
                                                       'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        FishyscapesClass('in-distribution', 0, 0, False, False, (144, 238, 144)),
        FishyscapesClass('out-distribution', 1, 1, False, False, (255, 102, 102)),
        FishyscapesClass('unlabeled', 2, 255, False, True, (0, 0, 0)),
    ]

    train_id_in = 0
    train_id_out = 1
    cs = Cityscapes(CS_ROOT)
    mean = cs.mean
    std = cs.std
    num_eval_classes = cs.num_train_ids
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, split='test', root=FS_ROOT, transform=None, useLostAndFoundSubset=False):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.type = "LostAndFound" if useLostAndFoundSubset else "Static"
        self.split = split
        self.images = []        # list of all raw input images
        self.targets = []       # list of all ground truth TrainIds images
        for filename in os.listdir(os.path.join(root, self.type, "original")):
            if os.path.splitext(filename)[1] == '.png':
                filename_base = os.path.splitext(filename)[0]
                self.images.append(os.path.join(root, self.type, "original", filename_base + '.png'))
                self.targets.append(os.path.join(root, self.type, "labels", filename_base + '.png'))
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()
