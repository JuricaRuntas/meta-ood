import argparse
import time
import os
import pickle
import sys
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from config_helpers import config_evaluation_setup
from src.imageaugmentations import Compose, Normalize, ToTensor
from src.model_utils import probs_gt_load
from src.helper import metrics_dump, components_dump, concatenate_metrics, metrics_to_dataset, components_load
from src.model_utils import inference
from multiprocessing import Pool, cpu_count
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression

try:
    from src.metaseg.metrics import compute_metrics_components, compute_metrics_mask
except ImportError:
    compute_metrics_components, compute_metrics_mask = None, None
    print("MetaSeg ImportError: Maybe need to compile (src/metaseg/)metrics.pyx ....")
    exit()


class ClassifierType(Enum):
    LOGISTIC_REGRESSION = 1
    NEURAL_NETWORK = 2


def metaseg_prepare(params, roots, dataset, start_num_images, stop_num_images):
    """Generate Metaseg input which are .hdf5 files"""
    inf = inference(params, roots, dataset, dataset.num_eval_classes)
    for i in range(int(start_num_images), int(stop_num_images)):
        inf.probs_gt_save(i)
    exit(0)


def entropy_segments_mask(probs, t):
    """Generate OoD prediction mask from softmax probabilities"""
    ent = entropy(probs) / np.log(probs.shape[0])
    ent[ent < t] = 0
    ent[ent >= t] = 1
    return ent.astype("uint8")


class compute_metrics(object):
    """
    Compute the hand-crafted segment-wise metrics serving as meta classification input
    """
    def __init__(self, params, roots, dataset, num_imgs=None, metaseg_dir=None, num_cores=1):
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.thresh = params.entropy_threshold
        self.dataset = dataset
        if self.epoch == 0:
            self.load_dir = os.path.join(roots.io_root, "probs/baseline")
            self.save_subdir = "baseline" + "_t" + str(self.thresh)
        else:
            self.load_dir = os.path.join(roots.io_root, "probs/epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha))
            self.save_subdir = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha) + "_t" + str(self.thresh)
        self.num_imgs = len(dataset) if num_imgs is None else num_imgs
        if metaseg_dir is None:
            self.metaseg_dir = os.path.join(roots.io_root, "metaseg_io")
        else:
            self.metaseg_dir = metaseg_dir
        self.num_cores = num_cores

    def compute_metrics_per_image(self, num_cores=None):
        """Perform segment search and compute corresponding segment-wise metrics"""
        print("Calculating statistics for", self.save_subdir, flush=True)
        if num_cores is None:
            num_cores = self.num_cores
        p_args = [(k,) for k in range(self.num_imgs)]
        Pool(num_cores).starmap(self.compute_metrics_pred_i, p_args)
        Pool(num_cores).starmap(self.compute_metrics_gt_i, p_args)

    def compute_metrics_pred_i(self, i):
        """Compute metrics for predicted segments in one image"""
        start_i = time.time()
        probs, gt_train, _, img_path = probs_gt_load(i, load_dir=self.load_dir)
        ent_mask = entropy_segments_mask(probs, self.thresh)
        metrics, components = compute_metrics_components(probs=probs, gt_train=gt_train, ood_mask=ent_mask,
                                                         ood_index=self.dataset.train_id_out)
        metrics_dump(metrics, i, metaseg_root=self.metaseg_dir, subdir=self.save_subdir)
        components_dump(components, i, metaseg_root=self.metaseg_dir, subdir=self.save_subdir)
        print("image", i, "processed in {}s\r".format(round(time.time() - start_i)), flush=True)

    def compute_metrics_gt_i(self, i):
        """Compute metrics for ground truth segments in one image"""
        start_i = time.time()
        probs, gt_train, gt_label, img_path = probs_gt_load(i, load_dir=self.load_dir)
        ent_mask = entropy_segments_mask(probs, self.thresh)
        metrics, components = compute_metrics_mask(probs=probs, mask=ent_mask, gt_train=gt_train, gt_label=gt_label,
                                                   ood_index=self.dataset.train_id_out)
        metrics_dump(metrics, i, metaseg_root=self.metaseg_dir, subdir=self.save_subdir + "_gt")
        components_dump(components, i, metaseg_root=self.metaseg_dir, subdir=self.save_subdir + "_gt")
        print("image", i, "processed in {}s\r".format(round(time.time() - start_i)), flush=True)


class meta_classification(object):
    """
    Perform meta classification with the aid of logistic regressions in order to remove false positive OoD predictions
    """
    def __init__(self, params, roots, dataset=None, num_imgs=None, 
                 metaseg_dir=None, classifier=ClassifierType.LOGISTIC_REGRESSION, 
                 use_pretrained_classifier=False):
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.thresh = params.entropy_threshold
        self.classifier = classifier
        self.use_pretrained_classifier = use_pretrained_classifier
        self.dataset = dataset
        self.net = roots.model_name
        self.weights_dir = roots.weights_dir
        if self.epoch == 0:
            self.load_subdir = "baseline" + "_t" + str(self.thresh)
        else:
            self.load_subdir = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha) + "_t" + str(self.thresh)
        if metaseg_dir is None:
            self.metaseg_dir = os.path.join(roots.io_root, "metaseg_io")
        else:
            self.metaseg_dir = metaseg_dir
        self.num_imgs = len(self.dataset) if num_imgs is None else num_imgs

    def classifier_fit_and_predict(self):
        """Fit a logistic regression and cross validate performance"""
        print(f"""Classifier fit and predict with {'logistic regression' if self.classifier == ClassifierType.LOGISTIC_REGRESSION 
                                                     else 'neural network'}\n""", flush=True)
        metrics, start = concatenate_metrics(metaseg_root=self.metaseg_dir, subdir=self.load_subdir,
                                             num_imgs=self.num_imgs)
        Xa, _, _, y0a, X_names, class_names = metrics_to_dataset(metrics, self.dataset.num_eval_classes)
        y_pred_proba = np.zeros((len(y0a), 2))
        
        loo = LeaveOneOut()

        if not self.use_pretrained_classifier:
            
            if self.classifier == ClassifierType.LOGISTIC_REGRESSION:
                for train_index, test_index in loo.split(Xa):
                    model = LogisticRegression(solver="liblinear")
                    print("TRAIN:", train_index, "TEST:", test_index, flush=True)
                    X_train, X_test = Xa[train_index], Xa[test_index]
                    y_train, y_test = y0a[train_index], y0a[test_index]
                    model.fit(X_train, y_train)
                    y_pred_proba[test_index] = model.predict_proba(X_test)
                
                print("Saving meta classifiers checkpoint", os.path.join(self.metaseg_dir, "metrics", 
                                                                         self.load_subdir, "log_regression_meta_classifier.pkl"), flush=True)
                
                with open(os.path.join(self.metaseg_dir, "metrics", self.load_subdir, "log_regression_meta_classifier.pkl"), "wb") as file:
                    pickle.dump(model, file)
            
            elif self.classifier == ClassifierType.NEURAL_NETWORK:
                for train_index, test_index in loo.split(Xa):
                    print("TRAIN:", train_index, "TEST:", test_index, flush=True)
                    model = nn.Sequential(
                        nn.Linear(Xa.shape[1], Xa.shape[1]),
                        nn.ReLU(),

                        nn.Linear(Xa.shape[1], Xa.shape[1]),
                        nn.ReLU(),
                        
                        nn.Linear(Xa.shape[1], Xa.shape[1]),
                        nn.ReLU(),

                        nn.Linear(Xa.shape[1], 1),
                        nn.Sigmoid()
                    ).cuda()

                    optimizer = optim.Adam(model.parameters(), weight_decay=0.005)
                    criterion = nn.BCELoss()
                    
                    dataset = TensorDataset(torch.tensor(Xa[train_index], dtype=torch.float32).cuda(), 
                                            torch.tensor(y0a[train_index], dtype=torch.float32).cuda())

                    for i in range(50):
                        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
                        
                        for X, Y in dataloader:
                            
                            pred = model(X)
                            loss = criterion(pred.squeeze(), Y)
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    y_pred_proba[test_index[0]][1] = model(torch.tensor(Xa[test_index], dtype=torch.float32).cuda()).item()
                    y_pred_proba[test_index[0]][0] = 1 - y_pred_proba[test_index[0]][1]

                print("Saving meta classifiers checkpoint", 
                      os.path.join(self.metaseg_dir, "metrics", self.load_subdir, self.net + "_" + self.load_subdir + "_meta_classifier.pth"), flush=True)

                torch.save(model.state_dict(), os.path.join(self.metaseg_dir, "metrics", self.load_subdir, 
                                                            self.net + "_" + self.load_subdir + "_meta_classifier.pth"))               
                
        else:
            if self.classifier == ClassifierType.LOGISTIC_REGRESSION:
                
                with open(os.path.join(self.metaseg_dir, "metrics", self.load_subdir, "log_regression_meta_classifier.pkl"), "rb") as f:
                    model = pickle.load(f)

                for i, x in enumerate(Xa):
                    y_pred_proba[i] = model.predict_proba(x.reshape(1,-1))

            elif self.classifier == ClassifierType.NEURAL_NETWORK:
                state_dict = torch.load(os.path.join(self.metaseg_dir, "metrics", self.load_subdir, self.net + "_" + self.load_subdir + "_meta_classifier.pth"))
                
                model.load_state_dict(state_dict, strict=False)

                for i, x in enumerate(Xa):
                    y_pred_proba[i][1] = model(torch.tensor(x, dtype=torch.float32).cuda()).item()
                    y_pred_proba[i][0] = 1 - y_pred_proba[i][1]

        auroc = roc_auc_score(y0a, y_pred_proba[:, 1])
        auprc = average_precision_score(y0a, y_pred_proba[:, 1])
        file_name = "meta_classifier_predictions_logistic.p" if self.classifier == ClassifierType.LOGISTIC_REGRESSION \
                                                             else "meta_classifier_predictions_nn.p"
        save_path = os.path.join(self.metaseg_dir, "metrics", self.load_subdir, file_name)

        with open(save_path, "wb") as f:
            predictions = {"y0a" : y0a, "y_pred_proba": y_pred_proba, "y_pred": np.argmax(y_pred_proba, axis=-1)}
            pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
            print("Saved meta classifier predictions:", save_path, flush=True)

        y_pred = np.argmax(y_pred_proba, axis=-1)
        acc = accuracy_score(y0a, y_pred)
        print("\nMeta classifier performance scores:", flush=True)
        print("AUROC:", auroc, flush=True)
        print("AUPRC:", auprc, flush=True)
        print("Accuracy:", acc, flush=True)

        metrics["kick"] = y_pred
        metrics["start"] = start
        metrics["auroc"] = auroc
        metrics["auprc"] = auprc
        metrics["acc"] = acc

        save_path = os.path.join(self.metaseg_dir, "metrics", self.load_subdir, "meta_classified.p")
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
            print("Saved meta classified:", save_path, flush=True)

        file_name = "meta_classified_logistic.p" if self.classifier == ClassifierType.LOGISTIC_REGRESSION \
                                                             else "meta_classified_nn.p"
        save_path = os.path.join(self.metaseg_dir, "metrics", self.load_subdir, file_name)
        
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
            print("Saved meta classified:", save_path, flush=True)

        return metrics, start

    def remove(self, recompute=False):
        """Based on a meta classifier's decision, remove false positive predictions"""
        print("\nRemoving False positive OoD segment predictions", flush=True)
        load_path = os.path.join(self.metaseg_dir, "metrics", self.load_subdir, "meta_classified.p")
        if os.path.isfile(load_path) and not recompute:
            with open(load_path, "rb") as metrics_file:
                metrics = pickle.load(metrics_file)
                K = metrics["kick"]
                start = metrics["start"]
        else:
            metrics, start = self.classifier_fit_and_predict()
            K = metrics["kick"]
        fn_after = 0
        for i in range(len(start) - 1):
            comp_pred = abs(components_load(i, self.metaseg_dir, self.load_subdir)).flatten()
            for l, k in enumerate(K[start[i]:start[i + 1]]):
                if k == 1:
                    comp_pred[comp_pred == l + 1] = 0
            comp_pred[comp_pred > 0] = 1
            comp_gt = abs(components_load(i, self.metaseg_dir, self.load_subdir + "_gt")).flatten()
            for c in np.unique(comp_gt)[1:]:
                comp_c = np.squeeze([comp_gt == c])
                if np.sum(comp_c[comp_pred > 0]) == 0:
                    fn_after += 1
            print("\rImages Processed: %d,  Num FNs: %d" % (i + 1, fn_after), end=' ', flush=True)
            sys.stdout.flush()
        fp_before = len([i for i in range(len(metrics["iou"])) if metrics["iou"][i] == 0])
        fp_after = np.sum([metrics["kick"] == 1]) - np.sum(np.array(metrics["iou0"])[metrics["kick"] == 1])
        return fn_after, fp_before, fp_after


def main(args):
    config = config_evaluation_setup(args)

    transform = Compose([ToTensor(), Normalize(config.dataset.mean, config.dataset.std)])
    datloader = config.dataset(root=config.roots.eval_dataset_root, split="test", transform=transform)
    start = time.time()

    """Perform Meta Classification"""
    if not args["metaseg_prepare"] and not args["segment_search"] and not args["fp_removal"]:
        args["metaseg_prepare"] = args["segment_search"] = args["fp_removal"] = True
    if args["metaseg_prepare"]:
        print("PREPARE METASEG INPUT", flush=True)
        metaseg_prepare(config.params, config.roots, datloader, args["START_NUM_IMAGES"], args["STOP_NUM_IMAGES"])
    if args["segment_search"]:
        print("SEGMENT SEARCH", flush=True)
        compute_metrics(config.params, config.roots, datloader, num_cores=cpu_count()//2 - 2).compute_metrics_per_image()
    if args["fp_removal"]:
        print("FALSE POSITIVE REMOVAL VIA META CLASSIFICATION", flush=True)
        assert args["METACLASSIFIER"] in ("Regression", "NN", None)
        
        classifier = ClassifierType.LOGISTIC_REGRESSION \
        if args["METACLASSIFIER"] is None or args["METACLASSIFIER"] == "Regression" else ClassifierType.NEURAL_NETWORK
    
        meta_classification(config.params, config.roots, datloader, 
                            classifier=classifier, use_pretrained_classifier=args["pretrained_classifier"]).classifier_fit_and_predict()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), flush=True)


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-val", "--VALSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-start_num_images", "--START_NUM_IMAGES", nargs="?", type=str)
    parser.add_argument("-stop_num_images", "--STOP_NUM_IMAGES", nargs="?", type=str)
    parser.add_argument("-classifier", "--METACLASSIFIER", nargs="?", type=str)
    parser.add_argument("-epoch", "--val_epoch", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-threshold", "--entropy_threshold", nargs="?", type=float)
    parser.add_argument("-prepare", "--metaseg_prepare", action='store_true')
    parser.add_argument("-segment", "--segment_search", action='store_true')
    parser.add_argument("-removal", "--fp_removal", action='store_true')
    parser.add_argument("-pretrained", "--pretrained_classifier", action="store_true")
    main(vars(parser.parse_args()))
