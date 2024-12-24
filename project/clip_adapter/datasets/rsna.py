import os
import pickle
import random
import pandas as pd
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, read_json, write_json

@DATASET_REGISTRY.register()
class RSNAPneumonia(DatasetBase):

    dataset_dir = "rsna_pneumonia"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir_train = os.path.join(self.dataset_dir, "train_jpg")
        self.image_dir_test = os.path.join(self.dataset_dir, "test_jpg")
        self.label_file = os.path.join(self.dataset_dir, "train.csv")
        self.test_label_file = os.path.join(self.dataset_dir, "test.csv")
        self.split_path = os.path.join(self.dataset_dir, "split_rsna.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir_train, self.image_dir_test)
        else:
            trainval = self.read_data()
            train, val = self.split_trainval(trainval)
            test = self.read_test_data()
            self.save_split(train, val, test, self.split_path)

        print(len(train), 'Train Len After Reading Split')
        print(len(val), 'Val Len After Reading Split')
        print(len(test), 'Test Len After Reading Split')

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
                    print(len(train), 'Train Size')
                    print(len(val), 'Val Size')
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, val, test = self.subsample_classes(train, val, test)

        print(len(train), 'Train Len After Subsample')
        print(len(val), 'Val Len After Subsample')
        print(len(test), 'Test Len After Subsample')

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self):
        """Read training data and labels."""
        df = pd.read_csv(self.label_file)
        items = []
        for _, row in df.iterrows():
            patient_id = row["patientId"]
            label = row["Target"]
            img_path = os.path.join(self.image_dir_train, f"{patient_id}.jpg")
            classname = "pneumonia" if label == 1 else "normal"
            item = Datum(impath=img_path, label=label, classname=classname)
            items.append(item)
        return items

    def read_test_data(self):
        """Read test data and labels from test.csv."""

        df = pd.read_csv(self.test_label_file)
        items = []    
        for _, row in df.iterrows():
            patient_id = row["patientId"]
            label = row["Target"]
            img_path = os.path.join(self.image_dir_test, f"{patient_id}.jpg")
            classname = "pneumonia" if label == 1 else "normal"
            item = Datum(impath=img_path, label=label, classname=classname)
            items.append(item)
        return items


    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        """Split training data into train and validation sets."""
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val
    
    @staticmethod
    def save_split(train, val, test, filepath):
        """Save train/val/test splits to a JSON file."""
        def _extract(items):
            return [(item.impath, item.label, item.classname) for item in items]

        split = {
            "train": _extract(train),
            "val": _extract(val),
            "test": _extract(test),
        }

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, train_path_prefix, test_path_prefix):
        """Read train/val/test splits from a JSON file."""
        def _convert(items, path_prefix):
            return [Datum(
                impath=os.path.join(path_prefix, impath),
                label=int(label),
                classname=classname
            ) for impath, label, classname in items]

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"], train_path_prefix)
        val = _convert(split["val"], train_path_prefix)
        test = _convert(split["test"], test_path_prefix)
        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output

    