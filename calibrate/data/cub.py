import os.path as osp
from albumentations.augmentations.geometric.resize import SmallestMaxSize
import cv2
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import Callable, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CUBDataset(Dataset):
    def __init__(self, data_root,
                 is_train: bool = True,
                 transformer: Optional[Callable] = None) -> None:
        super().__init__()
        self.data_root = data_root
        self.img_dir = osp.join(self.data_root, "images")
        self.is_train = is_train
        self.transformer = transformer
        self.load_list()

    def load_list(self):
        img_txt_file = osp.join(self.data_root, "images.txt")
        with open(img_txt_file, "r") as f:
            all_img_names = [
                line.strip().split(" ")[-1]
                for line in f
            ]
        label_txt_file = osp.join(self.data_root, "image_class_labels.txt")
        with open(label_txt_file, "r") as f:
            all_labels = [
                int(line.strip().split(" ")[-1]) - 1
                for line in f
            ]
        train_test_file = osp.join(self.data_root, "train_test_split.txt")
        with open(train_test_file, "r") as f:
            train_test = [
                int(line.strip().split(" ")[-1])
                for line in f
            ]

        if self.is_train:
            self.img_names = [
                x for i, x in zip(train_test, all_img_names) if i
            ]
            self.labels = [
                x for i, x in zip(train_test, all_labels) if i
            ]
        else:
            self.img_names = [
                x for i, x in zip(train_test, all_img_names) if not i
            ]
            self.labels = [
                x for i, x in zip(train_test, all_labels) if not i
            ]

    def __getitem__(self, i: int):
        label = self.labels[i]

        img_path = osp.join(self.img_dir, self.img_names[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transformer is not None:
            result = self.transformer(image=img)
            img = result["image"]

        return img, label

    def __len__(self) -> int:
        return len(self.img_names)

    def __repr__(self) -> str:
        return (
            "CUBDataset(data_root={}, is_train={}\tSamples : {})".format(
                self.data_root, self.is_train, self.__len__()
            )
        )


def data_transformer(is_train: bool = True, scale_size=256, crop_size=224):
    if is_train:
        transformer = A.Compose([
            A.Resize(scale_size, scale_size),
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.Resize(scale_size, scale_size),
            A.CenterCrop(crop_size, crop_size),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


def get_train_val_loader(
    data_root, batch_size=32, scale_size=256, crop_size=224,
    num_workers=8, pin_memory=True
):
    train_dataset = CUBDataset(
        data_root=data_root,
        is_train=True,
        transformer=data_transformer(
            is_train=True, scale_size=scale_size, crop_size=crop_size
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_dataset = CUBDataset(
        data_root=data_root,
        is_train=False,
        transformer=data_transformer(
            is_train=False, scale_size=scale_size, crop_size=crop_size
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def get_test_loader(
    data_root, batch_size=32, scale_size=256, crop_size=224,
    num_workers=8, pin_memory=True
):
    test_dataset = CUBDataset(
        data_root=data_root,
        is_train=False,
        transformer=data_transformer(
            is_train=False, scale_size=scale_size, crop_size=crop_size
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return test_loader
