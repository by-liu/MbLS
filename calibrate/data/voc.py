import os.path as osp
import numpy as np
import cv2
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


class VOCSegmentation(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        data_transform: Optional[Callable] = None,
        return_id=False
    ):
        assert split in {"train", "val", "trainval", "test"}
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.data_transform = data_transform
        self.return_id = return_id

        self.classes = CLASSES
        self.num_classes = 21

        self.load_list()

    def load_list(self):
        self.img_dir = osp.join(self.data_root, "JPEGImages")
        self.mask_dir = osp.join(self.data_root, "SegmentationClass")
        self.split_dir = osp.join(self.data_root, "ImageSets/Segmentation")

        split_file = osp.join(self.split_dir, "{}.txt".format(self.split))

        with open(split_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [osp.join(self.img_dir, x + ".jpg") for x in file_names]
        self.masks = [osp.join(self.mask_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

    def convert_to_segmentation_mask(self, mask, onehot=False):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, self.num_classes), dtype=long
        )
        for label_index, label in enumerate(PALETTE):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(long)

        return segmentation_mask

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(self.masks[index]))

        if self.data_transform is not None:
            result = self.data_transform(
                image=img, mask=mask
            )
            img = result["image"]
            mask = result["mask"].long()

        if self.return_id:
            return (
                img, mask, self.images[index].split("/")[-1].split(".")[0]
            )
        else:
            return img, mask

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return (
            "VOCSegmentation (data_root={},split={})\tSamples : {}".format(
                self.data_root, self.split, self.__len__()
            )
        )


def data_transformer(is_train: bool = True):
    if is_train:
        transformer = A.Compose([
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(
                min_height=512, min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0), mask_value=255
            ),
            A.RandomCrop(height=512, width=512),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.LongestMaxSize(max_size=480),
            A.PadIfNeeded(
                min_height=480, min_width=480,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0), mask_value=255
            ),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


def get_train_val_loader(
    data_root, batch_size=32, num_workers=8, pin_memory=True
):
    train_dataset = VOCSegmentation(
        data_root,
        split="train",
        data_transform=data_transformer(is_train=True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_dataset = VOCSegmentation(
        data_root,
        split="val",
        data_transform=data_transformer(is_train=False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_test_loader(
    data_root, batch_size=32,
    num_workers=8, pin_memory=True
):
    test_dataset = VOCSegmentation(
        data_root,
        split="val",
        data_transform=data_transformer(is_train=False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader
