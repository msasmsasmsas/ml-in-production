# E:\ml-in-production\HW5_Training_Experiments\PR1\dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_simple_transforms(image_size, is_training=True):
    """
    Creates simple transformations for images
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class FastAgriculturalRiskDataset(Dataset):
    """
    Dataset for agricultural risk classification
    """

    def __init__(self, csv_file, image_dir, transform=None, risk_type="all"):
        self.image_dir = image_dir
        self.transform = transform
        self.risk_type = risk_type

        # Load data
        self.data_df = pd.read_csv(csv_file)

        # Create class mapping
        self.classes = self.data_df["disease_name"].unique()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Prepare data entries
        self.data = []
        for _, row in self.data_df.iterrows():
            class_name = row["disease_name"]
            image_path = row["image_path"]

            if os.path.exists(image_path):
                self.data.append({
                    "image_path": image_path,
                    "class": class_name,
                    "class_idx": self.class_to_idx.get(class_name, 0)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, sample["class_idx"]
        except Exception as e:
            # Return a dummy image in case of error
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, sample["class_idx"]