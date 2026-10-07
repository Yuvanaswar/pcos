"""
PyTorch Dataset and DataLoader implementation for 3-Class Ovarian Ultrasound Benchmark
"""

from typing import Tuple, List, Optional
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.preprocessing import UltrasoundPreprocessingPipeline

class OvarianUltrasoundDataset(Dataset):
    """
    Standardized Dataset for 3-Class Ovarian Ultrasound.
    """
    def __init__(
        self,
        manifest_path: str = "manifest.csv",
        split: str = "train",
        is_train: Optional[bool] = None,
        inpaint: bool = True
    ):
        super().__init__()
        self.split = split
        self.is_train = (split == "train") if is_train is None else is_train
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")
            
        self.transform = UltrasoundPreprocessingPipeline(
            target_size=(224, 224),
            is_train=self.is_train,
            inpaint=inpaint
        )
        
        # Dynamically fix paths from manifest to be relative to the current environment's DATA_DIR
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from config import DATA_DIR
        def fix_path(p):
            p = str(p).replace("\\", "/")
            parts = p.split("/")
            # The structure is usually .../clean_data/split/class/image.png
            # We want to keep split/class/image.png or similar.
            try:
                idx = parts.index("clean_data")
                rel_path = "/".join(parts[idx:])
                return str(DATA_DIR / rel_path)
            except ValueError:
                # If clean_data is not in path, return as is or join with DATA_DIR
                return str(DATA_DIR / Path(p).name)
        
        self.targets = self.data["class_id"].values.astype(np.int64)
        self.paths = [fix_path(p) for p in self.data["image_path"].values]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.data.iloc[idx]
        img_path = self.paths[idx]
        label = int(row["class_id"])
        
        tensor = self.transform(img_path)
        return tensor, label, img_path


def get_dataloaders(
    manifest_path: str = "manifest.csv",
    batch_size: int = 16,
    num_workers: int = 0,
    use_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Builds standard DataLoaders for Train, Validation, and Test sets.
    Computes class weights and applies WeightedRandomSampler on the training split.
    """
    train_dataset = OvarianUltrasoundDataset(manifest_path, split="train", is_train=True)
    val_dataset = OvarianUltrasoundDataset(manifest_path, split="val", is_train=False)
    test_dataset = OvarianUltrasoundDataset(manifest_path, split="test", is_train=False)

    # Class balance weights computation
    train_targets = train_dataset.targets
    classes, counts = np.unique(train_targets, return_counts=True)
    total_samples = len(train_targets)
    
    # Class weights: total / (num_classes * count)
    class_weights = total_samples / (len(classes) * counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Sample weights for WeightedRandomSampler
    if use_sampler:
        sample_weights = np.array([class_weights[t] for t in train_targets])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_weights_tensor
