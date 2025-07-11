import os
from pathlib import Path

BASE_DATA_DIR = Path("/oak/stanford/groups/roxanad/arnavs11")

PATHS = {
    "chexpert": {
        "train": BASE_DATA_DIR / "icl" / "chexpert_train.csv",
        "test": BASE_DATA_DIR / "icl" / "chexpert_test.csv",
    },
    "ddi": {
        "train": BASE_DATA_DIR / "icl" / "ManyICL" / "ManyICL" / "dataset" / "DDI" / "ddi_demo_metadata.csv",
        "test": BASE_DATA_DIR / "icl" / "ManyICL" / "ManyICL" / "dataset" / "DDI" / "ddi_test_metadata.csv",
        "images": BASE_DATA_DIR / "DDI" / "ddidiversedermatologyimages",
    },
    "gastrovision": {
        "train": BASE_DATA_DIR / "Gastrovision" / "processed_trainset.csv",
        "test": BASE_DATA_DIR / "Gastrovision" / "processed_testset.csv",
    }
} 