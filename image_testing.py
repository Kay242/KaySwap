import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from lib.image_alt import DflPng
from lib.png_meta import DeepFaceLabMetaKeys

original_images_dir = Path("/home/kay/Desktop/xseg/1_TEST_Actual/")
extracted_images_dir = Path("/home/kay/Desktop/xseg/2_TEST_PY/")
pickled_file_path = "/home/kay/Desktop/dfl_model/data_src/Py_Aligned/transformation_info.pkl"


def retrieve_images(originals_dir, extracted_dir) -> Tuple[List[Path], List[Path]]:
    original_images: List[Path] = list(x for x in originals_dir.iterdir() if (x.is_file() and 'info' not in x.stem))
    extracted_images: List[Path] = list(x for x in extracted_dir.iterdir() if (x.is_file() and 'info' not in x.stem))
    return original_images, extracted_images


if __name__ == "__main__":
    og_images, _ = retrieve_images(original_images_dir, extracted_images_dir)
    dfl_png = []
    for og_image in og_images:
        dfl_png.append(DflPng.load(og_image))
    print()
