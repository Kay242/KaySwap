from pathlib import Path
from typing import List
from lib.deepfacelab.leras.neuralnet import NeuralNet
from lib.image_alt import DflPng
from plugins.train.trainer import xseg
from scripts import xseg_util
from tools.xseg import xseg_editor

#dst_step3_path = Path("")
src_step3_path = Path("/home/kay/Documents/ML_AI/ryan_reynolds/4_DFL_XSEG_Editor/")
#dst_step4_path = Path("")
src_step4_path = Path("/home/kay/Documents/ML_AI/ryan_reynolds/5_DFL_XSEG_Train/")
src_step5_path = Path("/home/kay/Documents/ML_AI/ryan_reynolds/6_DFL_XSEG_Applied/")

xseg_model_path = Path("/home/kay/Documents/ML_AI/ryan_reynolds/5_XSEG_Model/")


def first_time_setup_extracted_fs_faces_for_xseg(step_3_image_dir: Path):
    images: List[Path] = list(x for x in step_3_image_dir.iterdir())
    for image in images:
        dfl_png_img: DflPng = DflPng.load(image)
        dfl_png_img.save()


def launch_xseg_editor(aligned_data_path: Path):
    xseg_editor.start(aligned_data_path)


def apply_xseg(path_to_apply: Path, model_path: Path):
    NeuralNet.initialize_main_env()
    xseg_util.apply_xseg(path_to_apply, model_path)


def xseg_train_src_or_dst(path_to_train: Path, model_path: Path, is_source: bool = True):
    src_path: Path
    dst_path: Path
    if is_source:
        src_path = path_to_train
        dst_path = Path("/")
    else:
        dst_path = path_to_train
        src_path = Path("/")

    NeuralNet.initialize_main_env()
    NeuralNet.initialize()
    xseg.main(training_data_dst_path=dst_path,
              training_data_src_path=src_path,
              saved_models_path=model_path,
              silent_start=False,
              debug=False)


if __name__ == "__main__":
    #first_time_setup_extracted_fs_faces_for_xseg(src_step3_path)

    #launch_xseg_editor(src_step3_path)
    xseg_train_src_or_dst(src_step4_path, xseg_model_path, is_source=True)

    #apply_xseg(src_step5_path, xseg_model_path)
    #launch_xseg_editor(src_step5_path)
