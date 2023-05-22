import argparse
import logging

from pathlib import Path
from lib.logger import FaceswapLogger
from tools.manual.manual import Manual
from lib.cli.launcher import ScriptExecutor
from lib.cli.args import TrainArgs, FullHelpArgumentParser

side_a_path = Path("")
side_b_path = Path("/run/media/kay/Windows HDD/AI/z1_SideB/maddii_edee/1_Inputs/Images/2")
side_b_align = Path("/run/media/kay/Windows HDD/AI/z1_SideB/maddii_edee/1_Inputs/Images/2/alignments.fsa")
model_path = Path("")

args: argparse.Namespace = argparse.Namespace()
parser = FullHelpArgumentParser()


def man():
    args.alignments_path = side_b_align
    args.frames = side_b_path
    args.exclude_gpus = None
    args.thumb_regen = False
    args.single_process = False
    args.loglevel = "INFO"

    cus_manual: Manual = Manual(args)
    cus_manual.process()


def train():
    logging.setLoggerClass(FaceswapLogger)

    args.input_a = str(side_a_path)
    args.input_b = str(side_b_path)
    args.model_dir = str(model_path)
    args.load_weights = False
    args.trainer = "villain"
    args.summary = False
    args.freeze_weights = False
    args.batch_size = 4
    args.iterations = int(600000)
    args.distributed = False
    args.save_interval = 250
    args.snapshot_interval = 25000
    args.timelapse_input_a = None
    args.timelapse_input_b = None
    args.timelapse_output = None
    args.preview = True
    args.write_image = False
    args.no_logs = False
    args.preview_scale = 100
    args.warp_to_landmarks = True
    args.no_flip = False
    args.no_augment_color = False
    args.no_warp = False
    args.exclude_gpus = None
    args.redirect_gui = False
    args.colab = False
    args.loglevel = "Debug"
    args.logfile = "loggg.log"
    args.configfile = None

    script_exec = ScriptExecutor("train")
    script_exec.execute_script(args)

    #train_script = Train(args)
    #train_script.process()


if __name__ == "__main__":
    man()
    #train()
