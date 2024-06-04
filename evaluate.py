"""Module used for evauating models"""
import os
import sys
import importlib
import torch

import options
from util import log

def main():
    """Evaluation Entrypoint"""

    log.process(os.getpid())
    log.title(f"[{sys.argv[0]}] (PyTorch code for evaluating NeRF/BARF)")

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set_opt(opt_cmd=opt_cmd)

    with torch.cuda.device(opt.device):

        model = importlib.import_module(f"model.{opt.model}")
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test")
        m.build_networks(opt)

        if opt.model=="barf":
            m.generate_videos_pose(opt)

        m.restore_checkpoint(opt)
        if opt.data.dataset in ["blender","llff"]:
            m.evaluate_full(opt)
        m.generate_videos_synthesis(opt)

if __name__=="__main__":
    main()
