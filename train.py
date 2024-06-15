"""Main training entrypoint, generically usable for all BARF models"""

import os
import sys
import importlib
import torch

import options
from util import log

def main():
    """Main training entrypoint, generically usable for all BARF models"""

    log.process(os.getpid())
    log.title(f"[{sys.argv[0]}] (PyTorch code for training NeRF/BARF)")

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set_opt(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):

        model = importlib.import_module(f"model.{opt.model}")
        m = model.Model(opt)

        m.load_dataset()
        m.build_networks()
        m.setup_optimizer()
        m.restore_checkpoint()
        m.setup_visualizer()

        m.train()

if __name__=="__main__":
    main()
