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
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):

        model = importlib.import_module(f"model.{opt.model}")
        m = model.Model(opt)

        m.load_dataset(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.restore_checkpoint(opt)
        m.setup_visualizer(opt)

        m.train(opt)

if __name__=="__main__":
    main()
