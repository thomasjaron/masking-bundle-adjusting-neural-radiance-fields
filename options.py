"""Module for parsing and working with options deifned in ./options directory"""
import os
import random
import string
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

import util
from util import log


def parse_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
    """
    opt_cmd = {}
    for arg in args:
        assert arg.startswith("--")
        if "=" not in arg[2:]:
            key_str, value = (
                arg[2:-1], "false") if arg[-1] == "!" else (arg[2:], "true")
        else:
            key_str, value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub:
                opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub, keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd


def set_opt(opt_cmd=None):
    """Set Options"""
    if opt_cmd is None:
        opt_cmd = {}
    log.info("setting configurations...")
    assert "model" in opt_cmd
    # load config from yaml file
    assert "yaml" in opt_cmd
    fname = f"options/{opt_cmd.yaml}.yaml"
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(opt_base, opt_cmd, key_stack=[], safe_check=True)
    process_options(opt)
    log.options(opt)
    return opt


def load_options(fname):
    """Load Options"""
    with open(fname) as file:  # pylint: disable=unspecified-encoding
        opt = edict(yaml.safe_load(file))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if isinstance(parent_fnames, str):
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = override_options(opt_parent, opt, key_stack=[])
            opt = opt_parent
    print(f"loading {fname}...")
    return opt


def override_options(opt, opt_over, key_stack=None, safe_check=False):
    """Override Options"""
    for key, value in opt_over.items():
        if isinstance(value, dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(
                opt.get(key, dict()), value, key_stack=key_stack+[key], safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y", "n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input(
                        f'"{key_str}" not found in original opt, add? (y/n) '
                    )
                if add_new == "n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt


def process_options(opt):
    """Process options"""
    # set seed
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        if opt.seed != 0:
            opt.name = str(opt.name) + f"_seed{opt.seed}"
    else:
        # create random string as run ID
        randkey = "".join(random.choice(string.ascii_uppercase)
                          for _ in range(4))
        opt.name = str(opt.name) + f"_{randkey}"
    # other default options
    opt.output_path = f"{opt.output_root}/{opt.group}/{opt.name}"
    os.makedirs(opt.output_path, exist_ok=True)
    # disable multi-GPU support for now, single is enough
    assert isinstance(opt.gpu, int)
    opt.device = "cpu" if opt.cpu or not torch.cuda.is_available(
    ) else f"cuda:{opt.gpu}"
    opt.H, opt.W = opt.data.image_size


def save_options_file(opt):
    """Save Options File"""
    opt_fname = f"{opt.output_path}/options.yaml"
    if os.path.isfile(opt_fname):
        with open(opt_fname) as file:  # pylint: disable=unspecified-encoding
            opt_old = yaml.safe_load(file)
        if opt != opt_old:
            # prompt if options are not identical
            opt_new_fname = f"{opt.output_path}/options_temp.yaml"
            with open(opt_new_fname, "w") as file:  # pylint: disable=unspecified-encoding
                yaml.safe_dump(util.to_dict(opt), file,
                               default_flow_style=False, indent=4)
            print("existing options file found (different from current one)...")
            os.system(f"diff {opt_fname} {opt_new_fname}")
            os.system(f"rm {opt_new_fname}")
            override = None
            while override not in ["y", "n"]:
                override = input("override? (y/n) ")
            if override == "n":
                print("safe exiting...")
                exit()
        else:
            print("existing options file found (identical)")
    else:
        print("(creating new options file...)")
    with open(opt_fname, "w") as file:  # pylint: disable=unspecified-encoding
        yaml.safe_dump(util.to_dict(opt), file,
                       default_flow_style=False, indent=4)
