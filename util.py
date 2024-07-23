"""Utility functions used over the framework"""
import os
import sys
import time
import shutil
import datetime
import socket
import contextlib
import torch
import ipdb
import termcolor
from easydict import EasyDict as edict

def green(message, **kwargs):
    """Print a green string"""
    return termcolor.colored(
        str(message), color="green", attrs=[k for k, v in kwargs.items() if v is True]
        )

def cyan(message, **kwargs):
    """Print a cyan string"""
    return termcolor.colored(
        str(message), color="cyan", attrs=[k for k, v in kwargs.items() if v is True]
        )

def yellow(message, **kwargs):
    """Print a yellow string"""
    return termcolor.colored(
        str(message), color="yellow", attrs=[k for k, v in kwargs.items() if v is True]
        )

def magenta(message, **kwargs):
    """Print a magenta string"""
    return termcolor.colored(
        str(message), color="magenta", attrs=[k for k, v in kwargs.items() if v is True]
        )

def grey(message, **kwargs):
    """Print a grey string"""
    return termcolor.colored(
        str(message), color="grey", attrs=[k for k, v in kwargs.items() if v is True]
        )

class Log:
    """Logger Class"""
    def process(self, pid):
        """Print the given process id in grey"""
        print(grey(f"Process ID: {pid}", bold=True))

    def title(self, message):
        """Print a title string"""
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        """Print an info string"""
        print(magenta(message, bold=True))

    def options(self, opt, level=0):
        """Print the used options of this training cycle"""
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value, level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":", yellow(value))

log = Log()

def update_timer(opt, timer, ep, it_per_ep):
    """Update the timer used with the loader with the currently valid values."""
    if not opt.max_epoch:
        return
    momentum = 0.99
    timer.elapsed = time.time() - timer.start
    timer.it = timer.it_end - timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean * momentum + timer.it * \
        (1 - momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean * it_per_ep * (opt.max_epoch - ep)

def move_to_device(x, device):
    """Move the given input value to the selected device (cpu or cuda)"""
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = move_to_device(v, device)
    elif isinstance(x, list):
        for i, e in enumerate(x):
            x[i] = move_to_device(e, device)
    elif isinstance(x, tuple) and hasattr(x, "_fields"):  # collections.namedtuple
        dd = x._asdict()
        dd = move_to_device(dd, device)
        return type(x)(**dd)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device)
    return x

def to_dict(d, dict_type=dict):
    """Convert the given input value to a python dictionary"""
    d = dict_type(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = to_dict(v, dict_type)
    return d

def get_layer_dims(layers):
    """Create layer tuples out of an array defining the layer dimensionalities"""
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))

def colorcode_to_number(code):
    """Convert a color code to a number"""
    ords = [ord(c) for c in code[1:]]
    ords = [n - 48 if n < 58 else n - 87 for n in ords]
    rgb = (ords[0] * 16 + ords[1], ords[2] * 16 + ords[3], ords[4] * 16 + ords[5])
    return rgb
