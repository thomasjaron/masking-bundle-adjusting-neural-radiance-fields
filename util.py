"""Docstring"""
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

# convert to colored strings


def red(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="red", attrs=[k for k, v in kwargs.items() if v is True]
        )


def green(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="green", attrs=[k for k, v in kwargs.items() if v is True]
        )


def blue(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="blue", attrs=[k for k, v in kwargs.items() if v is True]
        )


def cyan(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="cyan", attrs=[k for k, v in kwargs.items() if v is True]
        )


def yellow(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="yellow", attrs=[k for k, v in kwargs.items() if v is True]
        )


def magenta(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="magenta", attrs=[k for k, v in kwargs.items() if v is True]
        )


def grey(message, **kwargs):
    """Docstring"""
    return termcolor.colored(
        str(message), color="grey", attrs=[k for k, v in kwargs.items() if v is True]
        )


def get_time(sec):
    """Docstring"""
    d = int(sec//(24*60*60))
    h = int(sec//(60*60) % 24)
    m = int((sec//60) % 60)
    s = int(sec % 60)
    return d, h, m, s


def add_datetime(func):
    """Docstring"""
    def wrapper(*args, **kwargs):
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(grey(f"[{datetime_str}] ", bold=True), end="")
        return func(*args, **kwargs)
    return wrapper


def add_functionname(func):
    """Docstring"""
    def wrapper(*args, **kwargs):
        print(grey(f"[{func.__name__}] ", bold=True))
        return func(*args, **kwargs)
    return wrapper


def pre_post_actions(pre=None, post=None):
    """Docstring"""
    def func_decorator(func):
        def wrapper(*args, **kwargs):
            if pre:
                pre()
            retval = func(*args, **kwargs)
            if post:
                post()
            return retval
        return wrapper
    return func_decorator


debug = ipdb.set_trace


class Log:
    """Logger Class"""
    def process(self, pid):
        """Docstring"""
        print(grey(f"Process ID: {pid}", bold=True))

    def title(self, message):
        """Docstring"""
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        """Docstring"""
        print(magenta(message, bold=True))

    def options(self, opt, level=0):
        """Docstring"""
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value, level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":", yellow(value))

log = Log()


def update_timer(opt, timer, ep, it_per_ep):
    """Docstring"""
    if not opt.max_epoch:
        return
    momentum = 0.99
    timer.elapsed = time.time()-timer.start
    timer.it = timer.it_end-timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean*momentum+timer.it * \
        (1-momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean*it_per_ep*(opt.max_epoch-ep)

# move tensors to device in-place


def move_to_device(x, device):
    """Docstring"""
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
    """Docstring"""
    d = dict_type(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = to_dict(v, dict_type)
    return d


def get_child_state_dict(state_dict, key):
    """Docstring"""
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith(f"{key}.")}


def restore_checkpoint(opt, model, load_name=None, resume=False):
    """Docstring"""
    # resume can be True/False or epoch numbers
    assert (load_name is None) == (resume is not False)
    if resume:
        load_name = f"{opt.output_path}/model.ckpt" if resume is True else \
                    f"{opt.output_path}/model/{resume}.ckpt"
    checkpoint = torch.load(load_name, map_location=opt.device)
    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"], name)
        if child_state_dict:
            print(f"restoring {name}...")
            child.load_state_dict(child_state_dict)
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print(f"restoring {key}...")
            getattr(model, key).load_state_dict(checkpoint[key])
    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        if resume is not True:
            assert resume == (ep or it)
        print(f"resuming from epoch {ep} (iteration {it})")
    else:
        ep, it = None, None
    return ep, it


def save_checkpoint(opt, model, ep, it, latest=False, children=None):
    """Docstring"""
    os.makedirs(f"{opt.output_path}/model", exist_ok=True)
    if children is not None:
        graph_state_dict = {
            k: v for k, v in model.graph.state_dict().items() if k.startswith(children)}
    else:
        graph_state_dict = model.graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            if val := getattr(model, key):
                checkpoint.update({key: val.state_dict()})
    torch.save(checkpoint, f"{opt.output_path}/model.ckpt")
    if not latest:
        shutil.copy(f"{opt.output_path}/model.ckpt",
                    # if ep is None, track it instead
                    f"{opt.output_path}/model/{ep or it}.ckpt")


def check_socket_open(hostname, port):
    """Docstring"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def get_layer_dims(layers):
    """Docstring"""
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))


@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    """Docstring"""
    with open(os.devnull, "w") as devnull: # pylint: disable=unspecified-encoding
        if stdout:
            old_stdout, sys.stdout = sys.stdout, devnull
        if stderr:
            old_stderr, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            if stdout:
                sys.stdout = old_stdout
            if stderr:
                sys.stderr = old_stderr


def colorcode_to_number(code):
    """Docstring"""
    ords = [ord(c) for c in code[1:]]
    ords = [n-48 if n < 58 else n-87 for n in ords]
    rgb = (ords[0]*16+ords[1], ords[2]*16+ords[3], ords[4]*16+ords[5])
    return rgb


# def align_images_with_keypoints(self):
#     """Testing with keypoints alignment -Thomas """
#     # Load the images
#     first = self.image_batches[0].cpu().permute(1,2,0).numpy() # H x W x C

#     images = []
#     masks = []

#     images.append(self.image_batches[0])
#     masks.append(self.mask_batches[0])

#     for i in range(1, self.batch_size):
#         img = self.image_batches[i]
#         mask = self.mask_batches[i]
#         # Convert images to grayscale
#         gray1 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(img.cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)

#         # Define the motion model
#         warp_mode = cv2.MOTION_HOMOGRAPHY

#         # Initialize the matrix to identity
#         warp_matrix = np.eye(3, 3, dtype=np.float32)

#         # Number of iterations
#         number_of_iterations = 10000

#         # Specify the threshold of the increment
#         termination_eps = 1e-10

#         # Define the termination criteria
#         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

#         # Run the ECC algorithm. The results are stored in warp_matrix.
#         (cc, warp_matrix) = cv2.findTransformECC(gray1, gray2, warp_matrix, warp_mode, criteria)

#         # Use warpPerspective for Homography
#         img2_aligned = cv2.warpPerspective(img.cpu().permute(1,2,0).numpy(), warp_matrix, (first.shape[1], first.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#         mask_aligned = cv2.warpPerspective(mask.cpu().permute(1,2,0).numpy(), warp_matrix, (first.shape[1], first.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

#         img_new = torchvision_F.to_tensor(img2_aligned).to(self.opt.device)
#         mask_new = torchvision_F.to_tensor(mask_aligned).to(self.opt.device)

#         images.append(img_new)
#         masks.append(mask_new)
#         imageio.imsave(f"{i}.png", (img2_aligned * 255).astype(np.uint8))
#         imageio.imsave(f"{i}-m.png", (mask_aligned * 255).astype(np.uint8))
#     self.image_batches = torch.stack(images) # [B, 3, H, W]
#     self.mask_batches = torch.stack(masks) # [B, 3, H, W]