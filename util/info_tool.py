
from ast import arg
import time
import os 
import torch

def write_info(*args):
    text = ""
    for arg in args:
        text += " " + str(arg)
    print("#=======", text, " =======#")

    
def create_name(args, prefix=""):
    # tensorboard_dir = "summary/"
    if args.named == 'prototree-fl':
        time_str = time.strftime('%m%d%H%M')
        if prefix:
            tensorboard_name = prefix
            tensorboard_name += "_"+args.fed_method
        else:
            tensorboard_name = args.fed_method
        # tensorboard_name += "_"+str(args.classifier)
        tensorboard_name += "_C"+args.client_classifier
        tensorboard_name += "_S"+args.server_classifier

        if args.server_resampler:
            tensorboard_name += "_SerResmpl"
        if args.client_resampler:
            tensorboard_name += "_CliResmpl"
        if args.fed_method == "gkt":
            tensorboard_name += "_"+str(args.local_epoch)+"Lepoch"
            tensorboard_name += "_"+str(args.global_epoch)+"Sepoch"
        else:
            tensorboard_name += "_"+str(args.local_epoch)+"epoch"
        tensorboard_name += "_"+str(args.num_features)+"feat"
        

        tensorboard_name += "_"+args.net
        tensorboard_name += "_"+time_str
        if not args.comment == "":
            tensorboard_name += "_"+args.comment
    else:
        tensorboard_name = args.named
    # tensorboard_name = tensorboard_dir+tensorboard_name
    return tensorboard_name

def load_ckpt(args):
    ckpt_dir = "ckpt/"
    ckpt_dir = ckpt_dir+args.ckpt_dir+".pt"
    upsample_dir = "result/"
    try:
        # --- Load ckpt --- 
        checkpoint = torch.load(ckpt_dir)
        # --- Create dir for ckpt upsample tree --- 
        if not os.path.exists(upsample_dir):
            os.makedirs(upsample_dir)
        save_dir = upsample_dir+args.ckpt_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # --- Load args in ckpt --- 
        args = checkpoint['args_dict']
        return checkpoint, args, save_dir
    except:
        raise Exception(f'Could not find checkpoint from "{ckpt_dir}"!')












