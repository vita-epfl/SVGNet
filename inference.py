from torch.utils.data import DataLoader, Subset
from deepsvg.config import _Config
import torch.nn as nn
from src.lyft.data import build_rasterizer
from l5kit.data import LocalDataManager, ChunkedDataset
import argparse
import importlib
from l5kit.configs import load_config_data
from src.model_and_dataset.svg_dataset import SVGDataset
from src.argoverse.utils.svg_utils import BaseDataset
from torchvision.models.resnet import resnet18,resnet50,resnet34
import cv2
from src.model_and_dataset.models.model_trajectory import ModelTrajectory
import math
import copy
from collections import OrderedDict
from tqdm import tqdm

model_cfg = importlib.import_module("configs.deepsvg.hierarchical_ordered").Config()

from src.model_and_dataset.models.model_trajectory import ModelTrajectory
from src.model_and_dataset.utils import neg_multi_log_likelihood
from src.model_and_dataset.svg_dataset import SVGDataset

from deepsvg.utils import Stats, TrainVars, Timer
import torch
from deepsvg import utils
from datetime import datetime
from tensorboardX import SummaryWriter
from deepsvg.utils.stats import SmoothedValue
import os
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
import pandas as pd
from src.argoverse.utils.evaluation import get_ade

import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union
import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# from logger import Logger
import src.argoverse.utils.baseline_config as config
import src.argoverse.utils.baseline_utils as baseline_utils
from src.argoverse.utils.map_features_utils import MapFeaturesUtils
from src.argoverse.utils.transform import *
import numpy as np
import matplotlib.pyplot as plt
CV2_SHIFT = 8  # how many bits to shift in drawing
from argoverse.evaluation.competition_util import generate_forecasting_h5










args = argparse.Namespace()
args.end_epoch=5000
args.joblib_batch_size=100
args.lr=0.001
args.model_path=None
args.normalize=True
args.obs_len=20
args.pred_len=30
args.test=False
args.test_batch_size=512
args.test_features='../../../forecasting_features_test.pkl'
args.train_batch_size=512
args.train_features='../../../forecasting_features_val.pkl'
args.traj_save_path=None
args.use_delta=False
args.use_map=False
args.use_social=False
args.val_batch_size=512
args.val_features='../../../forecasting_features_val.pkl'
# key for getting feature set
    # Get features
if args.use_map and args.use_social:
    baseline_key = "map_social"
elif args.use_map:
    baseline_key = "map"
elif args.use_social:
    baseline_key = "social"
else:
    baseline_key = "none"


NUM_gesses=6

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    # print(batch)
    # batch = filter(lambda x:x is not None, batch)
    batch = list(filter(None, batch))
    # print(batch)
    if len(batch) > 0:
        return default_collate(batch)
    else:
        return


    
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
    print("using cuda")
else:
    device = torch.device("cpu")
    print("using cpu")

# Get data
data_dict = baseline_utils.get_data(args, baseline_key)





model = ModelTrajectory(model_cfg=model_cfg,data_config=None, modes=1, future_len=30, in_channels=3)
model_path="/work/vita/sadegh/argo/argoverse-forecasting/utils/download/raster-svg/logs/svg-H20S-2gpu-bs32-tt1870-V6/checkpoint-9000"
checkpoint = torch.load(model_path,map_location=device)
print(model_path)

new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace("module.", "") # remove `module.`
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model = nn.DataParallel(model)

test_dataset = SVGDataset(data_type = "argo",model_args=model_cfg.model_args,
                                   max_num_groups=model_cfg.max_num_groups,max_seq_len=model_cfg.max_seq_len,
                                   data_dict=data_dict, args=args, mode="test")

bs=16

dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                                  num_workers=model_cfg.loader_num_workers,collate_fn=my_collate)


forecasted_trajectories={}
nums=0






progress_bar = tqdm(dataloader)
# for batch_idx,data in enumerate(test_dataloader):
for dd in progress_bar:
    model.eval()
    with torch.no_grad():

        ego_yaw=(-math.pi*dd["yaw_deg"]/180).numpy()
        centroids=dd["centroid"].numpy()
        
        seq_id=dd["seq_id"].numpy()
        model_args = [dd["image"][arg].to(device) for arg in model_cfg.model_args]
        entery = [*model_args, {}, True]
        output,conf = model(entery)
        output=output.reshape((dd["history_positions"].shape[0],NUM_gesses,30,2)).cpu()
        for i in range(output.shape[0]):
            forcasted_trajs=[]
            for j in range(NUM_gesses):
                rot_output=transform_points(output[i][j], yaw_as_rotation33(ego_yaw[i]))+centroids[i]
                forcasted_trajs.append(rot_output)
            forecasted_trajectories[seq_id[i]] = forcasted_trajs

output_path = 'me_competition_files/'

generate_forecasting_h5(forecasted_trajectories, output_path) #this might take awhile
         