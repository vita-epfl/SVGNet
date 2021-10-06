from torch.utils.data import DataLoader, Subset
from deepsvg.config import _Config
import torch.nn as nn
from src.lyft.data import build_rasterizer
from l5kit.data import LocalDataManager, ChunkedDataset
import argparse
import importlib
from l5kit.configs import load_config_data
from src.model_and_dataset.svg_dataset import SVGDataset

from src.model_and_dataset.models.model_trajectory import ModelTrajectory
from src.model_and_dataset.models.cheese import Cheese
from src.model_and_dataset.models.mlp_added_transformer import MLPTransformer
from src.model_and_dataset.utils import neg_multi_log_likelihood

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
from src.argoverse.utils.evaluation import get_ade

from src.argoverse.utils.raster_utils import RasterDataset


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


def train(model_cfg: _Config, args, model_name, experiment_name="", log_dir="./logs", debug=False, resume=" "):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_type == "lyft":
        data_cfg = load_config_data(args.config_data)
        # set env variable for data
        model_cfg.print_params()
        dm = LocalDataManager(args.data_path)
        # get config
        rasterizer = build_rasterizer(data_cfg, dm)

        train_zarr = ChunkedDataset(dm.require(data_cfg["train_dataloader"]["split"])).open()
        val_zarr = ChunkedDataset(dm.require(data_cfg["val_dataloader"]["split"])).open()

        train_dataset = SVGDataset(data_type="lyft", model_args=model_cfg.model_args,
                                   max_num_groups=model_cfg.max_num_groups, max_seq_len=model_cfg.max_seq_len,
                                   data_cfg=data_cfg, zarr_dataset=train_zarr, rasterizer=rasterizer)
        val_dataset = SVGDataset(data_type="lyft", model_args=model_cfg.model_args,
                                 max_num_groups=model_cfg.max_num_groups, max_seq_len=model_cfg.max_seq_len,
                                 data_cfg=data_cfg, zarr_dataset=val_zarr, rasterizer=rasterizer)

        if model_cfg.train_idxs is not None:
            train_dataset = Subset(train_dataset, pd.read_csv(model_cfg.train_idxs)['idx'])
        if model_cfg.val_idxs is not None:
            val_dataset = Subset(val_dataset, pd.read_csv(model_cfg.val_idxs)['idx'])

        criterion = neg_multi_log_likelihood
        model = ModelTrajectory(model_cfg=model_cfg, data_config=data_cfg, modes=args.modes).to(device)


    elif args.data_type == "argo":

        if args.use_map and args.use_social:
            baseline_key = "map_social"
        elif args.use_map:
            baseline_key = "map"
        elif args.use_social:
            baseline_key = "social"
        else:
            baseline_key = "none"
        import numpy as np

        data_dict = baseline_utils.get_data(args, baseline_key)

        # # Get PyTorch Dataset
        train_dataset = SVGDataset(data_type="argo", model_args=model_cfg.model_args,
                                   max_num_groups=model_cfg.max_num_groups, max_seq_len=model_cfg.max_seq_len,
                                   data_dict=data_dict, args=args, mode="val")

#        val_dataset = SVGDataset(data_type="argo", model_args=model_cfg.model_args,
#                                 max_num_groups=model_cfg.max_num_groups, max_seq_len=model_cfg.max_seq_len,
#                                data_dict=data_dict, args=args, mode="val")
        criterion = get_ade
        old_loss_criterion = nn.MSELoss()
        #         model = ModelTrajectory(model_cfg=model_cfg,data_config=None, modes=args.modes, future_len=30, in_channels=3).to(device)
        #         model = MLPTransformer(model_config=model_cfg, data_config= None,
        #                                modes=args.modes,history_num = 40,future_len=60).to(device)

        model = Cheese(modes=args.modes).to(device)
        print(model)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad) , '#trainable params')
#         exit(0)
    train_dataloader = DataLoader(train_dataset, batch_size=model_cfg.train_batch_size, shuffle=True,
                                  num_workers=model_cfg.loader_num_workers, collate_fn=my_collate)
#    validat_dataloader = DataLoader(val_dataset, batch_size=model_cfg.val_batch_size, shuffle=False,
#                                    num_workers=model_cfg.loader_num_workers, collate_fn=my_collate)

    stats = Stats(num_steps=model_cfg.num_steps, num_epochs=model_cfg.num_epochs, steps_per_epoch=len(train_dataloader),
                  stats_to_print=model_cfg.stats_to_print)
    stats.stats['val'] = defaultdict(SmoothedValue)
    print(stats.stats.keys())
    train_vars = TrainVars()
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)
    print(f"#Parameters: {stats.num_parameters:,}")

    # Summary Writer
    #     current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    experiment_identifier = f"{model_name}"

    summary_writer = SummaryWriter(os.path.join(log_dir, experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", model_name, experiment_name)
    print(checkpoint_dir)

    # model_cfg.set_train_vars(train_vars, train_dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = model_cfg.make_optimizers(model)
    scheduler_lrs = model_cfg.make_schedulers(optimizers, epoch_size=len(train_dataloader))
    scheduler_warmups = model_cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in model_cfg.make_losses()]

    #     if not resume == " ":
    #         ckpt_exists = utils.load_ckpt_list(checkpoint_dir, model, None, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)

    #     if not resume == " " and ckpt_exists:
    #         print(f"Resuming model at epoch {stats.epoch+1}")
    #         stats.num_steps = model_cfg.num_epochs * len(train_dataloader)
    #     if True:
    #         # Run a single forward pass on the single-device model_and_dataset for initialization of some modules
    #         single_foward_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,
    #                                               num_workers=model_cfg.loader_num_workers , collate_fn=my_collate)
    #         data = next(iter(single_foward_dataloader))
    #         if data is not None:
    #             model_args, params_dict = [data['image'][arg].to(device) for arg in model_cfg.model_args], model_cfg.get_params(0, 0)
    #             entery = [*model_args,{},True]
    #             out = model(entery)
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    epoch_range = utils.infinite_range(stats.epoch) if model_cfg.num_epochs is None else range(stats.epoch,
                                                                                               cfg.num_epochs)
    print(epoch_range)
    timer.reset()
    print(timer.get_elapsed_time())
    for epoch in epoch_range:
        print(f"Epoch {epoch + 1}")
        for n_iter, data in enumerate(train_dataloader):
            if data is None:
                continue
            step = n_iter + epoch * len(train_dataloader)

            if model_cfg.num_steps is not None and step > model_cfg.num_steps:
                return

            model.train()
#             model_args = [data['image'][arg].to(device) for arg in model_cfg.model_args]
            params_dict, weights_dict = model_cfg.get_params(step, epoch), model_cfg.get_weights(step, epoch)

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(
                    zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, model_cfg.optimizer_starts), 1):
                optimizer.zero_grad()
                history_positions = data["history_positions"].to(device).type(torch.float32)
                scene = data["padded_cntr_lines"].to(device).type(torch.float32)
                scene_lens = data["available_cntr_size"].to(device).type(torch.float32)
                agents = data["normal_agents_history"].to(device).type(torch.float32)
                agents_lens = data["agents_num"].to(device).type(torch.float32)
                # history = data["history_positions"].to(device)
                #                 entery = [*model_args, params_dict, True]
                # entery = [[*model_args, {}, True], history]
                output, conf = model([history_positions, scene, scene_lens, agents, agents_lens])
                loss_dict = {}
                loss_dict['loss'] = criterion(data['target_positions'].to(device),
                                              output.reshape(data['target_positions'].shape), ).mean()
                loss_dict['old_loss'] = old_loss_criterion(data['target_positions'].to(device),
                                                           output.reshape(data['target_positions'].shape), ).mean()

                if step >= optimizer_start:
                    loss_dict['loss'].backward()
                    if model_cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip)

                    optimizer.step()
                    if scheduler_lr is not None:
                        scheduler_lr.step()
                    if scheduler_warmup is not None:
                        scheduler_warmup.step()

                stats.update_stats_to_print("train", loss_dict)
                stats.update("train", step, epoch, {
                    ("lr" if i == 1 else f"lr_{i}"): optimizer.param_groups[0]['lr'],
                    **loss_dict
                })

            if step % model_cfg.log_every == 0 and step != 0:
                print("log train")
                stats.update("train", step, epoch, {
                    **weights_dict,
                    "time": timer.get_elapsed_time()
                })
                print(stats.get_summary("train"))
                stats.write_tensorboard(summary_writer, "train")
                summary_writer.flush()

            if step % model_cfg.val_every == 0:
                print("log val")
                timer.reset()
                torch.save(model.state_dict(),
                           log_dir + "/" + experiment_identifier + "/" + "checkpoint" + "-" + str(step))
#                 validation(validat_dataloader, model, model_cfg, device, criterion, epoch, stats, summary_writer, timer,
#                            step,
#                            old_loss_criterion)


#             if step % model_cfg.ckpt_every == 0:
#                 utils.save_ckpt_list(checkpoint_dir, model, model_cfg, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)
#                 print("save checkpoint")


def validation(val_dataloader, model, model_cfg, device, criterion, epoch, stats, summary_writer, timer, train_step,
               old_loss_criterion):
    model.eval()
    for n_iter, data in enumerate(val_dataloader):
        if data is None:
            continue
        step = n_iter

        model_args = [data['image'][arg].to(device) for arg in model_cfg.model_args]
        params_dict, weights_dict = model_cfg.get_params(step, epoch), model_cfg.get_weights(step, epoch)

        if model_cfg.val_num_steps is not None and step > model_cfg.val_num_steps:
            stats.update("val", train_step, epoch, {
                **weights_dict,
                "time": timer.get_elapsed_time()
            })
            print(stats.get_summary("val"))
            stats.write_tensorboard(summary_writer, "val")
            summary_writer.flush()
            return
        history_positions = data["history_positions"].to(device).type(torch.float32)
        scene = data["padded_cntr_lines"].to(device).type(torch.float32)
        scene_lens = data["available_cntr_size"].to(device).type(torch.float32)
        agents = data["normal_agents_history"].to(device).type(torch.float32)
        agents_lens = data["agents_num"].to(device).type(torch.float32)
        # history = data["history_positions"].to(device)
        #                 entery = [*model_args, params_dict, True]
        # entery = [[*model_args, {}, True], history]
        output, conf = model([history_positions, scene, scene_lens, agents, agents_lens])
        loss_dict = {}
        loss_dict['loss'] = criterion(data['target_positions'].to(device),
                                      output.reshape(data['target_positions'].shape), ).mean()
        loss_dict['old_loss'] = old_loss_criterion(data['target_positions'].to(device),
                 output.reshape(data['target_positions'].shape), ).mean()

        stats.update_stats_to_print("val", loss_dict)

        stats.update("val", train_step, epoch, {
            **loss_dict
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--modes", type=int, default=3)
    # lyft
    parser.add_argument("--config-data", type=str, required=False)
    parser.add_argument("--val-idxs", type=str, default=None)
    parser.add_argument("--train-idxs", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=False)
    # argo
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )

    args = parser.parse_args()

    cfg = importlib.import_module(args.config_module).Config()
    model_name, experiment_name = args.config_module.split(".")[-2:]
    print(model_name, experiment_name)
    if args.val_idxs is not None:
        cfg.val_idxs = args.val_idxs
    if args.train_idxs is not None:
        cfg.train_idxs = args.train_idxs
    train(model_cfg=cfg, args=args,
          model_name="cheese-test-1", experiment_name=experiment_name,
          log_dir="/work/vita/argo_code/logs", debug=args.debug, resume=args.resume)
