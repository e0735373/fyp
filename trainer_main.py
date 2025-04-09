import PIL.Image
import pddlgym
import imageio
import PIL
from transformeragent import TransformerAgent, DiffusionAgent, D3PMAgent
from pddlgym_planners.fd import FD
from pddlgym_planners.ff import FF
from main.dataset.mysokoban_lowdim_dataset import generate_mysokoban_dataset, MysokobanLowdimDataset, ActionMySokoban
from main.dataset.myhanoi_lowdim_dataset import generate_myhanoi_dataset, MyhanoiLowdimDataset, ActionMyHanoi
from main.dataset.mypath_lowdim_dataset import generate_mypath_dataset, MypathLowdimDataset, ActionMyPath
from main.dataset.mysudoku_lowdim_dataset import ActionMySudoku, generate_mysudoku_dataset, MysudokuLowdimDataset
from torch.utils.data import DataLoader
from worker import TrainTransformerWorkspace
from config import MysokobanConfig, MyhanoiConfig, MypathConfig, MysudokuConfig, MysokobanConfigWithLossReweighting, MyhanoiConfigWithLossReweighting, MysudokuConfigMaskToken, MypathConfigWithLossReweighting
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("--first_k_moves", type=int, default=None)
argparser.add_argument("--output_dir", type=str, required=True)
argparser.add_argument("--checkpoint_tag", type=str, default="latest")
argparser.add_argument("--model_type", type=str, required=True, choices=["diffusion", "d3pm", "transformer", "d3pm-reweighting"])
argparser.add_argument("--env_type", type=str, required=True, choices=["sokoban", "hanoi", "path", "sudoku"])
argparser.add_argument("--has_mask_token", action="store_true")
argparser.add_argument("--n_episodes", type=int, default=100)
argparser.add_argument("--do_train", action="store_true")
argparser.add_argument("--do_eval_unseen", action="store_true")

args = argparser.parse_args()

output_dir = args.output_dir
checkpoint_tag = args.checkpoint_tag

if args.env_type == "sokoban":
    zarr_path_train = "./storage/mysokoban-n8-b2-train"
    zarr_path_test = "./storage/mysokoban-n8-b2-test"
    pddl_env_name = "PDDLEnvMysokoban-n8-b2"
    if args.has_mask_token:
        raise ValueError("Mask token is not supported for sokoban")
    else:
        config_type = MysokobanConfig
    action_to_int_cls = ActionMySokoban
    dataset_cls = MysokobanLowdimDataset
elif args.env_type == "hanoi":
    zarr_path_train = "./storage/myhanoi-n6-p3-train"
    zarr_path_test = "./storage/myhanoi-n6-p3-test"
    pddl_env_name = "PDDLEnvMyhanoi"
    if args.has_mask_token:
        raise ValueError("Mask token is not supported for hanoi")
    else:
        config_type = MyhanoiConfig
    action_to_int_cls = ActionMyHanoi
    dataset_cls = MyhanoiLowdimDataset
elif args.env_type == "path":
    zarr_path_train = "./storage/mypath-n10-train"
    zarr_path_test = "./storage/mypath-n10-test"
    pddl_env_name = "PDDLEnvMypath"
    if args.has_mask_token:
        raise ValueError("Mask token is not supported for path")
    else:
        config_type = MypathConfig
    action_to_int_cls = ActionMyPath
    dataset_cls = MypathLowdimDataset
elif args.env_type == "sudoku":
    zarr_path_train = "./storage/mysudoku-train"
    zarr_path_test = "./storage/mysudoku-test"
    pddl_env_name = "PDDLEnvMysudoku"
    if args.has_mask_token:
        config_type = MysudokuConfigMaskToken
    else:
        config_type = MysudokuConfig
    action_to_int_cls = ActionMySudoku
    dataset_cls = MysudokuLowdimDataset
else:
    raise ValueError(f"Unknown env type {args.env_type}")


if args.model_type == "diffusion":
    model_type = DiffusionAgent
elif args.model_type == "d3pm":
    model_type = D3PMAgent
elif args.model_type == "transformer":
    model_type = TransformerAgent
elif args.model_type == "d3pm-reweighting":
    assert args.env_type == "sokoban" or args.env_type == "hanoi" or args.env_type == "path"
    model_type = D3PMAgent
    if args.env_type == "sokoban":
        config_type = MysokobanConfigWithLossReweighting
    elif args.env_type == "hanoi":
        config_type = MyhanoiConfigWithLossReweighting
    elif args.env_type == "path":
        config_type = MypathConfigWithLossReweighting
    else:
        raise ValueError(f"Unknown env type {args.env_type} for d3pm-reweighting")
else:
    raise ValueError(f"Unknown model type {args.model_type}")


do_train = args.do_train
do_eval_unseen = args.do_eval_unseen

workspace = TrainTransformerWorkspace(
    output_dir=output_dir,
    model_type=model_type,
    config_type=config_type,
)

if do_train:
    workspace.train(zarr_path=zarr_path_train, dataset_cls=dataset_cls)
elif do_eval_unseen:
    plans = workspace.eval(ckpt_tag=checkpoint_tag,
                           n_episodes=args.n_episodes,
                           first_k_moves=args.first_k_moves, action_to_int_cls=action_to_int_cls, env_name=pddl_env_name + "Test-v0")

    success = 0
    plan_lengths = []
    for p in plans:
        if p['success']:
            success += 1
            plan_lengths.append(len(p['plan']))

    print(f"Success rate: {success / len(plans)}")
    print(f"Average succesful plan length: {np.mean(plan_lengths)}, std: {np.std(plan_lengths)}")
    print(f"Plans:")
    print(plans)
else:
    workspace.eval_accuracy(ckpt_tag=checkpoint_tag, zarr_path=zarr_path_test, dataset_cls=dataset_cls)
