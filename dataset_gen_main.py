import PIL.Image
import pddlgym
import imageio
import PIL
import psutil
import csv
import copy
from pddlgym_planners.fd import FD
from pddlgym_planners.ff import FF
from main.dataset.mysokoban_lowdim_dataset import generate_mysokoban_dataset, MysokobanLowdimDataset
from main.dataset.myhanoi_lowdim_dataset import generate_myhanoi_dataset, MyhanoiLowdimDataset
from main.dataset.mypath_lowdim_dataset import generate_mypath_dataset, MypathLowdimDataset
from main.dataset.mysudoku_lowdim_dataset import generate_mysudoku_dataset, MysudokuLowdimDataset
from main.common.replay_buffer import ReplayBuffer

zarr_path = "./storage/myhanoi-n6-p3-test"
pddl_env_name = "PDDLEnvMyhanoiTest-v0"
n_episodes = 100

generate_myhanoi_dataset(
  zarr_path,
  pddl_env_name,
  n_episodes,
  n_workers=8
)

dataset = MyhanoiLowdimDataset(zarr_path, horizon=5, pad_before=4, pad_after=4)

print("len(dataset)", len(dataset))
print("dataset[0]", dataset[0])
