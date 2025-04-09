# Diffusion for Planning in Discrete Spaces

This repository contains the official codebase for the project titled **Diffusion for Planning in Discrete Spaces**, as part of the final year project (FYP) at the Department of Computer Science, School of Computing, National University of Singapore in the Academic Year 2024/2025.

##  Usage

There are several custom or modified environments used in this project. You may append `Test` to the environment name to get the test set version of the environment. Some of the environments are as follows:
- `PDDLEnvMysudoku`: Sudoku environment.
- `PDDLEnvMypath`: Simple Path environment with $10$ vertices and $9$ edges forming a tree graph.
- `PDDLEnvMyhanoi`: Tower of Hanoi environment with $6$ disks and $3$ pegs, with a specific goal state.
- `PDDLEnvMysokoban`: Sokoban environment with an $8 \times 8$ grid and $1$ box.
- `PDDLEnvMysokoban-n8-b2`: Sokoban environment with an $8 \times 8$ grid and $2$ boxes.

To use `PDDLEnvMysudoku` and [`altenvsudoku.py`](altenvsudoku.py), you need to download the `.csv` dataset from https://www.kaggle.com/datasets/bryanpark/sudoku and place it in the root folder as `sudoku.csv`.

In the folders corresponding to the environments (in `pddlgym/pddlgym/pddl/`), you will `gen.sh` files that contain the commands to generate the `.pddl` files. You will need to run these generation scripts to generate the `.pddl` files before being able to use the environments.

Please see [`dataset_gen_main.py`](dataset_gen_main.py) for an example script to generate the dataset.

After the dataset is generated, you can run the training script [`trainer_main.py`](trainer_main.py) to train the model. This script also contains code for evaluating the environment accuracy and the action accuracy of the test set.

## Acknowledgement

This codebase is built upon and adapted from the following repositories:
- [BioNeMo Framework](https://github.com/NVIDIA/bionemo-framework)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [PDDL Generators](https://github.com/AI-Planning/pddl-generators)
- [PDDLGym](https://github.com/tomsilver/pddlgym)
- [Planner Interface for PDDLGym](https://github.com/ronuchit/pddlgym_planners)

...and many others.

Thank you to the authors of these repositories for their contributions. Their work has been instrumental in the development of this project.
