from pddlgym.rendering.mysokoban import NUM_OBJECTS as Mysokoban_NUM_OBJECTS
from altenvsudoku import AltEnvSudoku
from altenvsokoban import AltEnvSokoban

class MysokobanConfig:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = False
    HAS_NO_PADDING = 0
    HAS_GOAL_CONDITION = 0
    EVAL_UNSEEN_STEP_LIMIT = 100
    OVERRIDE_PDDLGYM_ENVIRONMENT = True
    OVERRIDE_PDDLGYM_ENVIRONMENT_CLASS = AltEnvSokoban

    HORIZON = 8 # number of action steps to predict
    N_OBS_STEPS = 2 # number of past observations to condition on
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    ACTION_DIM = 4
    PIXEL_DIM = Mysokoban_NUM_OBJECTS
    IMAGE_SIZE = 8*8
    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 1

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MysokobanConfigWithLossReweighting:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = True
    HAS_NO_PADDING = 0
    HAS_GOAL_CONDITION = 0
    EVAL_UNSEEN_STEP_LIMIT = 100
    OVERRIDE_PDDLGYM_ENVIRONMENT = True
    OVERRIDE_PDDLGYM_ENVIRONMENT_CLASS = AltEnvSokoban

    HORIZON = 8 # number of action steps to predict
    N_OBS_STEPS = 2 # number of past observations to condition on
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    ACTION_DIM = 4
    PIXEL_DIM = Mysokoban_NUM_OBJECTS
    IMAGE_SIZE = 8*8
    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 1

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MyhanoiConfig:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = False
    HAS_NO_PADDING = 0
    HAS_GOAL_CONDITION = 1
    EVAL_UNSEEN_STEP_LIMIT = 100
    OVERRIDE_PDDLGYM_ENVIRONMENT = False

    HORIZON = 8 # number of action steps to predict
    N_OBS_STEPS = 2 # number of past observations to condition on (we +1 from goal conditioning)
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    N_PEGS = 3
    N_DISCS = 6

    ACTION_DIM = N_PEGS * N_PEGS # (peg * peg), 3 pegs
    PIXEL_DIM = N_PEGS
    IMAGE_SIZE = N_DISCS

    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 5

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MyhanoiConfigWithLossReweighting:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = True
    HAS_NO_PADDING = 0
    HAS_GOAL_CONDITION = 1
    EVAL_UNSEEN_STEP_LIMIT = 100
    OVERRIDE_PDDLGYM_ENVIRONMENT = False

    HORIZON = 8 # number of action steps to predict
    N_OBS_STEPS = 2 # number of past observations to condition on (we +1 from goal conditioning)
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    N_PEGS = 3
    N_DISCS = 6

    ACTION_DIM = N_PEGS * N_PEGS # (peg * peg), 3 pegs
    PIXEL_DIM = N_PEGS
    IMAGE_SIZE = N_DISCS

    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 5

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MypathConfig:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = False
    HAS_NO_PADDING = 1
    HAS_GOAL_CONDITION = 0
    EVAL_UNSEEN_STEP_LIMIT = 20
    OVERRIDE_PDDLGYM_ENVIRONMENT = False

    HORIZON = 10 # number of action steps to predict
    N_OBS_STEPS = 1 # number of past observations to condition on (we +1 from goal conditioning)
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    N_NODES = 10

    ACTION_DIM = N_NODES
    PIXEL_DIM = 5
    IMAGE_SIZE = N_NODES * N_NODES

    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 5

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MypathConfigWithLossReweighting:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = True
    HAS_NO_PADDING = 1
    HAS_GOAL_CONDITION = 0
    EVAL_UNSEEN_STEP_LIMIT = 20
    OVERRIDE_PDDLGYM_ENVIRONMENT = False

    HORIZON = 10 # number of action steps to predict
    N_OBS_STEPS = 1 # number of past observations to condition on (we +1 from goal conditioning)
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    N_NODES = 10

    ACTION_DIM = N_NODES
    PIXEL_DIM = 5
    IMAGE_SIZE = N_NODES * N_NODES

    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 5

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MysudokuConfig:
    HAS_MASK_TOKEN = 0
    LOSS_REWEIGHTING = False
    HAS_NO_PADDING = 1
    HAS_GOAL_CONDITION = 0
    EVAL_UNSEEN_STEP_LIMIT = 81
    OVERRIDE_PDDLGYM_ENVIRONMENT = True
    OVERRIDE_PDDLGYM_ENVIRONMENT_CLASS = AltEnvSudoku

    N_NUMBERS = 9

    HORIZON = N_NUMBERS * N_NUMBERS # number of action steps to predict
    N_OBS_STEPS = 1 # number of past observations to condition on (we +1 from goal conditioning)
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    ACTION_DIM = N_NUMBERS
    PIXEL_DIM = 10
    IMAGE_SIZE = N_NUMBERS * N_NUMBERS

    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 5

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

class MysudokuConfigMaskToken:
    HAS_MASK_TOKEN = 1
    LOSS_REWEIGHTING = False
    HAS_NO_PADDING = 1
    HAS_GOAL_CONDITION = 0
    EVAL_UNSEEN_STEP_LIMIT = 81
    OVERRIDE_PDDLGYM_ENVIRONMENT = True
    OVERRIDE_PDDLGYM_ENVIRONMENT_CLASS = AltEnvSudoku

    N_NUMBERS = 9

    HORIZON = N_NUMBERS * N_NUMBERS # number of action steps to predict
    N_OBS_STEPS = 1 # number of past observations to condition on (we +1 from goal conditioning)
    N_ACTION_STEPS = HORIZON # number of next actions to predict, usually same as HORIZON in our case

    ACTION_DIM = N_NUMBERS
    PIXEL_DIM = 10
    IMAGE_SIZE = N_NUMBERS * N_NUMBERS

    PIXEL_OUT_DIM = 32
    COND_DIM = PIXEL_OUT_DIM

    N_LAYER = 8
    N_HEAD = 4
    N_EMB = 256
    P_DROP_EMB = 0.0
    P_DROP_ATTN = 0.01

    TIME_AS_COND = True
    OBS_AS_COND = True
    N_COND_LAYERS = 4

    OPT_LEARNING_RATE = 1.0e-4
    OPT_WEIGHT_DECAY = 1.0e-1
    OPT_BETAS = [0.9, 0.95]

    DATALOADER_BATCH_SIZE = 64
    DATALOADER_SHUFFLE = True

    LR_SCHEDULER = 'cosine'
    LR_WARMUP_STEPS = 500

    NUM_EPOCHS = 300
    TRAINING_CHECKPOINT_EVERY = 5

    NOISESCHEDULER_NUM_TRAIN_TIMESTEPS = 100
    NOISESCHEDULER_BETA_START = 0.0001
    NOISESCHEDULER_BETA_END = 0.02
    NOISESCHEDULER_BETA_SCHEDULE = 'squaredcos_cap_v2'
    NOISESCHEDULER_VARIANCE_TYPE = 'fixed_small'
    NOISESCHEDULER_CLIP_SAMPLE = True
    NOISESCHEDULER_PREDICTION_TYPE = 'epsilon'

