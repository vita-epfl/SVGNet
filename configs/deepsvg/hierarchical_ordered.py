from .default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=2)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None
        self.train_ratio = 1.0

        self.max_num_groups = 10
        self.max_seq_len = 200
        self.max_total_len = 2000



        # Dataloader
        self.loader_num_workers = 20

        # Training
        self.num_epochs = 100

        # Optimization
        self.learning_rate = 1e-4
        self.train_batch_size = 32
        self.val_batch_size = 8

        self.val_every = 1500
        self.log_every = 100
        self.ckpt_every = 100

        self.val_num_steps =1500
        self.stats_to_print = {
            "train": ["lr", "time"],
            "val": ["time"]
        }

        self.val_idxs = None
        self.train_idxs = None
