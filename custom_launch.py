import argparse
import contextlib
import logging
import os
import sys
import pickle
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.callbacks import (
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
    ProgressCallback,
)
from threestudio.utils.config import ExperimentConfig, load_config
from threestudio.utils.misc import get_rank
from threestudio.utils.typing import Optional
from ldm.models.diffusion import options

class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True

# Replace main() with this new version
def main(
    config_path: str,
    gpu: str = "0",
    verbose: bool = False,
    extras: Optional[list] = None,
):
    if extras is None:
        extras = []

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]
    devices = -1
    if len(env_gpus) > 0:
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    torch.set_float32_matmul_precision('medium')
    options.LDM_DISTILLATION_ONLY = True
    logger = logging.getLogger("pytorch_lightning")
    if verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())

    cfg: ExperimentConfig
    cfg = load_config(config_path, cli_args=extras, n_gpus=n_gpus)
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    submodule_path = "ZeroNVS/zeronvs_diffusion/zero123"
    assert os.path.exists(submodule_path)
    sys.path.insert(0, submodule_path)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint),
        LearningRateMonitor(logging_interval="step"),
        ConfigSnapshotCallback(
            config_path,
            cfg,
            os.path.join(cfg.trial_dir, "configs"),
            use_version=False,
        ),
        CustomProgressBar(refresh_rate=1),
    ]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    rank_zero_only(
        lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
    )()
    loggers += [
        TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
        CSVLogger(cfg.trial_dir, name="csv_logs"),
    ] + system.get_loggers()
    rank_zero_only(
        lambda: write_to_text(
            os.path.join(cfg.trial_dir, "cmd.txt"),
            [f"main(config_path={config_path}, gpu={gpu})"],
        )
    )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )
    print(system)

    system.geometry = torch.compile(system.geometry)
    trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
    # trainer.test(system, datamodule=dm)

# Keep CLI compatibility
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--verbose", action="store_true")
    args, extras = parser.parse_known_args()
    main(args.config, args.gpu, args.verbose, extras)
