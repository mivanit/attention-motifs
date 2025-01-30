# imports and setup
# ==================================================
from pathlib import Path

import torch

# custom utils
from trnbl.loggers.wandb import WandbLogger


from attention_motifs.vit_ae import VitAEConfig, VitAE
from attention_motifs.train import get_dataset, set_up_model, train
from attention_motifs.dataset.dataset import DataloaderMock


# Configuration
# ==================================================

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training data
ACTIVATIONS_PATH: Path = Path("../data/activations/medium")
BATCH_SIZE: int = 10
N_TRAIN_BATCHES: int = 50

# validation data
VAL_ACTIVATIONS_PATH: Path = Path("../data/activations/small_val")
VAL_BATCH_SIZE: int = 8
N_VAL_BATCHES: int = 8

# model
MODEL_CONFIG: VitAEConfig = VitAEConfig(
	d_latent=128,
	num_epochs=1,
)


# get data
# ==================================================


TRAIN_LOADER: DataloaderMock
TRAIN_LOADER, DATASET_INFO, _, _ = get_dataset(
	activations_path=ACTIVATIONS_PATH,
	batch_size=BATCH_SIZE,
	n_batches=N_TRAIN_BATCHES,
)

VAL_LOADER: DataloaderMock
VAL_LOADER, VAL_DATASET_INFO, _, _ = get_dataset(
	activations_path=VAL_ACTIVATIONS_PATH,
	batch_size=VAL_BATCH_SIZE,
	n_batches=N_VAL_BATCHES,
	shuffle=False,
)


# set up model and train
# ==================================================

MODEL, OPTIMIZER, LR_SCHEDULER = set_up_model(
	config=MODEL_CONFIG,
	model_cls=VitAE,
	device=DEVICE,
)


# init logger
LOGGER: WandbLogger = WandbLogger.create(
	config=dict(
		model_config=MODEL.zanj_model_config.serialize(),
		dataset_info=DATASET_INFO,
		val_dataset_info=VAL_DATASET_INFO,
		model_str=str(MODEL),
		device=str(DEVICE),
	),
	project="vit-ae",
	# name="vit-ae" + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"),
)

MODEL = train(
	logger=LOGGER,
	device=DEVICE,
	model=MODEL,
	optimizer=OPTIMIZER,
	lr_scheduler=LR_SCHEDULER,
	train_loader=TRAIN_LOADER,
	val_loader=VAL_LOADER,
)
