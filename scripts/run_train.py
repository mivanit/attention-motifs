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

DEVICE: torch.device = torch.device("cuda:2")

# training data
ACTIVATIONS_PATH: Path = Path("data/activations/medium")
BATCH_SIZE: int = 32
N_TRAIN_BATCHES: int = 12750

# validation data
VAL_ACTIVATIONS_PATH: Path = Path("data/activations/small_val")
VAL_BATCH_SIZE: int = 8
N_VAL_BATCHES: int = 8

# model
MODEL_CONFIG: VitAEConfig = VitAEConfig(
	d_latent=64,
	num_epochs=10,
	d_model=128,
	mlp_dim=256,
	num_heads=8,
	encoder_depth=3,
	decoder_depth=3,
)


# get data
# ==================================================


TRAIN_DATASET: DataloaderMock
TRAIN_DATASET, DATASET_INFO, _, _ = get_dataset(
	activations_path=ACTIVATIONS_PATH,
	batch_size=BATCH_SIZE,
	n_batches=N_TRAIN_BATCHES,
	show=False,
	return_loader=False,
)

VAL_LOADER: DataloaderMock
VAL_LOADER, VAL_DATASET_INFO, _, _ = get_dataset(
	activations_path=VAL_ACTIVATIONS_PATH,
	batch_size=VAL_BATCH_SIZE,
	n_batches=N_VAL_BATCHES,
	shuffle=False,
	show=False,
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
	train_dataset=TRAIN_DATASET,
	batch_size=BATCH_SIZE,
	n_batches=N_TRAIN_BATCHES,
	val_loader=VAL_LOADER,
	eval_plots_interval="1/10 run",
)
