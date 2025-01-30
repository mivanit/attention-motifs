import functools
import json
from pathlib import Path
from typing import TypeVar
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Int, Float


# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
)
from muutils.misc import shorten_numerical_to_str
from zanj.torchutil import ConfiguredModel
from trnbl.loggers.base import TrainingLoggerBase

try:
	from trnbl.loggers.wandb import WandbLogger
except ImportError as e:
	warnings.warn(f"failed to import wandb, can't log figures: {e}")
	WandbLogger = None
from trnbl import TrainingManager
from zanj import ZANJ

# this project
from attention_motifs.dataset.dataset import (
	DataloaderMock,
	CollectedAttentionPatternDataloader,
)
from attention_motifs.dataset.util import AttentionPatternMetadata
from attention_motifs.train_util import contrastive_loss


def get_dataset(
	activations_path: Path,
	batch_size: int,
	n_batches: int,
	shuffle: bool = True,
	show: bool = True,
) -> tuple[DataloaderMock, dict, torch.Tensor, list]:
	"returns dataloader, dataset info, example patterns, example metadata"
	activations_path = Path(activations_path)

	# load dataset and print summary
	train_dataset: CollectedAttentionPatternDataloader = (
		CollectedAttentionPatternDataloader.read(activations_path)
	)
	summary_short_str: str = json.dumps(train_dataset.summary_short(), indent=2)
	print(summary_short_str)

	# turn dataset into dataloader
	train_loader: DataloaderMock = train_dataset.dataloader(
		batch_size=batch_size,
		shuffle=shuffle,
		max_batches=n_batches,
	)

	# print info
	print(
		f"loader: {len(train_loader)} batches, {len(train_loader.dataset)} samples"
	)

	# show example pattern
	x_mat, x_meta = next(
		iter(train_dataset.dataloader(10, shuffle=shuffle, max_batches=1))
	)
	x_mat.shape
	print(x_meta[0])
	plt.matshow(x_mat[0])
	if show:
		plt.show()
	else:
		plt.savefig("example_pattern.png")

	dataset_info: dict = dict(
		summary_short_str=summary_short_str,
		n_patterns=len(train_loader.dataset),
		batch_size=batch_size,
		n_batches=len(train_loader),
		activations_path=activations_path.as_posix(),
		summary_short=train_dataset.summary_short(),
		summary=train_dataset.summary(),
		dataset_str=str(train_dataset),
	)

	return train_loader, dataset_info, x_mat, x_meta


T_Config = TypeVar("T_Config", bound=SerializableDataclass)


def set_up_model(
	config: T_Config,
	model_cls: type[ConfiguredModel[T_Config]],
	device: torch.device,
) -> tuple[
	ConfiguredModel[T_Config],
	torch.optim.Optimizer,
	torch.optim.lr_scheduler._LRScheduler,
]:
	# set up model
	model: ConfiguredModel[T_Config] = model_cls(config)

	model_n_params: int = sum(p.numel() for p in model.parameters())
	print(
		f"model has {model_n_params} ({shorten_numerical_to_str(model_n_params)}) parameters"
	)
	model = model.to(device)
	model.train()

	# set up optimizer and lr scheduler
	optim, lrs = model.config.get_optim_and_lrs(model)

	return model, optim, lrs


def eval_plots(
	model: ConfiguredModel[T_Config],
	dataloader: DataloaderMock,
	device: torch.device,
	logger: TrainingLoggerBase,
	show: bool = True,
) -> None:
	model.eval()

	reconstruction_diffs: list[float] = list()
	mean_diffs: list[float] = list()
	mean_std: list[float] = list()
	latent_std: list[float] = list()

	with torch.no_grad():
		for batch_idx, (patterns, metadata) in enumerate(dataloader):
			
			x_recon, x_latent = model(patterns.to(device).to(torch.float32).unsqueeze(1))

			latent_std.append(x_latent.std(dim=0).mean().item())

			x_recon_mean = x_recon.mean(dim=0)[0].detach().cpu().numpy()
			
			for i in range(len(metadata)):
				x_recon_np = x_recon[i, 0].detach().cpu().numpy()
				fig, axs = plt.subplots(1, 4, figsize=(12, 4))

				axs[0].matshow(patterns[i])
				axs[0].axis("off")

				axs[1].matshow(x_recon_np)
				axs[1].axis("off")

				recon_diff = x_recon_np - patterns[i].numpy()
				axs[2].matshow(recon_diff, cmap="RdBu", vmin=-1, vmax=1)
				reconstruction_diffs.append(np.abs(recon_diff).mean())

				axs[2].axis("off")
				mean_diff: np.ndarray = x_recon_np - x_recon_mean
				axs[3].matshow(mean_diff, cmap="RdBu", vmin=-1, vmax=1)
				mean_diffs.append(np.abs(mean_diff).mean())
				mean_std.append(mean_diff.std())
				axs[3].axis("off")



				fig.suptitle(metadata[i])

				try:
					if (WandbLogger is not None) and isinstance(logger, WandbLogger):
						logger._run.log({f"eval/batch_{batch_idx}/pattern_{i}": fig})
				except Exception as e:
					warnings.warn(f"failed to log figure to wandb {i}: {e}")

					try:
						if show:
							plt.show()
						else:
							plt.savefig(f"eval_pattern_{i}.png")
					except Exception as e:
						warnings.warn(f"failed to save or show pattern {i}: {e}")

				finally:
					plt.close(fig)

	model.train()

	return {
		"val/reconstruction": sum(reconstruction_diffs) / len(reconstruction_diffs),
		"val/mean_diff": sum(mean_diffs) / len(mean_diffs),
		"val/latent_std": sum(latent_std) / len(latent_std),
	}


_TRAINING_MANAGER_KWARGS_DEFAULT: dict = dict(
	checkpoint_interval="1/2 run",
	model_save_path="{run_path}/checkpoints/model.checkpoint-{latest_checkpoint}.zanj",
	model_save_path_special="{run_path}/model.{alias}.zanj",
)

def train(
	logger: TrainingLoggerBase,
	device: torch.device,
	model: ConfiguredModel[T_Config],
	optimizer: torch.optim.Optimizer,
	lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
	train_loader: DataloaderMock,
	val_loader: DataloaderMock | None = None,
	training_manager_kwargs: dict|None = None,
) -> ConfiguredModel[T_Config]:
	
	model_config: T_Config = model.config

	if training_manager_kwargs is None:
		training_manager_kwargs = dict()
	
	training_manager_kwargs = {
		**_TRAINING_MANAGER_KWARGS_DEFAULT,
		**training_manager_kwargs,
	}

	evals = list()
	
	if val_loader is not None: 
		evals.append((
		"1/10 run",
		functools.partial(
			eval_plots,
			dataloader=val_loader,
			device=device,
			logger=logger,
			show=False,
		),
	))

	with TrainingManager(
		model=model,
		logger=logger,
		save_model=ZANJ().save,
		evals=evals,
		** training_manager_kwargs,
	) as tr:
		for epoch in tr.epoch_loop(range(model_config.num_epochs), use_tqdm=False):
			patterns: Float[torch.Tensor, "*batch n_ctx n_ctx"]
			metadata: list[AttentionPatternMetadata]
			for patterns, metadata in tr.batch_loop(train_loader, use_tqdm=True):
				this_batch_size: int = len(metadata)

				# move to device, convert type, add channel dim
				patterns = patterns.to(device).to(torch.float32).unsqueeze(1)

				# reset gradients
				optimizer.zero_grad()

				# forward pass
				patterns_recon, embeddings = model(patterns)

				# reconstruction loss
				recon_loss = F.mse_loss(patterns_recon, patterns)

				# compute "classes" for contrastive loss
				# classes is a tensor of the same shape as the batch, where each element is an integer
				classes: Int[torch.Tensor, " batch"] = (
					AttentionPatternMetadata.contrastive_classes(metadata).to(device)
				)

				# compute contrastive loss
				contrast_loss = contrastive_loss(embeddings, classes)

				# combined loss and backward pass
				total_loss = (
					model_config.recon_weight * recon_loss
					+ model_config.contrast_weight * contrast_loss
				)

				# backward pass
				total_loss.backward()
				optimizer.step()
				lr_scheduler.step(epoch)

				# log metrics
				metrics: dict[str, float] = {
					"train/loss": total_loss.item() / this_batch_size,
					"train/recon_loss": recon_loss.item() / this_batch_size,
					"train/contrast_loss": contrast_loss.item() / this_batch_size,
					"lr": lr_scheduler.get_last_lr()[0],
				}
				tr.batch_update(
					samples=len(metadata),
					**metrics,
				)

				# cleanup
				del (
					patterns,
					patterns_recon,
					embeddings,
					recon_loss,
					contrast_loss,
					total_loss,
				)

	return model
