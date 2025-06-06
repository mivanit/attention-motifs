import torch

from attention_motifs.dataset.dataset import (
	APGenerationConfig,
	PromptDatasetConfig,
	CollectedAttentionPatternDataloader,
)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda", help="device to use")
	args = parser.parse_args()

	device: torch.device = torch.device(args.device)

	d = CollectedAttentionPatternDataloader.generate(
		config=APGenerationConfig(
			prompts_config=PromptDatasetConfig.from_source_path(
				"data/pile_5_val.jsonl",
				char_len_min=64,
				char_len_max=256,
			),
			model_names=[
				"pythia-14m",
				"gpt2-small",
				"meta-llama/Llama-3.2-1B",
			],
			token_len_min=16,
			prompt_token_len_tolerance=16,
		),
		max_batch_size=8,
		model_device=device,
	)

	d.save("data/activations/small_val", verbose=True)

	print("=" * 50)
	print(d.summary())
	print("=" * 50)
	print(d.summary_short())
	print("=" * 50)
