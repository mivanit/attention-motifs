import matplotlib.pyplot as plt

from attention_motifs.dataset.dataset import (
	APGenerationConfig,
	PromptDatasetConfig,
	CollectedAttentionPatternDataloader,
)

d = CollectedAttentionPatternDataloader.generate(
	config=APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path("data/pile_5.jsonl"),
		model_names=[
			# "meta-llama/Llama-3.2-1B",
			"gpt2-small",
			# "pythia-14m",
		],
	),
	max_batch_size=8,
)

d.save("data/activations/pile_1k", verbose=True)


for x in d.batches(2):
	print(x[0].shape)
	print(x[1])
	plt.matshow(x[0][0].cpu().numpy())
	plt.show()
	plt.matshow(x[0][1].cpu().numpy())
	plt.show()
	break
