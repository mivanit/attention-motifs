from transformer_lens import HookedTransformer

for model_name in "pythia-14m,pythia-1b,tiny-stories-1M,gpt2-small,gpt2-medium,meta-llama/Llama-3.2-1B,gemma-2b".split(
	","
):
	model = HookedTransformer.from_pretrained(model_name)
	del model