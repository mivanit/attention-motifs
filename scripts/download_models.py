from transformer_lens import HookedTransformer

if __name__ == "__main__":
	import sys
	
	if len(sys.argv) > 2:
		print("Usage: python download_models.py model_name_1,model_name_2,model_name_n")
		sys.exit(1)

	models: list[str] = sys.argv[1].split(",")

	for model_name in models:
		model = HookedTransformer.from_pretrained(model_name)
		del model