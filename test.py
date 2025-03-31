from transformer_lens import HookedTransformer, HookedTransformerConfig

model_trained = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
model_cfg = model_trained.cfg

# randomize the weights of model
model = HookedTransformer(HookedTransformerConfig.from_dict(model_cfg.to_dict()))


for k, tensor in model.state_dict().items():
	if tensor.isnan().any():
		print(f"{k} has NaNs!")
		print(f"{tensor.isnan().sum().item() = }")
		# indicies of NaNs
		print(f"{tensor.device = }")
		print(tensor.isnan().nonzero(as_tuple=True))

import sys
print("python:", sys.version)
import torch
print("torch:", torch.__version__)
import transformers
print("transformers:", transformers.__version__)
import transformer_lens
from importlib.metadata import version
print("transformer_lens: ", version("transformer_lens"))