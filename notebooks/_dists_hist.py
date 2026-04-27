# %%
import numpy as np
import matplotlib.pyplot as plt
# %%


x = np.load("distances.npy")
x_f32 = np.load("distances_f32.npy")

# np.save("distances_f16.npy", x.astype(np.float16))

x_f16 = np.load("distances_f16.npy")

# #%%
np.max(np.abs(x - x_f32))
# #%%
# np.max(np.abs(x - x_f16))

# #%%
# np.max(np.abs(x_f16 - x_f32))

# %%
x.shape

# %%

plt.hist(x.flatten(), bins=200)
plt.show()

plt.hist(x_f16.flatten(), bins=200)
plt.show()

# %%

import json
from pathlib import Path

labels: list[str] = json.loads(Path("dists_meta.json").read_text())["cls_values"]

labels_mdl: list[str] = [x.split(":")[0] for x in labels]

n_heads: int = len(labels_mdl)

assert n_heads == x.shape[0], (
	"Number of heads does not match the shape of distances array."
)
assert n_heads == x.shape[1], (
	"Number of heads does not match the shape of distances array."
)

all_models: set[str] = set(labels_mdl)

# all_models
# {'Llama-3-2-1B',
#  'gemma-2-2b',
#  'gemma-2b',
#  'gpt2-medium',
#  'gpt2-small',
#  'pythia-1b'}
colors: dict[str, str] = {
	"Llama-3-2-1B": "brown",
	"gemma-2-2b": "purple",
	"gemma-2b": "blue",
	"gpt2-medium": "red",
	"gpt2-small": "orange",
	"pythia-1b": "green",
}

# %%

vmin = np.min(x)
vmax = np.max(x)

_ = vmin, vmax

# %%


# bins = np.linspace(0.0, vmax, 100)
bins = np.linspace(0.0, 5.0, 100)

all_hists = [np.histogram(x[i], bins=bins)[0] for i in range(n_heads)]


for i in range(n_heads):
	plt.plot(
		bins[:-1],
		all_hists[i],
		alpha=0.01,
		color=colors[labels_mdl[i]],
	)

plt.show()
