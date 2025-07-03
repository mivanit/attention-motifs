 # Inline TODOs


# BUG

## [`attn_embed/math/matrix_powers.py`](/attn_embed/math/matrix_powers.py)

- breaks with integer matrices???  
  local link: [`/attn_embed/math/matrix_powers.py#80`](/attn_embed/math/matrix_powers.py#80) 
  | view on GitHub: [attn_embed/math/matrix_powers.py#L80](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/math/matrix_powers.py#L80)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=breaks%20with%20integer%20matrices%3F%3F%3F&body=%23%20source%0A%0A%5B%60attn_embed%2Fmath%2Fmatrix_powers.py%23L80%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Fmath%2Fmatrix_powers.py%23L80%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20BUG%3A%20breaks%20with%20integer%20matrices%3F%3F%3F%0Adef%20matrix_powers_torch%28%0A%09A%3A%20Float%5Btorch.Tensor%2C%20%22n%20n%22%5D%2C%0A%60%60%60&labels=bug)

  ```python
# BUG: breaks with integer matrices???
def matrix_powers_torch(
	A: Float[torch.Tensor, "n n"],
  ```





# HACK

## [`attn_embed/features/head_analysis.py`](/attn_embed/features/head_analysis.py)

- add metadata  
  local link: [`/attn_embed/features/head_analysis.py#195`](/attn_embed/features/head_analysis.py#195) 
  | view on GitHub: [attn_embed/features/head_analysis.py#L195](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/features/head_analysis.py#L195)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=add%20metadata&body=%23%20source%0A%0A%5B%60attn_embed%2Ffeatures%2Fhead_analysis.py%23L195%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Ffeatures%2Fhead_analysis.py%23L195%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09df%20%3D%20df.with_columns%28pl.Series%28f%22embed.%7Bi%7D%22%2C%20embedding%5B%3A%2C%20i%5D%29%29%0A%0A%09%23%20HACK%3A%20add%20metadata%0A%09df._embed_meta%20%3D%20dict%28%0A%09%09embedding_method%3Dembedding_method%2C%0A%60%60%60&labels=HACK)

  ```python
df = df.with_columns(pl.Series(f"embed.{i}", embedding[:, i]))

	# HACK: add metadata
	df._embed_meta = dict(
		embedding_method=embedding_method,
  ```





# TODO

## [`.old/autoencoder/ae.py`](/.old/autoencoder/ae.py)

- add pos embeds?  
  local link: [`/.old/autoencoder/ae.py#185`](/.old/autoencoder/ae.py#185) 
  | view on GitHub: [.old/autoencoder/ae.py#L185](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/ae.py#L185)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=add%20pos%20embeds%3F&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fae.py%23L185%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fae.py%23L185%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09h%3A%20Float%5BTensor%2C%20%22batch%20channels%20n_ctx%20n_ctx%22%5D%20%3D%20self.conv%28x%29%0A%09%09%23%20apply%20linear%20layers%20to%20each%20pixel%0A%09%09%23%20TODO%3A%20add%20pos%20embeds%3F%0A%09%09h_reshape%20%3D%20h.flatten%282%29.reshape%28h.size%280%29%2C%20-1%2C%20h.size%281%29%29%0A%09%09h%20%3D%20self.linear_prepool%28h_reshape%29%0A%60%60%60&labels=enhancement)

  ```python
h: Float[Tensor, "batch channels n_ctx n_ctx"] = self.conv(x)
		# apply linear layers to each pixel
		# TODO: add pos embeds?
		h_reshape = h.flatten(2).reshape(h.size(0), -1, h.size(1))
		h = self.linear_prepool(h_reshape)
  ```


- first conv doesn't correctly read last preunpool layer size  
  local link: [`/.old/autoencoder/ae.py#245`](/.old/autoencoder/ae.py#245) 
  | view on GitHub: [.old/autoencoder/ae.py#L245](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/ae.py#L245)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=first%20conv%20doesn%27t%20correctly%20read%20last%20preunpool%20layer%20size&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fae.py%23L245%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fae.py%23L245%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09self.linear_preunpool%3A%20nn.Module%20%3D%20nn.Sequential%28%2Apreunpool_layers%29%0A%0A%09%09%23%20TODO%3A%20first%20conv%20doesn%27t%20correctly%20read%20last%20preunpool%20layer%20size%0A%0A%09%09%23%20Transposed%20convolution%20layers%0A%60%60%60&labels=enhancement)

  ```python
self.linear_preunpool: nn.Module = nn.Sequential(*preunpool_layers)

		# TODO: first conv doesn't correctly read last preunpool layer size

		# Transposed convolution layers
  ```




## [`.old/autoencoder/dataset/dataset.py`](/.old/autoencoder/dataset/dataset.py)

- switch to `_n{n_ctx}` for the dataset name  
  local link: [`/.old/autoencoder/dataset/dataset.py#360`](/.old/autoencoder/dataset/dataset.py#360) 
  | view on GitHub: [.old/autoencoder/dataset/dataset.py#L360](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/dataset/dataset.py#L360)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=switch%20to%20%60_n%7Bn_ctx%7D%60%20for%20the%20dataset%20name&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fdataset%2Fdataset.py%23L360%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fdataset%2Fdataset.py%23L360%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09disable%3Dnot%20verbose%2C%0A%09%09%29%3A%0A%09%09%09%23%20TODO%3A%20switch%20to%20%60_n%7Bn_ctx%7D%60%20for%20the%20dataset%20name%0A%09%09%09z.save%28dataset%2C%20path%20%2F%20f%22dataset_%7Bn_ctx%7D.zanj%22%29%0A%60%60%60&labels=enhancement)

  ```python
disable=not verbose,
		):
			# TODO: switch to `_n{n_ctx}` for the dataset name
			z.save(dataset, path / f"dataset_{n_ctx}.zanj")
  ```


- switch to `_n{n_ctx}` for the dataset name  
  local link: [`/.old/autoencoder/dataset/dataset.py#398`](/.old/autoencoder/dataset/dataset.py#398) 
  | view on GitHub: [.old/autoencoder/dataset/dataset.py#L398](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/dataset/dataset.py#L398)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=switch%20to%20%60_n%7Bn_ctx%7D%60%20for%20the%20dataset%20name&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fdataset%2Fdataset.py%23L398%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fdataset%2Fdataset.py%23L398%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09for%20d_m%20in%20dataset_meta%3A%0A%09%09%09n_ctx%3A%20int%20%3D%20d_m%5B%22n_ctx%22%5D%0A%09%09%09%23%20TODO%3A%20switch%20to%20%60_n%7Bn_ctx%7D%60%20for%20the%20dataset%20name%0A%09%09%09ds_path%3A%20Path%20%3D%20path%20%2F%20f%22dataset_%7Bn_ctx%7D.zanj%22%0A%09%09%09ds%3A%20AttentionPatternDataset%20%3D%20z.read%28ds_path%29%0A%60%60%60&labels=enhancement)

  ```python
for d_m in dataset_meta:
			n_ctx: int = d_m["n_ctx"]
			# TODO: switch to `_n{n_ctx}` for the dataset name
			ds_path: Path = path / f"dataset_{n_ctx}.zanj"
			ds: AttentionPatternDataset = z.read(ds_path)
  ```


- save them incrementally. not enough dedidated wam  
  local link: [`/.old/autoencoder/dataset/dataset.py#531`](/.old/autoencoder/dataset/dataset.py#531) 
  | view on GitHub: [.old/autoencoder/dataset/dataset.py#L531](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/dataset/dataset.py#L531)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=save%20them%20incrementally.%20not%20enough%20dedidated%20wam&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fdataset%2Fdataset.py%23L531%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fdataset%2Fdataset.py%23L531%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%23%20create%20datasets%20from%20binned%20data%0A%09%09%23%20TODO%3A%20save%20them%20incrementally.%20not%20enough%20dedidated%20wam%0A%09%09with%20SpinnerContext%28message%3D%22assembling%20datasets%22%29%3A%0A%09%09%09datasets%3A%20dict%5Bint%2C%20AttentionPatternDataset%5D%20%3D%20%7B%0A%60%60%60&labels=enhancement)

  ```python
# create datasets from binned data
		# TODO: save them incrementally. not enough dedidated wam
		with SpinnerContext(message="assembling datasets"):
			datasets: dict[int, AttentionPatternDataset] = {
  ```




## [`.old/autoencoder/dataset/util.py`](/.old/autoencoder/dataset/util.py)

- why is it warning us here? look into that error, ignoring for now.  
  local link: [`/.old/autoencoder/dataset/util.py#101`](/.old/autoencoder/dataset/util.py#101) 
  | view on GitHub: [.old/autoencoder/dataset/util.py#L101](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/dataset/util.py#L101)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=why%20is%20it%20warning%20us%20here%3F%20look%20into%20that%20error%2C%20ignoring%20for%20now.&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fdataset%2Futil.py%23L101%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fdataset%2Futil.py%23L101%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20why%20is%20it%20warning%20us%20here%3F%20look%20into%20that%20error%2C%20ignoring%20for%20now.%0A%40serializable_dataclass%28on_typecheck_mismatch%3DErrorMode.IGNORE%29%0Aclass%20AttentionPatternMetadataArray%28SerializableDataclass%29%3A%0A%60%60%60&labels=enhancement)

  ```python
# TODO: why is it warning us here? look into that error, ignoring for now.
@serializable_dataclass(on_typecheck_mismatch=ErrorMode.IGNORE)
class AttentionPatternMetadataArray(SerializableDataclass):
  ```


- wtf? why are these not being deserialized properly?  
  local link: [`/.old/autoencoder/dataset/util.py#105`](/.old/autoencoder/dataset/util.py#105) 
  | view on GitHub: [.old/autoencoder/dataset/util.py#L105](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/dataset/util.py#L105)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=wtf%3F%20why%20are%20these%20not%20being%20deserialized%20properly%3F&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fdataset%2Futil.py%23L105%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fdataset%2Futil.py%23L105%29%0A%0A%23%20context%0A%60%60%60python%0Aclass%20AttentionPatternMetadataArray%28SerializableDataclass%29%3A%0A%09model_names_map%3A%20list%5Bstr%5D%0A%09%23%20TODO%3A%20wtf%3F%20why%20are%20these%20not%20being%20deserialized%20properly%3F%0A%09data%3A%20UInt16%5Bnp.ndarray%2C%20%22%20model_name%2Fidx_layer%2Fidx_head%2Fn_ctx%3D4%20n_patterns%22%5D%20%3D%20%28%0A%09%09serializable_field%28%0A%60%60%60&labels=enhancement)

  ```python
class AttentionPatternMetadataArray(SerializableDataclass):
	model_names_map: list[str]
	# TODO: wtf? why are these not being deserialized properly?
	data: UInt16[np.ndarray, " model_name/idx_layer/idx_head/n_ctx=4 n_patterns"] = (
		serializable_field(
  ```


- create AttentionPatternMetadataArray here instead, then concatenate them all at the end  
  local link: [`/.old/autoencoder/dataset/util.py#376`](/.old/autoencoder/dataset/util.py#376) 
  | view on GitHub: [.old/autoencoder/dataset/util.py#L376](https://github.com/mivanit/attention-motifs/blob/main/.old/autoencoder/dataset/util.py#L376)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=create%20AttentionPatternMetadataArray%20here%20instead%2C%20then%20concatenate%20them%20all%20at%20the%20end&body=%23%20source%0A%0A%5B%60.old%2Fautoencoder%2Fdataset%2Futil.py%23L376%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.old%2Fautoencoder%2Fdataset%2Futil.py%23L376%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09head_patterns%3A%20AttentionPatternBatch%20%3D%20layer_patterns%5B%3A%2C%20head%5D%0A%0A%09%09%09%09%23%20TODO%3A%20create%20AttentionPatternMetadataArray%20here%20instead%2C%20then%20concatenate%20them%20all%20at%20the%20end%0A%09%09%09%09%23%20will%20require%20messing%20around%20with%20the%20model%20index%2C%20maybe%20make%20that%20a%20hash%3F%0A%60%60%60&labels=enhancement)

  ```python
head_patterns: AttentionPatternBatch = layer_patterns[:, head]

				# TODO: create AttentionPatternMetadataArray here instead, then concatenate them all at the end
				# will require messing around with the model index, maybe make that a hash?
  ```




## [`attn_embed/features/features.py`](/attn_embed/features/features.py)

- mass as a function of distance from diagonal  
  local link: [`/attn_embed/features/features.py#182`](/attn_embed/features/features.py#182) 
  | view on GitHub: [attn_embed/features/features.py#L182](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/features/features.py#L182)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=mass%20as%20a%20function%20of%20distance%20from%20diagonal&body=%23%20source%0A%0A%5B%60attn_embed%2Ffeatures%2Ffeatures.py%23L182%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Ffeatures%2Ffeatures.py%23L182%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09prefix%3D%22hist%22%2C%0A%09%29%0A%09%23%20TODO%3A%20mass%20as%20a%20function%20of%20distance%20from%20diagonal%0A%60%60%60&labels=enhancement)

  ```python
prefix="hist",
	)
	# TODO: mass as a function of distance from diagonal
  ```


- standard features on decay rate  
  local link: [`/attn_embed/features/features.py#202`](/attn_embed/features/features.py#202) 
  | view on GitHub: [attn_embed/features/features.py#L202](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/features/features.py#L202)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=standard%20features%20on%20decay%20rate&body=%23%20source%0A%0A%5B%60attn_embed%2Ffeatures%2Ffeatures.py%23L202%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Ffeatures%2Ffeatures.py%23L202%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%2A%2Aprefix_dict%28vec_features%28A%5B%3A%2C%200%5D%2C%20reduced%3DFalse%29%2C%20prefix%3D%22first_tok%22%29%2C%0A%09%09%23%20transition%20tensor%3A%20standard%20features%2C%20standard%20features%20on%20diff%2C%20linear%20envelope%20on%20transition%20time%0A%09%09%23%20%09TODO%3A%20standard%20features%20on%20decay%20rate%0A%09%09%23%20markov%20transition%20not%20that%20important%3F%0A%09%09%23%20%2A%2Aprefix_dict%28%0A%60%60%60&labels=enhancement)

  ```python
**prefix_dict(vec_features(A[:, 0], reduced=False), prefix="first_tok"),
		# transition tensor: standard features, standard features on diff, linear envelope on transition time
		# 	TODO: standard features on decay rate
		# markov transition not that important?
		# **prefix_dict(
  ```


- fit fft in `gram_features`, but this is expensive  
  local link: [`/attn_embed/features/features.py#209`](/attn_embed/features/features.py#209) 
  | view on GitHub: [attn_embed/features/features.py#L209](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/features/features.py#L209)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=fit%20fft%20in%20%60gram_features%60%2C%20but%20this%20is%20expensive&body=%23%20source%0A%0A%5B%60attn_embed%2Ffeatures%2Ffeatures.py%23L209%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Ffeatures%2Ffeatures.py%23L209%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%23%20%29%2C%0A%09%09%23%20%23%20%7Blog%2C%20raw%7D%20gram%20matrix%20of%20%7Brows%2C%20cols%2C%20rows%20of%20skewed%7D%3A%20beta%20fit%20hist%0A%09%09%23%20%23%20%09TODO%3A%20fit%20fft%20in%20%60gram_features%60%2C%20but%20this%20is%20expensive%0A%09%09%2A%2Aprefix_dict%28%0A%09%09%09gram_features%28A%20%40%20A.T%29%2C%0A%60%60%60&labels=enhancement)

  ```python
# ),
		# # {log, raw} gram matrix of {rows, cols, rows of skewed}: beta fit hist
		# # 	TODO: fit fft in `gram_features`, but this is expensive
		**prefix_dict(
			gram_features(A @ A.T),
  ```




## [`attn_embed/frontend/embeds-old/assemble_display.py`](/attn_embed/frontend/embeds-old/assemble_display.py)

- move this to muutils  
  local link: [`/attn_embed/frontend/embeds-old/assemble_display.py#5`](/attn_embed/frontend/embeds-old/assemble_display.py#5) 
  | view on GitHub: [attn_embed/frontend/embeds-old/assemble_display.py#L5](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/frontend/embeds-old/assemble_display.py#L5)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=move%20this%20to%20muutils&body=%23%20source%0A%0A%5B%60attn_embed%2Ffrontend%2Fembeds-old%2Fassemble_display.py%23L5%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Ffrontend%2Fembeds-old%2Fassemble_display.py%23L5%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20move%20this%20to%20muutils%0Adef%20inline_html_assets%28%0A%09html%3A%20str%2C%0A%60%60%60&labels=enhancement)

  ```python
# TODO: move this to muutils
def inline_html_assets(
	html: str,
  ```




## [`attn_embed/util/pipeline_cfg.py`](/attn_embed/util/pipeline_cfg.py)

- check models actually exist in TransformerLens?  
  local link: [`/attn_embed/util/pipeline_cfg.py#111`](/attn_embed/util/pipeline_cfg.py#111) 
  | view on GitHub: [attn_embed/util/pipeline_cfg.py#L111](https://github.com/mivanit/attention-motifs/blob/main/attn_embed/util/pipeline_cfg.py#L111)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=check%20models%20actually%20exist%20in%20TransformerLens%3F&body=%23%20source%0A%0A%5B%60attn_embed%2Futil%2Fpipeline_cfg.py%23L111%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattn_embed%2Futil%2Fpipeline_cfg.py%23L111%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20validate_cfg%28self%29%20-%3E%20None%3A%0A%09%09%23%20TODO%3A%20check%20models%20actually%20exist%20in%20TransformerLens%3F%0A%09%09assert%20all%28isinstance%28model%2C%20str%29%20for%20model%20in%20self.models%29%2C%20%28%0A%09%09%09%22All%20models%20must%20be%20strings%22%0A%60%60%60&labels=enhancement)

  ```python
def validate_cfg(self) -> None:
		# TODO: check models actually exist in TransformerLens?
		assert all(isinstance(model, str) for model in self.models), (
			"All models must be strings"
  ```




## [`README.md`](/README.md)

-   
  local link: [`/README.md#72`](/README.md#72) 
  | view on GitHub: [README.md#L72](https://github.com/mivanit/attention-motifs/blob/main/README.md#L72)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=Issue%20from%20inline%20todo&body=%23%20source%0A%0A%5B%60README.md%23L72%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2FREADME.md%23L72%29%0A%0A%23%20context%0A%60%60%60markdown%0A%23%20TODO%3A%0A%0A-%20computing%20features%3A%0A%60%60%60&labels=enhancement)

  ```markdown
# TODO:

- computing features:
  ```




## [`tests/integration/test_train.py`](/tests/integration/test_train.py)

- finish this test to use the train() function  
  local link: [`/tests/integration/test_train.py#115`](/tests/integration/test_train.py#115) 
  | view on GitHub: [tests/integration/test_train.py#L115](https://github.com/mivanit/attention-motifs/blob/main/tests/integration/test_train.py#L115)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=finish%20this%20test%20to%20use%20the%20train%28%29%20function&body=%23%20source%0A%0A%5B%60tests%2Fintegration%2Ftest_train.py%23L115%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Ftests%2Fintegration%2Ftest_train.py%23L115%29%0A%0A%23%20context%0A%60%60%60python%0A%09%29%0A%0A%09%23%20TODO%3A%20finish%20this%20test%20to%20use%20the%20train%28%29%20function%0A%60%60%60&labels=enhancement)

  ```python
)

	# TODO: finish this test to use the train() function
  ```




