# Inline TODOs


# BUG

## [`attention_motifs/math/matrix_powers.py`](/attention_motifs/math/matrix_powers.py)

- breaks with integer matrices???  
  local link: [`/attention_motifs/math/matrix_powers.py:80`](/attention_motifs/math/matrix_powers.py#L80) 
  | view on GitHub: [attention_motifs/math/matrix_powers.py#L80](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/math/matrix_powers.py#L80)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=breaks%20with%20integer%20matrices%3F%3F%3F&body=%23%20source%0A%0A%5B%60attention_motifs%2Fmath%2Fmatrix_powers.py%23L80%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Fmath%2Fmatrix_powers.py%23L80%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20BUG%3A%20breaks%20with%20integer%20matrices%3F%3F%3F%0Adef%20matrix_powers_torch%28%0A%09A%3A%20Float%5Btorch.Tensor%2C%20%22n%20n%22%5D%2C%0A%60%60%60&labels=bug)

  ```python
  # BUG: breaks with integer matrices???
  def matrix_powers_torch(
  	A: Float[torch.Tensor, "n n"],
  ```





# HACK

## [`.meta/scripts/make_docs.py`](/.meta/scripts/make_docs.py)

- this is kind of fragile  
  local link: [`/.meta/scripts/make_docs.py:152`](/.meta/scripts/make_docs.py#L152) 
  | view on GitHub: [.meta/scripts/make_docs.py#L152](https://github.com/mivanit/attention-motifs/blob/main/.meta/scripts/make_docs.py#L152)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=this%20is%20kind%20of%20fragile&body=%23%20source%0A%0A%5B%60.meta%2Fscripts%2Fmake_docs.py%23L152%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.meta%2Fscripts%2Fmake_docs.py%23L152%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%22%22%22name%20of%20the%20module%2C%20which%20is%20the%20package%20name%20with%20%27-%27%20replaced%20by%20%27_%27%0A%0A%09%09HACK%3A%20this%20is%20kind%20of%20fragile%0A%09%09%22%22%22%0A%09%09return%20self.package_name.replace%28%22-%22%2C%20%22_%22%29%0A%60%60%60&labels=HACK)

  ```python
  """name of the module, which is the package name with '-' replaced by '_'

  HACK: this is kind of fragile
  """
  return self.package_name.replace("-", "_")
  ```




## [`attention_motifs/features/head_analysis.py`](/attention_motifs/features/head_analysis.py)

- add metadata  
  local link: [`/attention_motifs/features/head_analysis.py:206`](/attention_motifs/features/head_analysis.py#L206) 
  | view on GitHub: [attention_motifs/features/head_analysis.py#L206](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/features/head_analysis.py#L206)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=add%20metadata&body=%23%20source%0A%0A%5B%60attention_motifs%2Ffeatures%2Fhead_analysis.py%23L206%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Ffeatures%2Fhead_analysis.py%23L206%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09df%20%3D%20df.with_columns%28pl.Series%28f%22embed.%7Bi%7D%22%2C%20embedding%5B%3A%2C%20i%5D%29%29%0A%0A%09%23%20HACK%3A%20add%20metadata%0A%09df._embed_meta%20%3D%20dict%28%20%20%23%20type%3A%20ignore%5Battr-defined%5D%0A%09%09embedding_method%3Dembedding_method%2C%0A%60%60%60&labels=HACK)

  ```python
  	df = df.with_columns(pl.Series(f"embed.{i}", embedding[:, i]))

  # HACK: add metadata
  df._embed_meta = dict(  # type: ignore[attr-defined]
  	embedding_method=embedding_method,
  ```





# TODO

## [`.meta/scripts/docs_clean.py`](/.meta/scripts/docs_clean.py)

- this is not recursive  
  local link: [`/.meta/scripts/docs_clean.py:71`](/.meta/scripts/docs_clean.py#L71) 
  | view on GitHub: [.meta/scripts/docs_clean.py#L71](https://github.com/mivanit/attention-motifs/blob/main/.meta/scripts/docs_clean.py#L71)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=this%20is%20not%20recursive&body=%23%20source%0A%0A%5B%60.meta%2Fscripts%2Fdocs_clean.py%23L71%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2F.meta%2Fscripts%2Fdocs_clean.py%23L71%29%0A%0A%23%20context%0A%60%60%60python%0A%09%22%22%22delete%20files%20not%20in%20preserved%20set%0A%0A%09TODO%3A%20this%20is%20not%20recursive%0A%09%22%22%22%0A%09for%20path%20in%20docs_dir.iterdir%28%29%3A%0A%60%60%60&labels=enhancement)

  ```python
  """delete files not in preserved set

  TODO: this is not recursive
  """
  for path in docs_dir.iterdir():
  ```




## [`attention_motifs/features/features.py`](/attention_motifs/features/features.py)

- mass as a function of distance from diagonal  
  local link: [`/attention_motifs/features/features.py:206`](/attention_motifs/features/features.py#L206) 
  | view on GitHub: [attention_motifs/features/features.py#L206](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/features/features.py#L206)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=mass%20as%20a%20function%20of%20distance%20from%20diagonal&body=%23%20source%0A%0A%5B%60attention_motifs%2Ffeatures%2Ffeatures.py%23L206%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Ffeatures%2Ffeatures.py%23L206%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09prefix%3D%22hist%22%2C%0A%09%29%0A%09%23%20TODO%3A%20mass%20as%20a%20function%20of%20distance%20from%20diagonal%0A%60%60%60&labels=enhancement)

  ```python
  	prefix="hist",
  )
  # TODO: mass as a function of distance from diagonal
  ```


- standard features on decay rate  
  local link: [`/attention_motifs/features/features.py:228`](/attention_motifs/features/features.py#L228) 
  | view on GitHub: [attention_motifs/features/features.py#L228](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/features/features.py#L228)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=standard%20features%20on%20decay%20rate&body=%23%20source%0A%0A%5B%60attention_motifs%2Ffeatures%2Ffeatures.py%23L228%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Ffeatures%2Ffeatures.py%23L228%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%2A%2Aprefix_dict%28vec_features%28A%5B%3A%2C%200%5D%2C%20reduced%3DFalse%29%2C%20prefix%3D%22first_tok%22%29%2C%0A%09%09%23%20transition%20tensor%3A%20standard%20features%2C%20standard%20features%20on%20diff%2C%20linear%20envelope%20on%20transition%20time%0A%09%09%23%20%09TODO%3A%20standard%20features%20on%20decay%20rate%0A%09%09%23%20markov%20transition%20not%20that%20important%3F%0A%09%09%23%20%2A%2Aprefix_dict%28%0A%60%60%60&labels=enhancement)

  ```python
  **prefix_dict(vec_features(A[:, 0], reduced=False), prefix="first_tok"),
  # transition tensor: standard features, standard features on diff, linear envelope on transition time
  # 	TODO: standard features on decay rate
  # markov transition not that important?
  # **prefix_dict(
  ```


- fit fft in `gram_features`, but this is expensive  
  local link: [`/attention_motifs/features/features.py:235`](/attention_motifs/features/features.py#L235) 
  | view on GitHub: [attention_motifs/features/features.py#L235](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/features/features.py#L235)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=fit%20fft%20in%20%60gram_features%60%2C%20but%20this%20is%20expensive&body=%23%20source%0A%0A%5B%60attention_motifs%2Ffeatures%2Ffeatures.py%23L235%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Ffeatures%2Ffeatures.py%23L235%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%23%20%29%2C%0A%09%09%23%20%23%20%7Blog%2C%20raw%7D%20gram%20matrix%20of%20%7Brows%2C%20cols%2C%20rows%20of%20skewed%7D%3A%20beta%20fit%20hist%0A%09%09%23%20%23%20%09TODO%3A%20fit%20fft%20in%20%60gram_features%60%2C%20but%20this%20is%20expensive%0A%09%09%2A%2Aprefix_dict%28%0A%09%09%09gram_features%28A%20%40%20A.T%29%2C%0A%60%60%60&labels=enhancement)

  ```python
  # ),
  # # {log, raw} gram matrix of {rows, cols, rows of skewed}: beta fit hist
  # # 	TODO: fit fft in `gram_features`, but this is expensive
  **prefix_dict(
  	gram_features(A @ A.T),
  ```




## [`attention_motifs/math/transition_tensor.py`](/attention_motifs/math/transition_tensor.py)

- bug -- when i_idx == 0, tt_resampled[-1] grabs the *last* element  
  local link: [`/attention_motifs/math/transition_tensor.py:146`](/attention_motifs/math/transition_tensor.py#L146) 
  | view on GitHub: [attention_motifs/math/transition_tensor.py#L146](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/math/transition_tensor.py#L146)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=bug%20--%20when%20i_idx%20%3D%3D%200%2C%20tt_resampled%5B-1%5D%20grabs%20the%20%2Alast%2A%20element&body=%23%20source%0A%0A%5B%60attention_motifs%2Fmath%2Ftransition_tensor.py%23L146%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Fmath%2Ftransition_tensor.py%23L146%29%0A%0A%23%20context%0A%60%60%60python%0A%09if%20residuals%3A%0A%09%09res_resampled%20%3D%20np.full%28%28n_idxs%2C%20n_ctx%29%2C%20np.nan%2C%20dtype%3DA.dtype%29%0A%09%09%23%20TODO%3A%20bug%20--%20when%20i_idx%20%3D%3D%200%2C%20tt_resampled%5B-1%5D%20grabs%20the%20%2Alast%2A%20element%0A%09%09%23%20%28highest%20power%29%20via%20Python%20negative%20indexing%20instead%20of%20identity.%0A%09%09%23%20Docstring%20says%20residuals%5B0%5D%20should%20be%20NaN.%20Fix%3A%20range%281%2C%20n_idxs%29.%0A%60%60%60&labels=enhancement)

  ```python
  if residuals:
  	res_resampled = np.full((n_idxs, n_ctx), np.nan, dtype=A.dtype)
  	# TODO: bug -- when i_idx == 0, tt_resampled[-1] grabs the *last* element
  	# (highest power) via Python negative indexing instead of identity.
  	# Docstring says residuals[0] should be NaN. Fix: range(1, n_idxs).
  ```




## [`attention_motifs/pipeline/cfg.py`](/attention_motifs/pipeline/cfg.py)

- check models actually exist in TransformerLens?  
  local link: [`/attention_motifs/pipeline/cfg.py:296`](/attention_motifs/pipeline/cfg.py#L296) 
  | view on GitHub: [attention_motifs/pipeline/cfg.py#L296](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/pipeline/cfg.py#L296)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=check%20models%20actually%20exist%20in%20TransformerLens%3F&body=%23%20source%0A%0A%5B%60attention_motifs%2Fpipeline%2Fcfg.py%23L296%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Fpipeline%2Fcfg.py%23L296%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20validate_cfg%28self%29%20-%3E%20None%3A%0A%09%09%23%20TODO%3A%20check%20models%20actually%20exist%20in%20TransformerLens%3F%0A%09%09assert%20all%28isinstance%28model%2C%20str%29%20for%20model%20in%20self.models%29%2C%20%28%0A%09%09%09%22All%20models%20must%20be%20strings%22%0A%60%60%60&labels=enhancement)

  ```python
  def validate_cfg(self) -> None:
  	# TODO: check models actually exist in TransformerLens?
  	assert all(isinstance(model, str) for model in self.models), (
  		"All models must be strings"
  ```




## [`attention_motifs/pipeline/s3b_feat_fig.py`](/attention_motifs/pipeline/s3b_feat_fig.py)

- i think this is fine, but double check  
  local link: [`/attention_motifs/pipeline/s3b_feat_fig.py:90`](/attention_motifs/pipeline/s3b_feat_fig.py#L90) 
  | view on GitHub: [attention_motifs/pipeline/s3b_feat_fig.py#L90](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/pipeline/s3b_feat_fig.py#L90)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=i%20think%20this%20is%20fine%2C%20but%20double%20check&body=%23%20source%0A%0A%5B%60attention_motifs%2Fpipeline%2Fs3b_feat_fig.py%23L90%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Fpipeline%2Fs3b_feat_fig.py%23L90%29%0A%0A%23%20context%0A%60%60%60python%0A%09plt.legend%28%0A%09%09%23%20TODO%3A%20i%20think%20this%20is%20fine%2C%20but%20double%20check%0A%09%09handles%3Dhandles%2C%20%20%23%20pyright%3A%20ignore%5BreportPossiblyUnboundVariable%5D%0A%09%09loc%3D%22lower%20left%22%2C%0A%60%60%60&labels=enhancement)

  ```python
  plt.legend(
  	# TODO: i think this is fine, but double check
  	handles=handles,  # pyright: ignore[reportPossiblyUnboundVariable]
  	loc="lower left",
  ```




## [`attention_motifs/pipeline/s4c_clustering.py`](/attention_motifs/pipeline/s4c_clustering.py)

- could make method configurable via cfg  
  local link: [`/attention_motifs/pipeline/s4c_clustering.py:25`](/attention_motifs/pipeline/s4c_clustering.py#L25) 
  | view on GitHub: [attention_motifs/pipeline/s4c_clustering.py#L25](https://github.com/mivanit/attention-motifs/blob/main/attention_motifs/pipeline/s4c_clustering.py#L25)
  | [Make Issue](https://github.com/mivanit/attention-motifs/issues/new?title=could%20make%20method%20configurable%20via%20cfg&body=%23%20source%0A%0A%5B%60attention_motifs%2Fpipeline%2Fs4c_clustering.py%23L25%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fattention-motifs%2Fblob%2Fmain%2Fattention_motifs%2Fpipeline%2Fs4c_clustering.py%23L25%29%0A%0A%23%20context%0A%60%60%60python%0A%09%23%20Compute%20clustering%20using%20average%20linkage%20by%20default%0A%09%23%20TODO%3A%20could%20make%20method%20configurable%20via%20cfg%0A%09clustering%3A%20HierarchicalClusteringResult%20%3D%20%28%0A%09%09HierarchicalClusteringResult.from_distance_matrix%28%0A%60%60%60&labels=enhancement)

  ```python
  # Compute clustering using average linkage by default
  # TODO: could make method configurable via cfg
  clustering: HierarchicalClusteringResult = (
  	HierarchicalClusteringResult.from_distance_matrix(
  ```




