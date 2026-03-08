"""Ablation study infrastructure for induction head verification.

This module provides tools to:
1. Identify candidate induction heads from embedding proximity
2. Perform zero and mean ablation on attention heads
3. Evaluate ablation effects using multiple metrics
4. Run full ablation experiments with baseline comparisons
"""

from attention_motifs.ablation.candidates import (
	CandidateHeads,
	DistanceCandidates,
	get_known_induction_heads,
	find_candidate_induction_heads,
	get_control_heads,
)
from attention_motifs.ablation.ablate import (
	AblationMethod,
	HeadAblator,
)
from attention_motifs.ablation.metrics import (
	repeated_sequence_loss,
	prefix_matching_score,
	preceding_token_score,
	icl_score,
	copying_score,
	ov_copying_score,
	AblationResult,
)
from attention_motifs.ablation.data import (
	generate_repeated_sequences,
	load_icl_texts,
)
from attention_motifs.ablation.experiment import (
	AblationConfig,
	AblationResults,
	run_ablation_experiment,
	evaluate_induction_scores,
	get_all_head_scores,
)
from attention_motifs.ablation.frontend import write_ablation_frontend

__all__ = [
	# candidates
	"CandidateHeads",
	"DistanceCandidates",
	"get_known_induction_heads",
	"find_candidate_induction_heads",
	"get_control_heads",
	# ablate
	"AblationMethod",
	"HeadAblator",
	# metrics
	"repeated_sequence_loss",
	"prefix_matching_score",
	"preceding_token_score",
	"icl_score",
	"copying_score",
	"ov_copying_score",
	"AblationResult",
	# data
	"generate_repeated_sequences",
	"load_icl_texts",
	# experiment
	"AblationConfig",
	"AblationResults",
	"run_ablation_experiment",
	"evaluate_induction_scores",
	"get_all_head_scores",
	# frontend
	"write_ablation_frontend",
]
