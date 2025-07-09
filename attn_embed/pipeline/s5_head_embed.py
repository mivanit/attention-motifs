import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

# attention-motifs
from attn_embed.attnpedia.attnpedia import AttentionPedia
from attn_embed.features.analysis import DistanceTensorResult
from attn_embed.features.head_analysis import (
    create_embedding_df_multi,
    plot_head_embeddings_multi,
)
from attn_embed.util.pipeline_cfg import PipelineConfig, pipeline_step_major


def head_embed(cfg: PipelineConfig) -> None:
    """Generate head embeddings from distance matrix."""
    pipeline_step_major("pipeline step 5: generate head embeddings")
    
    # Load head distances from previous step
    head_dists: DistanceTensorResult = DistanceTensorResult.read(
        cfg.data_path("head_dists_zanj")
    )
    
    # Load AttentionPedia for head type information
    attentionpedia: AttentionPedia = AttentionPedia()
    
    # Create embeddings using multiple methods and parameters
    # This creates embeddings for all models in the data
    head_embed_df: pl.DataFrame = create_embedding_df_multi(
        head_dists=head_dists,
        attnpedia=attentionpedia,
        embedding_methods=["isomap", "umap", "tsne", "pca"],
        n_components_list=[2, 3],
        n_neighbors_list=[2, 4, 8, 16, 32, 64],
        match_model=None,  # Include all models
        save_path=None,  # We'll save manually to follow pipeline conventions
    )
    
    # Save embeddings for frontend visualization
    head_embed_df.write_ndjson(cfg.data_path("head_embed"))
    
    if cfg.verbose > 0:
        print(f"Generated head embeddings: {head_embed_df.shape}")
        
        # Extract method names from embedding column names
        embed_cols = [col for col in head_embed_df.columns if col.startswith("embed.")]
        methods = set()
        for col in embed_cols:
            # Column format: embed.{method}.d{n_components}.b{n_neighbors}.dim.{i}
            method = col.split(".")[1]
            methods.add(method)
        
        print(f"Embedding methods included: {sorted(list(methods))}")
        print(f"Models included: {sorted(head_embed_df['model'].unique().to_list())}")
    
    # Generate figures if enabled
    if cfg.do_figures:
        try:
            fig, axes = plot_head_embeddings_multi(
                head_embed_df,
                color_by="type.group",
                methods=["isomap", "umap", "tsne"],
                sizes=(12, 6),
                figsize=(24, 32),
            )
            
            # Save the figure
            fig.savefig(
                cfg.figure_path("head_embed"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            
            # Clean up
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not generate head embedding figures: {e}")


if __name__ == "__main__":
    import sys
    
    cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
    head_embed(cfg)