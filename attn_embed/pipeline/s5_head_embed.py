import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

# attention-motifs
from attn_embed.attnpedia.attnpedia import AttentionPedia
from attn_embed.features.analysis import DistanceTensorResult
from attn_embed.features.head_analysis import (
    create_embedding_df_multi,
    plot_head_embeddings_multi,
    plot_head_embeddings,
)
from attn_embed.util.pipeline_cfg import PipelineConfig, pipeline_step_major


def get_embedding_prefixes(df: pl.DataFrame) -> list[str]:
    """Extract all embedding prefixes from DataFrame columns."""
    prefixes = set()
    for col in df.columns:
        if col.startswith("embed.") and ".dim." in col:
            # Remove .dim.{n} suffix to get prefix
            prefix = col.rsplit(".dim.", 1)[0]
            prefixes.add(prefix)
    return sorted(prefixes)


def parse_prefix_info(prefix: str) -> dict[str, str]:
    """Parse embedding prefix to extract method, dimensions, and neighbors."""
    # Format: embed.{method}.d{n_components}.b{n_neighbors}
    parts = prefix.split(".")
    if len(parts) != 4 or parts[0] != "embed":
        raise ValueError(f"Invalid prefix format: {prefix}")
    
    method = parts[1]
    n_components = parts[2][1:]  # Remove 'd' prefix
    n_neighbors = parts[3][1:]   # Remove 'b' prefix
    
    return {
        "method": method,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
    }


def create_html_table(prefixes: list[str], figures_dir: Path) -> str:
    """Create HTML table view of embedding plots with links."""
    # Parse prefixes to organize by method and dimensions
    plots_by_method = {}
    for prefix in prefixes:
        info = parse_prefix_info(prefix)
        method = info["method"]
        n_components = info["n_components"]
        n_neighbors = info["n_neighbors"]
        
        if method not in plots_by_method:
            plots_by_method[method] = {}
        if n_components not in plots_by_method[method]:
            plots_by_method[method][n_components] = {}
        
        plots_by_method[method][n_components][n_neighbors] = prefix
    
    # Generate HTML
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Head Embedding Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .method-section { margin-bottom: 30px; }
        .method-title { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
        .dimension-section { margin-bottom: 20px; }
        .dimension-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; }
        .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .plot-item { border: 1px solid #ddd; padding: 10px; text-align: center; }
        .plot-item img { max-width: 100%; height: auto; }
        .plot-title { font-weight: bold; margin-bottom: 5px; }
        .plot-links { margin-top: 10px; }
        .plot-links a { 
            display: inline-block; 
            margin: 2px 5px; 
            padding: 5px 10px; 
            background-color: #007bff; 
            color: white; 
            text-decoration: none; 
            border-radius: 3px; 
            font-size: 12px;
        }
        .plot-links a:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <h1>Head Embedding Visualizations</h1>
    <p>Click on the links below each plot to open the interactive 3D visualization with the corresponding embedding method and parameters.</p>
"""
    
    for method in sorted(plots_by_method.keys()):
        html_content += f'    <div class="method-section">\n        <div class="method-title">{method.upper()}</div>\n'
        
        for n_components in sorted(plots_by_method[method].keys()):
            html_content += f'        <div class="dimension-section">\n            <div class="dimension-title">{n_components}D Embeddings</div>\n'
            html_content += '            <div class="plots-grid">\n'
            
            for n_neighbors in sorted(plots_by_method[method][n_components].keys(), key=int):
                prefix = plots_by_method[method][n_components][n_neighbors]
                svg_filename = f"{prefix}.svg"
                
                # Create URL for 3D visualization with method-specific parameters
                # Format: ?axes.x=0&axes.y=1&axes.z=2&defaultColorColumn=type.group&numericalPrefix=embed.{method}.d{n_components}.b{n_neighbors}.
                embed_url = f"../vis/embeds/heads/index.html?axes.x=0&axes.y=1&axes.z=2&defaultColorColumn=type.group&numericalPrefix={prefix}.dim."
                
                html_content += f'''                <div class="plot-item">
                    <div class="plot-title">{n_neighbors} neighbors</div>
                    <img src="{svg_filename}" alt="{prefix}">
                    <div class="plot-links">
                        <a href="{embed_url}" target="_blank">View Interactive 3D</a>
                    </div>
                </div>
'''
            
            html_content += '            </div>\n        </div>\n'
        
        html_content += '    </div>\n'
    
    html_content += """</body>
</html>"""
    
    return html_content


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
        head_embed_svgs_dir: Path = cfg.figures_dir / "head-embed-svgs"
        head_embed_svgs_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Create multi-plot PDF
            fig, _ = plot_head_embeddings_multi(
                head_embed_df,
                color_by="type.group",
                methods=["isomap", "umap", "tsne"],
                sizes=(12, 6),
                figsize=(24, 32),
            )
            
            # Save the multi-plot PDF
            fig.savefig(
                cfg.figure_path("head_embed"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            
            # Clean up
            plt.close(fig)
            
            # Generate individual SVG plots for each method/parameter combination
            print("Generating individual SVG plots...")
            prefixes = get_embedding_prefixes(head_embed_df)
            
            if cfg.verbose > 0:
                print(f"Generating {len(prefixes)} individual SVG plots...")
            
            for prefix in prefixes:
                try:
                    # Create individual plot
                    fig, _ = plot_head_embeddings(
                        head_embed_df,
                        prefix=prefix,
                        color_by="type.group",
                        sizes=(12, 6),
                        figsize=(8, 6),
                        title=f"Head Embeddings: {prefix}",
                    )
                    
                    # Save as SVG
                    svg_path = head_embed_svgs_dir / f"{prefix}.svg"
                    fig.savefig(
                        svg_path,
                        format="svg",
                        bbox_inches="tight",
                        pad_inches=0.1,
                    )
                    
                    # Clean up
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Warning: Could not generate plot for {prefix}: {e}")
            
            # Create HTML table view
            print("Creating HTML table view...")
            html_content = create_html_table(prefixes, cfg.figures_dir)
            html_path = cfg.figures_dir / "head_embed_table.html"
            
            with open(html_path, "w") as f:
                f.write(html_content)
            
            print(f"HTML table view saved to: {html_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate head embedding figures: {e}")


if __name__ == "__main__":
    import sys
    
    cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
    head_embed(cfg)