"""
Exports CSV tables to high-resolution PNG images for publication/review.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.io import ensure_dir

def export_table_as_png(csv_path: Path, out_path: Path, title: str):
    """
    Renders a pandas DataFrame as a clean matplotlib table.
    """
    df = pd.read_csv(csv_path)
    # Round floats for cleanliness
    df = df.round(4)
    
    fig, ax = plt.subplots(figsize=(10, len(df)*0.5 + 1))
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#001f3f') # Navy
            
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exported {csv_path.name} to {out_path}")

def main():
    table_dir = Path("results/tables")
    fig_dir = Path("results/figures/tables")
    ensure_dir(fig_dir)
    
    # 1. Ablation Table
    ablation_csv = table_dir / "ablation.csv"
    if ablation_csv.exists():
        export_table_as_png(ablation_csv, fig_dir / "ablation_table.png", "Ablation Study Results")
        
    # 2. Sample Complexity Table
    complexity_csv = table_dir / "theorem3_sample_complexity.csv"
    if complexity_csv.exists():
        export_table_as_png(complexity_csv, fig_dir / "theorem3_table.png", "Theorem 3: Sample Complexity Benchmarks")

if __name__ == "__main__":
    main()
