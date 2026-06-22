"""
Statistical significance testing suite for WCB-OT.
Implements paired t-tests, bootstrap CIs, and Bonferroni corrections.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any

def paired_t_test(novel_scores: List[float], baseline_scores: List[float]) -> Dict[str, Any]:
    """
    Perform a two-sided paired t-test between novel and baseline results.
    """
    n1, n2 = np.array(novel_scores), np.array(baseline_scores)
    t_stat, p_val = stats.ttest_rel(n1, n2)
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'df': len(n1) - 1,
        'mean_diff': float(np.mean(n1 - n2)),
        'significant_05': p_val < 0.05
    }

def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals.
    """
    arr = np.array(scores)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return (float(lower), float(upper))

def bonferroni_correct(p_values: List[float], n_tests: int) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons.
    """
    return [min(1.0, p * n_tests) for p in p_values]

def compare_all_methods(results: Dict[str, List[float]], metric: str, novel: str = 'wcb_ot') -> pd.DataFrame:
    """
    Compare 'novel' method against all other baselines in results.
    Returns summary DataFrame with t-tests and significance.
    """
    rows = []
    baselines = [m for m in results.keys() if m != novel]
    novel_scores = results[novel]
    
    # 1. Compute Base Stats
    for method, scores in results.items():
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        ci_low, ci_high = bootstrap_ci(scores)
        
        row = {
            'method': method,
            'mean': mean_val,
            'std': std_val,
            'ci_low': ci_low,
            'ci_high': ci_high
        }
        
        # 2. Add t-tests if it's a baseline
        if method != novel:
            test = paired_t_test(novel_scores, scores)
            row.update({
                't_stat': test['t_statistic'],
                'p_value': test['p_value']
            })
        else:
            row.update({'t_stat': np.nan, 'p_value': np.nan})
            
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 3. Apply Bonferroni
    mask = df['method'] != novel
    df.loc[mask, 'p_corrected'] = bonferroni_correct(df.loc[mask, 'p_value'].tolist(), len(baselines))
    
    # 4. Significance Stars
    def get_stars(p):
        if pd.isna(p): return '-'
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return 'n.s.'
    
    df['sig_level'] = df['p_corrected'].apply(get_stars)
    return df

def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Export a publication-ready LaTeX table.
    """
    df_fmt = df.copy()
    # Format mean ± std
    df_fmt['Results (Mean ± Std)'] = df_fmt.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    df_fmt['Confidence Interval (95%)'] = df_fmt.apply(lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
    
    output_cols = ['method', 'Results (Mean ± Std)', 'Confidence Interval (95%)', 'p_corrected', 'sig_level']
    final_df = df_fmt[output_cols]
    
    latex = final_df.to_latex(index=False, caption=caption, label=label, float_format="%.4f")
    # Small cleanup for boldness on novel method (manual post-processing or regex)
    return latex

if __name__ == "__main__":
    pass
