#!/usr/bin/env python3
"""
Generate publication-quality figures for the Negative Instruction paper.
Uses data from tables.json to create matplotlib figures.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Style settings for NeurIPS-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional and colorblind-friendly
COLORS = {
    'success': '#2166AC',  # Blue
    'failure': '#B2182B',  # Red
    'baseline': '#4DAF4A',  # Green
    'negative': '#984EA3',  # Purple
    'neutral': '#666666',  # Gray
    'highlight': '#FF7F00',  # Orange
}

def load_data():
    """Load tables.json"""
    with open('experiment_run_20251230_0159_output/tables.json', 'r') as f:
        return json.load(f)

def fig1_violation_rate_vs_pressure(data, output_dir):
    """Figure 1: Violation rate vs P0 with logistic fit"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Get violation rate by bin
    bins_data = data['violation_rate_by_bin']
    bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    means = [d['mean'] for d in bins_data]
    ci_lows = [d['ci_low'] for d in bins_data]
    ci_highs = [d['ci_high'] for d in bins_data]
    n_prompts = [d['n_prompts'] for d in bins_data]
    
    # Calculate error bars
    yerr_low = [m - l for m, l in zip(means, ci_lows)]
    yerr_high = [h - m for m, h in zip(means, ci_highs)]
    
    # Plot points with error bars
    ax.errorbar(bin_centers, means, yerr=[yerr_low, yerr_high], 
                fmt='o', color=COLORS['failure'], capsize=4, capthick=1.5,
                markersize=8, linewidth=1.5, label='Observed (±95% CI)')
    
    # Logistic fit
    x_fit = np.linspace(0, 1, 100)
    intercept, slope = -2.3981, 2.2709
    y_fit = 1 / (1 + np.exp(-(intercept + slope * x_fit)))
    ax.plot(x_fit, y_fit, '-', color=COLORS['neutral'], linewidth=2, 
            label=f'Logistic: σ({intercept:.2f} + {slope:.2f}·P₀)')
    
    # Isotonic approximation (step function through observed points)
    ax.step([0] + bin_centers + [1], [means[0]] + means + [means[-1]], 
            where='mid', color=COLORS['highlight'], linewidth=1.5, 
            linestyle='--', alpha=0.7, label='Isotonic fit')
    
    ax.set_xlabel('Semantic Pressure (P₀)')
    ax.set_ylabel('Violation Rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc='upper left', frameon=False)
    ax.set_title('Violation Probability Increases with Pressure')
    
    # Add sample sizes as annotations
    for i, (x, y, n) in enumerate(zip(bin_centers, means, n_prompts)):
        ax.annotate(f'n={n}', (x, y + 0.04), ha='center', fontsize=7, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_violation_rate.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig1_violation_rate.png'))
    plt.close()
    print("Created Figure 1: Violation rate vs pressure")

def fig2_suppression_by_outcome(data, output_dir):
    """Figure 2: Suppression delta by outcome"""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    # Data from posthoc results
    outcomes = ['Success', 'Failure']
    means = [0.227972, 0.051821]
    ci_lows = [0.213314, 0.016810]
    ci_highs = [0.243216, 0.087105]
    ns = [1996, 504]
    
    yerr_low = [m - l for m, l in zip(means, ci_lows)]
    yerr_high = [h - m for m, h in zip(means, ci_highs)]
    
    colors = [COLORS['success'], COLORS['failure']]
    x = [0, 1]
    
    bars = ax.bar(x, means, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, means, yerr=[yerr_low, yerr_high], fmt='none', 
                color='black', capsize=5, capthick=1.5, linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{o}\n(n={n})' for o, n in zip(outcomes, ns)])
    ax.set_ylabel('Suppression (ΔP = P₀ - P₁)')
    ax.set_title('Suppression is 4.4× Weaker\nin Failure Cases')
    ax.set_ylim(0, 0.30)
    
    # Add significance indicator
    ax.plot([0, 0, 1, 1], [0.26, 0.27, 0.27, 0.26], 'k-', linewidth=1)
    ax.text(0.5, 0.275, '***', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_suppression.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig2_suppression.png'))
    plt.close()
    print("Created Figure 2: Suppression by outcome")

def fig3_attention_metrics(data, output_dir):
    """Figure 3: Attention routing metrics by bin and outcome"""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    
    attn_data = data['attention_metrics_by_bin_outcome']
    
    # Organize by outcome
    bins = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    metrics = ['iar', 'nf', 'tmf']
    titles = ['Instruction Attention Ratio', 'Negation Focus', 'Target-Mention Focus']
    
    for ax, metric, title in zip(axes, metrics, titles):
        success_vals = []
        failure_vals = []
        
        for bin_name in bins:
            for d in attn_data:
                if d['bin'] == bin_name:
                    if d['outcome'] == 'success':
                        success_vals.append(d[metric])
                    else:
                        failure_vals.append(d[metric])
        
        ax.plot(bin_centers, success_vals, 'o-', color=COLORS['success'], 
                label='Success', linewidth=1.5, markersize=5)
        ax.plot(bin_centers, failure_vals, 's--', color=COLORS['failure'], 
                label='Failure', linewidth=1.5, markersize=5)
        
        ax.set_xlabel('P₀')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        
        if metric == 'iar':
            ax.set_ylabel('Attention Mass')
            ax.legend(loc='upper left', frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_attention.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig3_attention.png'))
    plt.close()
    print("Created Figure 3: Attention metrics")

def fig4_logit_lens(data, output_dir):
    """Figure 4: Logit lens curves across layers"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    lens_data = data['logit_lens_curves']
    
    conditions = [
        ('baseline/success', 'Base/Success', COLORS['success'], '-'),
        ('baseline/failure', 'Base/Failure', COLORS['failure'], '-'),
        ('negative/success', 'Neg/Success', COLORS['success'], '--'),
        ('negative/failure', 'Neg/Failure', COLORS['failure'], '--'),
    ]
    
    for key, label, color, style in conditions:
        layers = [d['layer'] for d in lens_data[key]]
        probs = [d['p_sem_first_token'] for d in lens_data[key]]
        ax.plot(layers, probs, style, color=color, label=label, linewidth=1.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('P(Target Token)')
    ax.set_title('Target Probability Diverges in Late Layers')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_xlim(0, 27)
    ax.set_ylim(-0.02, 0.75)
    
    # Highlight late-layer region
    ax.axvspan(21, 27, alpha=0.1, color='gray')
    ax.text(24, 0.02, 'Critical\nLayers', ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_logit_lens.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig4_logit_lens.png'))
    plt.close()
    print("Created Figure 4: Logit lens curves")

def fig5_attn_ffn_decomposition(data, output_dir):
    """Figure 5: Attention vs FFN contributions by layer"""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    decomp = data['attn_ffn_decomp']
    
    for ax, outcome in zip(axes, ['success', 'failure']):
        layers = [d['layer'] for d in decomp[outcome]]
        attn = [d['attn_contrib'] for d in decomp[outcome]]
        ffn = [d['ffn_contrib'] for d in decomp[outcome]]
        
        # Focus on late layers where effects are visible
        late_mask = [l >= 18 for l in layers]
        late_layers = [l for l, m in zip(layers, late_mask) if m]
        late_attn = [a for a, m in zip(attn, late_mask) if m]
        late_ffn = [f for f, m in zip(ffn, late_mask) if m]
        
        ax.bar([l - 0.2 for l in late_layers], late_attn, width=0.4, 
               color=COLORS['baseline'], label='Attention', edgecolor='black', linewidth=0.5)
        ax.bar([l + 0.2 for l in late_layers], late_ffn, width=0.4, 
               color=COLORS['highlight'], label='FFN', edgecolor='black', linewidth=0.5)
        
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Contribution to P(X)')
        ax.set_title(f'{outcome.capitalize()} Cases')
        
        if outcome == 'success':
            ax.legend(loc='upper left', frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_decomposition.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig5_decomposition.png'))
    plt.close()
    print("Created Figure 5: Attention/FFN decomposition")

def fig6_patching_effects(data, output_dir):
    """Figure 6: Activation patching effects by layer"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    patching = data['patching_effects_by_bin']['0.8-1.0']
    layers = [d['layer'] for d in patching]
    effects = [d['mean_delta_p'] for d in patching]
    
    # Color based on sign
    colors = [COLORS['success'] if e < 0 else COLORS['failure'] for e in effects]
    
    ax.bar(layers, effects, color=colors, width=0.8, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=1)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Patching Effect (ΔP)')
    ax.set_title('Late-Layer Patches Increase P(X)')
    
    # Annotate crossover
    ax.annotate('Crossover\n(Layer 23)', xy=(23, 0.01), xytext=(17, 0.05),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8, ha='center')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='Reduces P(X)'),
        Patch(facecolor=COLORS['failure'], label='Increases P(X)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_patching.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig6_patching.png'))
    plt.close()
    print("Created Figure 6: Patching effects")

def fig7_priming_index(data, output_dir):
    """Figure 7: Priming index vs violation rate scatter"""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    pi_data = data['pi_vs_violation_points']
    
    # Filter out NaN PI values
    valid = [(d['pi'], d['violation_rate']) for d in pi_data 
             if d['pi'] is not None and not np.isnan(d['pi'])]
    
    pis = [v[0] for v in valid]
    vrs = [v[1] for v in valid]
    
    ax.scatter(pis, vrs, alpha=0.3, s=15, c=COLORS['neutral'], edgecolors='none')
    
    # Add trend line
    z = np.polyfit(pis, vrs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(pis), max(pis), 100)
    ax.plot(x_line, p(x_line), '-', color=COLORS['failure'], linewidth=2, 
            label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax.set_xlabel('Priming Index (TMF - NF)')
    ax.set_ylabel('Violation Rate')
    ax.set_title('Higher Priming Index → Higher Failure')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_priming.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig7_priming.png'))
    plt.close()
    print("Created Figure 7: Priming index vs violation")

def fig8_failure_taxonomy(output_dir):
    """Figure 8: Failure mode taxonomy pie chart"""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    sizes = [441, 63]
    labels = ['Priming\n(87.5%)', 'Override\n(12.5%)']
    colors = [COLORS['failure'], COLORS['highlight']]
    explode = (0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='', startangle=90, textprops={'fontsize': 10})
    ax.set_title('Failure Mode Distribution\n(n=504 failures)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_taxonomy.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig8_taxonomy.png'))
    plt.close()
    print("Created Figure 8: Failure taxonomy")

def main():
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    data = load_data()
    
    print("\nGenerating publication-quality figures...")
    fig1_violation_rate_vs_pressure(data, output_dir)
    fig2_suppression_by_outcome(data, output_dir)
    fig3_attention_metrics(data, output_dir)
    fig4_logit_lens(data, output_dir)
    fig5_attn_ffn_decomposition(data, output_dir)
    fig6_patching_effects(data, output_dir)
    fig7_priming_index(data, output_dir)
    fig8_failure_taxonomy(output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")
    print("Ready for LaTeX inclusion.")

if __name__ == '__main__':
    main()
