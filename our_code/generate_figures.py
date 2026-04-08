"""
generate_figures.py - Generate all figures for the report

This script reads the result CSVs from our experiments and produces figures to be included in the report

Prerequisites:
    Run all experiments first:
    - reproduce_ucr.py 
    - reproduce_scalability.py 
    - sensitivity_analysis.py 
    - improvements.py 

Usage:
    python generate_figures.py --results_path results --paper_results_path ../original_authors/results --output_path figures

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def setup_style():
    """Set up matplotlib style for figures"""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def fig1_accuracy_comparison(results_path, output_path):
    """
    Figure 1: Scatter plot of our reproduced accuracy vs paper accuracy
    Each point is one of the 128 UCR datasets. Points near the diagonal
    indicate successful reproduction. Outliers are labelled.
    """
    print("Creating Figure 1: Accuracy comparison scatter plot")
    comp = pd.read_csv(
        os.path.join(results_path, 'comparison_ucr.csv'), index_col='dataset',
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        comp['paper_accuracy_mean'], comp['our_accuracy_mean'],
        s=20, alpha=0.6, color='#2166ac', edgecolors='white', linewidth=0.3,
    )

    # Diagonal line (perfect match)
    ax.plot([0.2, 1.05], [0.2, 1.05], 'k--', linewidth=0.8, alpha=0.5,
            label='Perfect match')

    # Label outliers with large differences
    outliers = comp[comp['abs_difference'] > 0.15]
    for name, row in outliers.iterrows():
        ax.annotate(
            name,
            (row['paper_accuracy_mean'], row['our_accuracy_mean']),
            fontsize=6, alpha=0.7, ha='left',
            xytext=(5, -5), textcoords='offset points',
        )

    ax.set_xlabel('Paper Accuracy')
    ax.set_ylabel('Our Reproduced Accuracy')
    ax.set_title(f'Reproduction: Our Results vs Paper ({len(comp)} Datasets)')
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.2, 1.05)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig1_accuracy_comparison.png'))
    plt.close()


def fig2_scalability_length(results_path, paper_results_path, output_path):
    """
    Figure 2: Training time vs time series length
    Compares our results against the paper's on the InlineSkate dataset.
    """
    print("Creating Figure 2: Scalability by time series length")
    ts_len = pd.read_csv(
        os.path.join(results_path, 'our_results_scalability_ts_length.csv'),
    )
    our_max_length = int(ts_len['length'].max())

    # Load paper results from their CSV
    paper_ts = pd.read_csv(
        os.path.join(paper_results_path, 'results_scalability_time_series_length.csv'),
    )
    # Only include paper data points where we also have data
    paper_ts = paper_ts[paper_ts['length'] <= our_max_length]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        ts_len['length'], ts_len['time_training_seconds'], 'o-',
        color='#2166ac', linewidth=2, markersize=6, label='Our results',
    )
    ax.plot(
        paper_ts['length'], paper_ts['time_training_seconds'], 's--',
        color='#b2182b', linewidth=2, markersize=6, label='Paper results',
    )

    ax.set_xlabel('Time Series Length')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Scalability: Training Time vs Time Series Length')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x)}'),
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.2f}'),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig2_scalability_length.png'))
    plt.close()


def fig3_num_kernels(results_path, output_path):
    """
    Figure 3: Mean accuracy vs number of kernels
    Shows diminishing returns as kernel count increases, confirming
    the paper's choice of 10,000 as a good default
    """
    print("Creating Figure 3: Number of kernels vs accuracy")
    nk = pd.read_csv(
        os.path.join(results_path, 'sensitivity_num_kernels.csv'),
    )
    num_datasets = nk['dataset'].nunique()
    nk_summary = nk.groupby('num_kernels')['accuracy_mean'].agg(
        ['mean', 'std'],
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        nk_summary['num_kernels'], nk_summary['mean'],
        yerr=nk_summary['std'], fmt='o-', color='#2166ac',
        linewidth=2, markersize=6, capsize=3,
    )
    ax.set_xlabel('Number of Kernels')
    ax.set_ylabel(f'Mean Accuracy ({num_datasets} datasets)')
    ax.set_title('Sensitivity: Effect of Number of Kernels')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x):,}'),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig3_num_kernels.png'))
    plt.close()


def fig4_feature_ablation(results_path, output_path):
    """
    Figure 4: PPV only vs max only vs both features
    Confirms the paper's claim that PPV is the most critical feature.
    """
    print("Creating Figure 4: Feature ablation")
    feat = pd.read_csv(
        os.path.join(results_path, 'sensitivity_features.csv'),
    )
    num_datasets = feat['dataset'].nunique()
    feat_summary = feat.groupby('feature_type')['accuracy_mean'].mean().reset_index()
    feat_summary = feat_summary.sort_values('accuracy_mean', ascending=True)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [
        '#b2182b' if 'max' in ft else '#2166ac' if 'ppv' in ft else '#1b7837'
        for ft in feat_summary['feature_type']
    ]
    bars = ax.barh(
        feat_summary['feature_type'], feat_summary['accuracy_mean'],
        color=colors, height=0.5,
    )
    ax.set_xlabel(f'Mean Accuracy ({num_datasets} datasets)')
    ax.set_title('Sensitivity: PPV vs Max vs Both Features')
    ax.set_xlim(0.80, 0.86)

    for bar, val in zip(bars, feat_summary['accuracy_mean']):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig4_feature_ablation.png'))
    plt.close()


def fig5_component_ablation(results_path, output_path):
    """
    Figure 5: Effect of removing dilation, bias, and padding
    Shows how much accuracy drops when each component is removed
    """
    print("Creating Figure 5: Component ablation")
    dil = pd.read_csv(os.path.join(results_path, 'sensitivity_dilation.csv'))
    bias = pd.read_csv(os.path.join(results_path, 'sensitivity_bias.csv'))
    pad = pd.read_csv(os.path.join(results_path, 'sensitivity_padding.csv'))

    num_datasets = dil['dataset'].nunique()

    components = ['Dilation', 'Bias', 'Padding']
    with_vals = [
        dil[dil['variant'].str.contains('with')]['accuracy_mean'].mean(),
        bias[bias['variant'].str.contains('with')]['accuracy_mean'].mean(),
        pad[pad['variant'].str.contains('random')]['accuracy_mean'].mean(),
    ]
    without_vals = [
        dil[dil['variant'].str.contains('no_d')]['accuracy_mean'].mean(),
        bias[bias['variant'].str.contains('no_b')]['accuracy_mean'].mean(),
        pad[pad['variant'].str.contains('no_p')]['accuracy_mean'].mean(),
    ]
    drops = [wo - w for w, wo in zip(with_vals, without_vals)]

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, with_vals, width, label='Default (with)', color='#2166ac')
    ax.bar(x + width / 2, without_vals, width, label='Without', color='#b2182b')

    ax.set_ylabel(f'Mean Accuracy ({num_datasets} datasets)')
    ax.set_title('Sensitivity: Effect of Removing Key Components')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.set_ylim(0.75, 0.86)

    for i, drop in enumerate(drops):
        ax.annotate(
            f'{drop:+.4f}',
            xy=(x[i] + width / 2, without_vals[i]),
            xytext=(0, -15), textcoords='offset points',
            ha='center', fontsize=8, color='#b2182b',
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig5_component_ablation.png'))
    plt.close()


def fig6_classifiers(results_path, output_path):
    """
    Figure 6: Comparison of different classifiers on ROCKET features
    """
    print("Creating Figure 6: Classifier comparison")
    clf = pd.read_csv(
        os.path.join(results_path, 'improvement_classifiers_summary.csv'),
    )
    num_datasets = clf['dataset'].nunique()
    clf_overall = clf.groupby('classifier')['accuracy_mean'].mean().sort_values(
        ascending=True,
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [
        '#1b7837' if 'Ridge' in c else '#4393c3'
        for c in clf_overall['classifier']
    ]
    bars = ax.barh(
        clf_overall['classifier'], clf_overall['accuracy_mean'],
        color=colors, height=0.5,
    )
    ax.set_xlabel(f'Mean Accuracy ({num_datasets} datasets)')
    ax.set_title('Improvement 1: Alternative Classifiers')
    ax.set_xlim(0.81, 0.86)

    for bar, val in zip(bars, clf_overall['accuracy_mean']):
        ax.text(
            val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig6_classifiers.png'))
    plt.close()


def fig7_normalisation(results_path, output_path):
    """
    Figure 7: Histogram of accuracy differences with vs without normalisation
    """
    print("Creating Figure 7: Normalisation effect")
    norm = pd.read_csv(
        os.path.join(results_path, 'improvement_normalisation.csv'),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    diffs = norm['difference']
    ax.hist(diffs, bins=30, color='#2166ac', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(
        x=diffs.mean(), color='#b2182b', linewidth=2, linestyle='-',
        label=f'Mean = {diffs.mean():+.4f}',
    )
    ax.set_xlabel('Accuracy Difference (with norm - without norm)')
    ax.set_ylabel('Number of Datasets')
    ax.set_title('Improvement 2: Effect of Z-Normalisation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig7_normalisation.png'))
    plt.close()


def fig8_stability(results_path, output_path):
    """
    Figure 8: Distribution of accuracy standard deviations across 30 runs
    """
    print("Creating Figure 8: Stability analysis")
    stab = pd.read_csv(
        os.path.join(results_path, 'improvement_stability_summary.csv'),
    )

    stab_detail = pd.read_csv(
        os.path.join(results_path, 'improvement_stability.csv'),
    )
    num_runs = stab_detail.groupby('dataset')['run'].count().mode().values[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        stab['accuracy_std'], bins=25, color='#2166ac',
        edgecolor='white', alpha=0.8,
    )
    ax.axvline(
        x=stab['accuracy_std'].mean(), color='#b2182b', linewidth=2,
        label=f'Mean std = {stab["accuracy_std"].mean():.4f}',
    )
    ax.set_xlabel(f'Standard Deviation of Accuracy ({num_runs} runs)')
    ax.set_ylabel('Number of Datasets')
    ax.set_title(f'Improvement 3: Seed Stability ({num_runs} Independent Runs)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig8_stability.png'))
    plt.close()

def fig9_feature_selection(results_path, output_path):
    """
    Figure 9: Feature selection: accuracy vs variance threshold.
    """
    print("Creating Figure 9: Feature Selection")
    fs = pd.read_csv(
        os.path.join(results_path, 'improvement_feature_selection_summary.csv'),
    )
    num_datasets = fs['dataset'].nunique()
    fs_overall = fs.groupby('threshold').agg(
        mean_acc=('accuracy_mean', 'mean'),
        mean_fraction=('mean_fraction_kept', 'mean'),
    ).reset_index()

    fs_overall = fs_overall.sort_values('mean_fraction', ascending=False)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    labels = fs_overall['threshold'].values
    x = np.arange(len(labels))
    width = 0.35

    bars = ax1.bar(x, fs_overall['mean_acc'], width, color='#2166ac', label='Accuracy')
    ax1.set_ylabel(f'Mean Accuracy ({num_datasets} datasets)')
    ax1.set_xlabel('Variance Threshold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [l.replace('no_selection ', 'None\n') for l in labels], fontsize=8,
    )
    ax1.set_ylim(0.80, 0.86)

    ax2 = ax1.twinx()
    ax2.plot(
        x, fs_overall['mean_fraction'] * 100, 's--', color='#b2182b',
        markersize=8, linewidth=2, label='Features kept (%)',
    )
    ax2.set_ylabel('Features Kept (%)', color='#b2182b')
    ax2.tick_params(axis='y', labelcolor='#b2182b')
    ax2.set_ylim(40, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    ax1.set_title('Improvement 4: Feature Selection (Variance Threshold)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig9_feature_selection.png'))
    plt.close()


def fig10_ensemble(results_path, output_path):
    """
    Figure 10: Ensemble of smaller ROCKET models
    """
    print("Creating Figure 10: Ensemble")
    ens = pd.read_csv(
        os.path.join(results_path, 'improvement_ensemble_summary.csv'),
    )
    num_datasets = ens['dataset'].nunique()
    ens_overall = ens.groupby('config').agg(
        mean_acc=('accuracy_mean', 'mean'),
        mean_time=('mean_time', 'mean'),
    ).reset_index()

    order = [
        'baseline_1x10000', 'ensemble_3x5000',
        'ensemble_5x2000', 'ensemble_5x5000',
    ]
    ens_overall['config'] = pd.Categorical(
        ens_overall['config'], categories=order, ordered=True,
    )
    ens_overall = ens_overall.sort_values('config')

    fig, ax1 = plt.subplots(figsize=(6, 4))

    x = np.arange(len(ens_overall))
    width = 0.35

    clean_labels = [
        '1×10K\n(baseline)', '3×5K\n(ensemble)',
        '5×2K\n(ensemble)', '5×5K\n(ensemble)',
    ]

    bars = ax1.bar(x, ens_overall['mean_acc'], width, color='#2166ac', label='Accuracy')
    ax1.set_ylabel(f'Mean Accuracy ({num_datasets} datasets)')
    ax1.set_xlabel('Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, fontsize=8)
    ax1.set_ylim(0.840, 0.855)

    ax2 = ax1.twinx()
    ax2.bar(
        x + width, ens_overall['mean_time'], width,
        color='#b2182b', alpha=0.7, label='Time (s)',
    )
    ax2.set_ylabel('Mean Time (seconds)', color='#b2182b')
    ax2.tick_params(axis='y', labelcolor='#b2182b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title('Improvement 5: Ensemble of Smaller ROCKET Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig10_ensemble.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for the ROCKET reproduction report.",
    )
    parser.add_argument(
        "--results_path", default="results",
        help="Path to our experiment results CSVs.",
    )
    parser.add_argument(
        "--paper_results_path", default="../original_authors/results",
        help="Path to the paper's published result CSVs.",
    )
    parser.add_argument(
        "--output_path", default="figures",
        help="Directory to save figures.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    setup_style()

    # Generate all figures
    fig1_accuracy_comparison(args.results_path, args.output_path)
    fig2_scalability_length(args.results_path, args.paper_results_path, args.output_path)
    fig3_num_kernels(args.results_path, args.output_path)
    fig4_feature_ablation(args.results_path, args.output_path)
    fig5_component_ablation(args.results_path, args.output_path)
    fig6_classifiers(args.results_path, args.output_path)
    fig7_normalisation(args.results_path, args.output_path)
    fig8_stability(args.results_path, args.output_path)
    fig9_feature_selection(args.results_path, args.output_path)
    fig10_ensemble(args.results_path, args.output_path)

    print(f"\nAll figures saved to {args.output_path}/")
    print("Files:")
    for f in sorted(os.listdir(args.output_path)):
        if f.endswith('.png'):
            print(f"  {f}")


if __name__ == "__main__":
    main()
