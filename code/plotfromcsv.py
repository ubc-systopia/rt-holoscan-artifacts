import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Patch
import numpy as np
from matplotlib.ticker import FuncFormatter
import ast
import matplotlib.patheffects as pe


def plot_scalabilityboxplots_from_csv(csv_path="data/scalability.csv",
                              save_path="data/runtime_boxplots.pdf"):
    # Gather summed t1+t2 per node-count for A, B, C
    sums_A = {}
    sums_B = {}
    sums_C = {}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                n       = int(row[0])
                t1, t2    = float(row[2]), float(row[3])
                q2t1, q2t2 = float(row[4]), float(row[5])
                q3t1, q3t2 = float(row[6]), float(row[7])
            except (IndexError, ValueError):
                print(f"Skipping malformed row: {row}")
                continue

            # sum & convert to ms
            sums_A.setdefault(n, []).append((t1 + t2) * 1000)
            sums_B.setdefault(n, []).append((q2t1 + q2t2) * 1000)
            sums_C.setdefault(n, []).append((q3t1 + q3t2) * 1000)

    all_nodes = sorted(sums_A.keys())
    data1 = sums_A
    data2 = sums_B
    data3 = sums_C

    x = np.arange(len(all_nodes))
    width = 0.25

    positions1 = x - width
    positions2 = x
    positions3 = x + width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw each set of boxplots with your light fill + simple median color
    plt.boxplot(
        [data1[n] for n in all_nodes],
        positions=positions1,
        widths=width * 0.9,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='blue')
    )
    plt.boxplot(
        [data2[n] for n in all_nodes],
        positions=positions2,
        widths=width * 0.9,
        patch_artist=True,
        boxprops=dict(facecolor='lightgreen'),
        medianprops=dict(color='green')
    )
    plt.boxplot(
        [data3[n] for n in all_nodes],
        positions=positions3,
        widths=width * 0.9,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray'),
        medianprops=dict(color='black')
    )

    # Log scale
    ax.set_yscale('log')
    def ms_to_sec(x, _):
        if x >= 1000:
            return f'{x / 1000:.1f}'  # show 1 decimal place
        else:
            return f'{x / 1000:.2f}'  # show 1 decimal place

    ax.yaxis.set_major_formatter(FuncFormatter(ms_to_sec))
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(15)

    plt.grid(axis='x', linestyle='--', linewidth=0.5)

    # Labels & grid
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in all_nodes], fontsize=15)

    ax.set_xlabel('Node Count', fontsize=20)
    ax.set_ylabel('Runtime (seconds, log scale)', fontsize=20)

    # Custom legend
    legend_handles = [
        Patch(facecolor='lightblue', edgecolor='blue', label='Queue size = 1'),
        Patch(facecolor='lightgreen', edgecolor='green', label='Queue size = 2'),
        Patch(facecolor='lightgray', edgecolor='black', label='Queue size = 3'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=20)
        
    ax.set_ylim(5, 1000000)


    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)



def plot_scalabilityline_from_csv(csv_path="data/scalability.csv", save_path="data/runtime_line.pdf"):
    results = []
    Q2_results = []
    Q3_results = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                n = int(row[0])
                t1 = float(row[2])
                t2 = float(row[3])
                q2t1 = float(row[4])
                q2t2 = float(row[5])
                q3t1 = float(row[6])
                q3t2 = float(row[7])
                results.append((n, t1, t2))
                Q2_results.append((n, q2t1, q2t2))
                Q3_results.append((n, q3t1, q3t2))
            except (IndexError, ValueError):
                print(f"Skipping malformed row: {row}")
                continue

        if not results or not Q2_results or not Q3_results:
            print("No valid data to plot.")
            return

        nodes, times, times2 = zip(*results)
        _, q2times, q2times2 = zip(*Q2_results)
        _, q3times, q3times2 = zip(*Q3_results)

        nodes = np.array(nodes)
        times = np.array(times)
        times2 = np.array(times2)
        q2times = np.array(q2times)
        q2times2 = np.array(q2times2)
        q3times = np.array(q3times)
        q3times2 = np.array(q3times2)

        unique_nodes = np.unique(nodes)

        def get_avg_percentages_and_errors(nodes, t1s, t2s):
            p1s, p2s = [], []
            e1s, e2s = [], []
            for n in unique_nodes:
                mask = (nodes == n)
                t1_vals = t1s[mask]
                t2_vals = t2s[mask]
                totals = t1_vals + t2_vals
                p1_vals = np.where(totals > 0, (t1_vals / totals) * 100, 0)
                p2_vals = np.where(totals > 0, (t2_vals / totals) * 100, 0)

                p1s.append(np.mean(p1_vals))
                p2s.append(np.mean(p2_vals))

                # Compute Standard Error of the Mean
                n_samples = len(p1_vals)
                e1s.append(np.std(p1_vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0)
                e2s.append(np.std(p2_vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0)
            return p1s, p2s, e1s, e2s

        avg_p1, avg_p2, err_p1, err_p2 = get_avg_percentages_and_errors(nodes, times, times2)
        q2_p1, q2_p2, q2_err1, q2_err2 = get_avg_percentages_and_errors(nodes, q2times, q2times2)
        q3_p1, q3_p2, q3_err1, q3_err2 = get_avg_percentages_and_errors(nodes, q3times, q3times2)       

        fig, ax = plt.subplots(figsize=(10, 5.2))

        # Color scheme
        colors = {
            'A': 'lightblue',
            'B': 'lightgreen',
            'C': 'lightgray'
        }

        outline = [pe.Stroke(linewidth=5, foreground='black'), pe.Normal()]

        # A (Queue size = 1)
        ax.plot(unique_nodes, avg_p1, linestyle='-', color=colors['A'], linewidth=4, path_effects=outline)
        ax.plot(unique_nodes, avg_p2, linestyle='--', color=colors['A'], linewidth=4, path_effects=outline)

        # B (Queue size = 2)
        ax.plot(unique_nodes, q2_p1, linestyle='-', color=colors['B'], linewidth=4, path_effects=outline)
        ax.plot(unique_nodes, q2_p2, linestyle='--', color=colors['B'], linewidth=4, path_effects=outline)

        # C (Queue size = 3)
        ax.plot(unique_nodes, q3_p1, linestyle='-', color=colors['C'], linewidth=4, path_effects=outline)
        ax.plot(unique_nodes, q3_p2, linestyle='--', color=colors['C'], linewidth=4, path_effects=outline)

        ax.set_xlabel('Node Count', fontsize=20)
        ax.set_ylabel('Runtime (% of Total)', fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(True, linestyle='--', linewidth=0.5)

        from matplotlib.lines import Line2D

        # Line style meaning
        legend_lines = [
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='TG-DFS'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Maximization'),
        ]

        # Color coding (all shown as colored lines)
        legend_colors = [
            Line2D([0], [0], color='lightblue', linestyle='-', linewidth=2, label='Queue size = 1'),
            Line2D([0], [0], color='lightgreen', linestyle='-', linewidth=2, label='Queue size = 2'),
            Line2D([0], [0], color='lightgray', linestyle='-', linewidth=2, label='Queue size = 3'),
        ]

        # Optional: SEM error bar indicator
        #legend_error = Line2D(
        #    [0], [0], color='black', linestyle='-', linewidth=1,
        #    marker='|', markersize=10, label='± SEM'
        #)

        # Combine all legend entries
        legend_handles = legend_lines + legend_colors #+ [legend_error]

        # Display the legend with manual placement
        ax.legend(
            handles=legend_handles,
            loc='upper left',               # Anchor point
            bbox_to_anchor=(0, 0.7),      # Manually position legend box (adjust these!)
            fontsize=14,
            frameon=True,
            framealpha=0.9,
            edgecolor='gray'
        )

        fig.tight_layout(rect=[0, 0, 0.85, 1])

        fig.savefig(save_path, format="pdf", bbox_inches='tight')

def plot_exprcount_from_csv(csv_path="data/exprcount.csv", save_path="data/exprcount_boxplots.pdf"):
    # Read raw data
    expr_results = []
    exprQ2_results = []
    exprQ3_results = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                n = int(row[0])
                exprs = float(row[2])
                exprsQ2 = float(row[3])
                exprsQ3 = float(row[4])
            except (IndexError, ValueError):
                print(f"Skipping malformed row: {row}")
                continue
            expr_results.append((n, exprs))
            exprQ2_results.append((n, exprsQ2))
            exprQ3_results.append((n, exprsQ3))

    # Organize data by node count
    all_nodes = sorted({n for n, _ in expr_results})
    data1 = {n: [] for n in all_nodes}
    data2 = {n: [] for n in all_nodes}
    data3 = {n: [] for n in all_nodes}

    for n, val in expr_results:
        data1[n].append(val)
    for n, val in exprQ2_results:
        data2[n].append(val)
    for n, val in exprQ3_results:
        data3[n].append(val)

    # Prepare positions for three boxplots per node
    x = np.array(all_nodes)
    width = 0.95
    positions1 = x - 1.5*width
    positions2 = x
    positions3 = x + 1.5*width

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Draw each set of boxplots
    plt.boxplot(
        [data1[n] for n in all_nodes],
        positions=positions1,
        widths=width * 0.9,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='blue')
    )
    plt.boxplot(
        [data2[n] for n in all_nodes],
        positions=positions2,
        widths=width * 0.9,
        patch_artist=True,
        boxprops=dict(facecolor='lightgreen'),
        medianprops=dict(color='green')
    )
    plt.boxplot(
        [data3[n] for n in all_nodes],
        positions=positions3,
        widths=width * 0.9,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray'),
        medianprops=dict(color='black')
    )

    # Labels & legend
    plt.xlabel('Node Count', fontsize=20)
    plt.ylabel('Expression Count \n(log scale, base 10)', fontsize=20)
    plt.yscale('log')
    plt.xticks(x, [str(n) for n in x], fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='x', linestyle='--', linewidth=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='lightblue', edgecolor='blue', label='Queue size = 1'),
        Patch(facecolor='lightgreen', edgecolor='green', label='Queue size = 2'),
        Patch(facecolor='lightgray', edgecolor='black', label='Queue size = 3'),
    ]
    plt.legend(handles=legend_handles, fontsize=20, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close()


def plot_qsizescaling(
    csv_path: str = "data/qsizescaling.csv",
    save_path: str = "data/qsizescaling.pdf"
) -> None:
    arrays = []
    with open(csv_path, 'r') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            # Evaluate the complete line (commas preserved)
            arr = ast.literal_eval(text)
            arrays.append(arr)

    # Default legend names
    legend_names = [f"{i}" for i in ["E", "F", "G", "H"]]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot
    for arr, name in zip(arrays, legend_names):
        x = range(10, len(arr) + 1)
        y = [a + b for a, b in arr[9:]]
        ax.plot(x, y, label=name, marker='o')

    # Log scale
    ax.set_yscale('log')

    def ms_to_sec(x, _):
        if x >= 1000:
            return f'{x :.1f}'  # show 1 decimal place
        else:
            return f'{x :.1f}'  # show 3 decimal places

    ax.yaxis.set_major_formatter(FuncFormatter(ms_to_sec))

    # Tick label font sizes
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(15)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(15)

    # Labels and legend
    ax.set_xlabel('Queue size', fontsize=20)
    ax.set_ylabel('Runtime (seconds, log scale)', fontsize=20)
    ax.legend(fontsize=20, loc='upper left')
    plt.tight_layout()

    # Save and close
    plt.savefig(save_path)
    plt.close()

def plotting_main():
    plot_scalabilityboxplots_from_csv()
    plot_scalabilityline_from_csv()
    plot_exprcount_from_csv()
    plot_qsizescaling()

if __name__ == '__main__':
    plotting_main()