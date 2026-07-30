import os
import sys
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


parser = argparse.ArgumentParser(description="Run UPPAAL verifyta and plot results")
parser.add_argument("xml_base_folder", help="Base path to folder containing configuration subfolders (255, 1000, default)")
parser.add_argument("query_file", help="Path to the UPPAAL query file (e.g., query.q)")
parser.add_argument("--verifyta_path", help="Path to the verifyta executable", default="/opt/uppaal/bin-Linux/verifyta")
parser.add_argument("--output_dir", help="Output directory for results", default="./results")
parser.add_argument("--configs", help="Comma-separated list of configs to process (subset of 255,1000,default)")
parser.add_argument("--no_plot", action="store_true", help="Skip plotting; only write CSV results")
parser.add_argument("--plot_only", action="store_true", help="Only generate plots from existing CSV files; skip verifyta execution")
parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Extra arguments for verifyta (optional). If omitted, defaults to -S 2 -s")
args = parser.parse_args()

xml_base_folder = args.xml_base_folder
query_file = args.query_file
verifyta_path = args.verifyta_path
output_dir = args.output_dir
# Default to memory-lean flags if none provided
extra_args = args.extra_args if args.extra_args else ["-S", "2", "-s"]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# This version of UPPAAL (4.1.26) is academic and doesn't require license acquisition
print("Using UPPAAL 4.1.26 Academic Edition - ready to proceed")                       

# Skip verifyta execution if plot_only mode
if getattr(args, "plot_only", False):
    print("--plot_only mode: Skipping verifyta execution, using existing CSV files")
    all_results = []
    # Load existing CSV data for plotting
    csv_output_file = os.path.join(output_dir, "uppaal_results.csv")
    if os.path.exists(csv_output_file):
        try:
            existing_df = pd.read_csv(csv_output_file)
            all_results = existing_df.to_dict('records')
            print(f"Loaded {len(all_results)} existing results from {csv_output_file}")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")
            all_results = []
else:
    # Process each configuration folder
    configurations = ["255", "1000", "default"]
    if args.configs:
        requested = [c.strip() for c in args.configs.split(",") if c.strip()]
        configurations = [c for c in configurations if c in requested]
    all_results = []

    for config in configurations:
        xml_folder = os.path.join(xml_base_folder, config)
        
        if not os.path.exists(xml_folder):
            print(f"Warning: Configuration folder not found: {xml_folder}")
            continue
            
        xml_files = sorted(f for f in os.listdir(xml_folder) if f.endswith(".xml"))
        
        if not xml_files:
            print(f"No XML files found in: {xml_folder}")
            continue
        
        print(f"\n=== Processing configuration: {config} ===")
        
        for xml_file in xml_files:
            xml_path = os.path.join(xml_folder, xml_file)
            print(f"Running verifyta on: {xml_file}")

            cmd = [verifyta_path] + extra_args + [xml_path, query_file]
            try:
                # Stream stdout line-by-line to avoid buffering entire output in memory
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                found_value = None
                assert process.stdout is not None
                for line in process.stdout:
                    # Optional: echo minimal progress when silenced flags are not used
                    # sys.stdout.write('.') ; sys.stdout.flush()
                    if "Timer.r <=" in line:
                        parts = line.strip().split("<=")
                        if len(parts) >= 2:
                            try:
                                found_value = float(parts[1].strip())
                            except ValueError:
                                found_value = None
                        # We can continue reading until process exits; no need to break

                # Ensure process completion and collect any remaining stderr without large buffering
                _, stderr_text = process.communicate()
                if process.returncode not in (0, None):
                    # If verifyta failed, report brief stderr tail
                    tail = "\n".join(stderr_text.splitlines()[-10:]) if stderr_text else ""
                    print(f"Error running verifyta on {xml_file}:\n{tail}")
                elif found_value is not None:
                    all_results.append({
                        "Application": os.path.splitext(xml_file)[0],
                        "CPU_User_Time": found_value,
                        "Config": config,
                    })

            except Exception as e:
                print(f"Error running verifyta on {xml_file}: {e}")


# Save combined results (append to existing if present)
csv_output_file = os.path.join(output_dir, "uppaal_results.csv")
df_results = pd.DataFrame(all_results)
if os.path.exists(csv_output_file) and not df_results.empty:
    try:
        existing_df = pd.read_csv(csv_output_file)
        df_results = pd.concat([existing_df, df_results], ignore_index=True)
        df_results.drop_duplicates(subset=["Application", "Config"], keep="last", inplace=True)
    except Exception:
        # If existing file is unreadable, overwrite with new df
        pass
df_results.to_csv(csv_output_file, index=False)
print(f"\nResults saved to {csv_output_file}")

# Skip plotting if requested
if getattr(args, "no_plot", False):
    print("--no_plot set, skipping plotting")
    sys.exit(0)

# Check if we have any results
if df_results.empty:
    print("No results found! All verifyta executions failed.")
    print("This might be due to:")
    print("- Network connectivity issues (UPPAAL trying to contact servers)")
    print("- Invalid XML model files")
    print("- Missing query file")
    exit(1)

# Set up plotting parameters
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "legend.title_fontsize": 20
})

# Generate combined comparison graph like plotuppaal.py
print(f"\nGenerating combined comparison graph...")

# Application name mappings (from UPPAAL graph names to short labels)
app_name_map = {
    "UPPAALGraph1": "A",  # Colonoscopy Segmentation
    "UPPAALGraph2": "B",  # Endoscopy Out of Body Detection
    "UPPAALGraph3": "C",  # Body Pose Estimation
    "UPPAALGraph4": "D",  # Endoscopy Depth Estimation
    "UPPAALGraph5": "E",  # Multi AI Endoscopy
    "UPPAALGraph6": "F",  # Endoscopy Depth Estimation (CLAHE)
    "UPPAALGraph7": "G",  # Orsi Multi AI and AR
    "UPPAALGraph8": "H"   # Multi AI Ultrasound
}

app_name_maplong = {
    "UPPAALGraph1": "Colonoscopy Segmentation",
    "UPPAALGraph2": "Endoscopy Out of Body Detection", 
    "UPPAALGraph3": "Body Pose Estimation",
    "UPPAALGraph4": "Endoscopy Depth Estimation",
    "UPPAALGraph5": "Multi AI Endoscopy",
    "UPPAALGraph6": "Endoscopy Depth Estimation (CLAHE)",
    "UPPAALGraph7": "Orsi Multi AI and AR",
    "UPPAALGraph8": "Multi AI Ultrasound"
}

# Prepare data for plotting
df_plot = df_results.copy()

# Replace application names with short labels
df_plot["Application"] = df_plot["Application"].replace(app_name_map)

# Normalize config labels and rename 'default' to clock range
df_plot["Config"] = df_plot["Config"].astype(str).str.strip()
df_plot["Config"] = df_plot["Config"].replace({"default": "215 - 3105"})

# Create pivot table for plotting
config_order = ["255", "1000", "215 - 3105"]
pivot_df = df_plot.pivot_table(index="Application", columns="Config", values="CPU_User_Time", aggfunc="first")
pivot_df = pivot_df.reindex(columns=config_order, fill_value=None)

# Colors matching the original plotuppaal.py
colors = ["lightblue", "lightgreen", "lightgray"]

# Create the plot
ax = pivot_df.plot(kind="bar", figsize=(10, 6), color=colors)
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Format y-axis to show seconds
def ms_to_sec(x, _):
    return f'{x / 1000:.1f}' if x >= 1000 else f'{x / 1000:.2f}'
ax.yaxis.set_major_formatter(FuncFormatter(ms_to_sec))
for lbl in ax.get_yticklabels():
    lbl.set_fontsize(15)

# Set labels and styling
plt.xlabel("Application")
plt.ylabel("CPU Runtime \n(seconds, log scale)")
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Create legends (matching plotuppaal.py style)
handles1, labels1 = ax.get_legend_handles_labels()
legend1 = ax.legend(
    handles=handles1,
    labels=labels1,
    title="Clock Speeds (in MHz)",
    loc='upper left',
    bbox_to_anchor=(0.01, 0.98),
    frameon=True
)

# Application names legend
app_legend_handles = [
    Line2D([0], [0], color='none', marker='', linestyle='',
           label=f"{app_name_map[orig_name]}: {app_name_maplong[orig_name]}")
    for orig_name in sorted(app_name_map.keys(), key=lambda x: app_name_map[x])
]

legend2 = ax.legend(
    handles=app_legend_handles,
    loc='upper left',
    bbox_to_anchor=(-0.05, 0.70), 
    frameon=False,
    ncol=1,
    title="Graph Names"
)

ax.add_artist(legend1)

plt.tight_layout()

# Save combined plot
plot_output_file = os.path.join(output_dir, "UPPAAL.png")
plt.savefig(plot_output_file, format="png", dpi=300, bbox_inches='tight')
print(f"Combined plot saved as {plot_output_file}")

# Also save as PDF
pdf_output_file = os.path.join(output_dir, "UPPAAL.pdf")
plt.savefig(pdf_output_file, format="pdf", dpi=300, bbox_inches='tight')
print(f"Combined plot saved as {pdf_output_file}")

plt.close()

print(f"\nCombined comparison graph generated in {output_dir}/")
