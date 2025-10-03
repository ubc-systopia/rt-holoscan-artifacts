import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Set font size globally ---
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "legend.title_fontsize": 20
})

# Load the CSV
csv_file = "data/GraphVSExecutionTime.csv"
df = pd.read_csv(csv_file)


# Custom names for each application
app_name_map = {
    "Body Pose Estimation": "C",
    "Colonoscopy Segmentation": "A",
    "Endoscopy Depth Estimation": "D",
    "Endoscopy Depth Estimation (CLAHE)": "F",
    "Endoscopy Out of Body Detection": "B",
    "MultiAI Ultrasound": "H",
    "MultiAi Endoscopy": "E",
    "Orsi Multi Ai and AR": "G"
}


app_name_maplong = {
    "Body Pose Estimation": "Body Pose Estimation",
    "Colonoscopy Segmentation": "Colonoscopy Segmentation",
    "Endoscopy Depth Estimation": "Endoscopy Depth Estimation",
    "Endoscopy Depth Estimation (CLAHE)": "Endoscopy Depth Estimation (CLAHE)",
    "Endoscopy Out of Body Detection": "Endoscopy Out of Body Detection",
    "MultiAI Ultrasound": "Multi AI Ultrasound",
    "MultiAi Endoscopy": "Multi AI Endoscopy",
    "Orsi Multi Ai and AR": "Orsi Multi AI and AR"
}

# Normalize config labels and rename 'base' to clock range
df["Config"] = df["Config"].astype(str).str.strip().str.lower()
df["Config"] = df["Config"].replace({"base": "215 - 3105"})


# Replace actual application names with labeled version
df["Application"] = df["Application"].replace(app_name_map)

colors = ["lightblue", "lightgreen", "lightgray"]  

config_order = ["255", "1000", "215 - 3105"]
pivot_df = df.pivot_table(index="Application", columns="Config", values="CPU_User_Time", aggfunc="first")

pivot_df = pivot_df[config_order]

# Plotting
ax = pivot_df.plot(kind="bar", figsize=(10, 6), color=colors)
ax.set_yscale('log')

# Fix rotated x-tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Format y-axis labels
def ms_to_sec(x, _):
    return f'{x / 1000:.1f}' if x >= 1000 else f'{x / 1000:.2f}'
ax.yaxis.set_major_formatter(FuncFormatter(ms_to_sec))
for lbl in ax.get_yticklabels():
    lbl.set_fontsize(15)

# Axis labels and grid
plt.xlabel("Application")
plt.ylabel("CPU Runtime \n(seconds, log scale)")
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.6)

handles1, labels1 = ax.get_legend_handles_labels()
legend1 = ax.legend(
    handles=handles1,
    labels=labels1,
    title="Clock Speeds (in MHz)",
    loc='upper left',
    bbox_to_anchor=(0.01, 0.98),
    frameon=True
)

from matplotlib.lines import Line2D

app_legend_handles = [
    Line2D([0], [0], color='none', marker='', linestyle='',
           label=f"{app_name_map[name]}: {app_name_maplong[name]}")
    for name in sorted(app_name_map, key=app_name_map.get)
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
plt.savefig("data/UPPAAL.pdf", format="pdf", bbox_inches='tight')