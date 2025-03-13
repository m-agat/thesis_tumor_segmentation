import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = "./output_swinunetr2574357.log"

# Dictionary to store HD95 values per configuration
configurations = {
    "Config 1": {"NCR": [], "ED": [], "ET": []},
    "Config 2": {"NCR": [], "ED": [], "ET": []},
    "Config 3": {"NCR": [], "ED": [], "ET": []},
    "Config 4": {"NCR": [], "ED": [], "ET": []},
}

# Read and parse the log file
current_config = None

with open(log_file_path, "r") as file:
    for line in file:
        # Detect a new configuration
        match_config = re.search(r"Testing Configuration (\d+)/\d+", line)
        if match_config:
            config_num = int(match_config.group(1))
            if config_num in range(1, 5):
                current_config = f"Config {config_num}"

        # Extract HD95 values
        match_hd95 = re.search(
            r"HD95 NCR: ([\d\.]+|nan), HD95 ED: ([\d\.]+), HD95 ET: ([\d\.]+)", line
        )
        if match_hd95 and current_config:
            hd95_ncr = (
                np.nan if match_hd95.group(1) == "nan" else float(match_hd95.group(1))
            )
            hd95_ed = float(match_hd95.group(2))
            hd95_et = float(match_hd95.group(3))

            # Append values
            configurations[current_config]["NCR"].append(hd95_ncr)
            configurations[current_config]["ED"].append(hd95_ed)
            configurations[current_config]["ET"].append(hd95_et)

# Convert to DataFrame
config_dfs = {key: pd.DataFrame(values) for key, values in configurations.items()}


# Function to create and save boxplots for each sub-region
def plot_and_save_boxplot(subregion, filename):
    plt.figure(figsize=(6, 4))
    data = [config_dfs[config][subregion].dropna() for config in configurations.keys()]
    plt.boxplot(data, labels=configurations.keys())
    plt.title(f"HD95 Boxplot - {subregion}")
    plt.ylabel("HD95 (mm)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# Generate and save individual plots
plot_and_save_boxplot("NCR", "hd95_swinunetr_ncr.png")
plot_and_save_boxplot("ED", "hd95_swinunetr_ed.png")
plot_and_save_boxplot("ET", "hd95_swinunetr_et.png")

print("Plots saved as hd95_ncr.png, hd95_ed.png, and hd95_et.png")
