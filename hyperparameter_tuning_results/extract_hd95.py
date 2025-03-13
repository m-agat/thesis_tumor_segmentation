import numpy as np
import re

# Path to the log file
log_file_path = "./attunet/output_attunet2598240.log"

# Dictionary to store HD95 values per configuration
configurations = {}

# Variables to track current configuration
current_config = None

# Read and parse the log file
with open(log_file_path, "r") as file:
    for line in file:
        # Detect a new configuration
        match_config = re.search(r"Testing Configuration (\d+)/\d+ for (\w+)", line)
        if match_config:
            current_config = f"Config {match_config.group(1)} ({match_config.group(2)})"
            if current_config not in configurations:
                configurations[current_config] = {
                    "NCR": [],
                    "ED": [],
                    "ET": [],
                }  # Store as lists

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

            # Append values to lists
            configurations[current_config]["NCR"].append(hd95_ncr)
            configurations[current_config]["ED"].append(hd95_ed)
            configurations[current_config]["ET"].append(hd95_et)

# Compute mean and standard deviation properly
for config, values in configurations.items():
    valid_ncr = np.array(values["NCR"])
    valid_ed = np.array(values["ED"])
    valid_et = np.array(values["ET"])

    # Compute mean and std for NCR, ED, and ET separately
    mean_ncr = np.nanmean(valid_ncr) if valid_ncr.size > 0 else np.nan
    std_ncr = np.nanstd(valid_ncr) if valid_ncr.size > 0 else np.nan

    mean_ed = np.nanmean(valid_ed) if valid_ed.size > 0 else np.nan
    std_ed = np.nanstd(valid_ed) if valid_ed.size > 0 else np.nan

    mean_et = np.nanmean(valid_et) if valid_et.size > 0 else np.nan
    std_et = np.nanstd(valid_et) if valid_et.size > 0 else np.nan

    # Compute overall mean HD95 using all three if available
    all_values = np.concatenate((valid_ncr, valid_ed, valid_et))
    all_values = all_values[~np.isnan(all_values)]  # Remove NaN values

    overall_mean_hd95 = np.nanmean(all_values) if all_values.size > 0 else np.nan
    overall_std_hd95 = np.nanstd(all_values) if all_values.size > 0 else np.nan

    # Print extracted values to verify correctness
    print(f"\n=== {config} ===")
    print(
        f"HD95 NCR: {mean_ncr:.4f} ± {std_ncr:.4f}"
        if not np.isnan(mean_ncr)
        else "HD95 NCR: Skipped (NaN)"
    )
    print(f"HD95 ED: {mean_ed:.4f} ± {std_ed:.4f}")
    print(f"HD95 ET: {mean_et:.4f} ± {std_et:.4f}")
    print(f"Overall HD95: {overall_mean_hd95:.4f} ± {overall_std_hd95:.4f}")
