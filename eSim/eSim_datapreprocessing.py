import math
import re
from typing import Dict, Any, List, Union
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

rcParams.update({'figure.autolayout': True})
def read_select_and_save(dat_file_path, sps_file_path, columns_to_select, output_csv_path):
    with open(sps_file_path, 'r', encoding='latin-1') as f:
        sps_content = f.read()

    data_list_match = re.search(
        r'DATA\s+LIST\s+FILE=DATA/\s*(.*?)\s*\.',
        sps_content,
        re.IGNORECASE | re.DOTALL
    )

    if not data_list_match:
        print("Error: 'DATA LIST FILE=DATA/' block not found in .sps file.")
        return None

    var_definitions = data_list_match.group(1)
    var_regex = re.compile(r'([a-zA-Z0-9_]+)\s+(\d+)(?:-(\d+))?')

    all_sps_col_names = []
    col_specs = []

    matches = var_regex.findall(var_definitions)

    if not matches:
        print("Error: No variable definitions found in the DATA LIST block.")
        return None

    for match in matches:
        var_name, start_pos_str, end_pos_str = match
        start_pos = int(start_pos_str) - 1

        if end_pos_str:
            end_pos = int(end_pos_str)
        else:
            end_pos = start_pos + 1

        all_sps_col_names.append(var_name)
        col_specs.append((start_pos, end_pos))

    # --- 1. FAST READ ---
    print(f"Starting fast read of {dat_file_path}...")
    df = pd.read_fwf(
        dat_file_path,
        colspecs=col_specs,
        names=all_sps_col_names,
        header=None,
        dtype='str'  # Read all as strings for max speed
    )
    print("Data loaded. Filtering columns...")

    # --- 2. CASE-INSENSITIVE FILTERING ---
    sps_name_map = {name.upper(): name for name in all_sps_col_names}

    final_columns_to_keep = []
    missing_cols = []

    for col in columns_to_select:
        col_upper = col.upper()
        if col_upper in sps_name_map:
            final_columns_to_keep.append(sps_name_map[col_upper])
        else:
            missing_cols.append(col)

    if missing_cols:
        print(f"Warning: The following requested columns were not found in the .sps file: {missing_cols}")

    df_filtered = df[final_columns_to_keep]

    # --- 3. CONVERT COLUMNS TO UPPERCASE (NEW STEP) ---
    print("Converting final column headers to uppercase.")
    df_filtered.columns = df_filtered.columns.str.upper()

    # --- 4. SAVE TO CSV ---
    print(f"Saving {len(final_columns_to_keep)} columns to {output_csv_path}...")
    df_filtered.to_csv(output_csv_path, index=False)

    print("Save complete.")
    print(df_filtered.head())
    return df_filtered
def filter_and_save(csv_file_path: str, values_to_remove_dict: Dict[str, Any], output_csv_path: str,
                    value_replace_dict: Dict[str, Dict] = None, recategorize_dict: Dict[str, Dict] = None,
                    column_rename_dict: Dict[str, str] = None, cols_to_zero_negatives: List[str] = None):
    # 1. Read the CSV (as strings for safe filtering).
    df = pd.read_csv(csv_file_path, dtype=str)
    initial_rows = len(df)
    print(f"Read {initial_rows} rows from {csv_file_path}.")

    # 2. Perform dictionary-based filtering
    for col_name, values in values_to_remove_dict.items():
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found. Skipping filter.")
            continue
        if not isinstance(values, (list, set, tuple)):
            values = [values]
        values_str = [str(v) for v in values]
        df = df[~df[col_name].isin(values_str)]
    print(f"Rows remaining after dictionary filtering: {len(df)}")

    # --- 3. UPDATED (Short Version): Convert Negatives to Zero ---
    if cols_to_zero_negatives:
        print(f"Converting negative values to 0 for: {cols_to_zero_negatives}")
        target_cols = [c for c in cols_to_zero_negatives if c in df.columns]

        # Convert to numeric and mask negatives to 0
        df[target_cols] = df[target_cols].apply(pd.to_numeric, errors='coerce')
        df[target_cols] = df[target_cols].mask(df[target_cols] < 0, 0)

    # 4. Replace 1-to-1 unique values
    if value_replace_dict:
        print("Starting 1-to-1 value replacement...")
        for col_name, replace_map in value_replace_dict.items():
            if col_name not in df.columns:
                print(f"  - Warning: Column '{col_name}' not found. Skipping 1-to-1 replacement.")
                continue
            replace_map_str = {str(k): str(v) for k, v in replace_map.items()}
            df[col_name] = df[col_name].replace(replace_map_str)
            print(f"  - 1-to-1 values replaced in column '{col_name}'.")

    # --- 6. Rename columns ---
    if column_rename_dict:
        missing_keys = list(set(column_rename_dict.keys()) - set(df.columns))
        if missing_keys:
            print(f"  - Warning: Original columns not found for renaming: {missing_keys}")
        df = df.rename(columns=column_rename_dict)
        print("Column renaming step complete.")

    # --- 5. NEW STEP: Re-categorize (many-to-one) values ---
    if recategorize_dict:
        print("Starting value re-categorization...")
        for col_name, mapping in recategorize_dict.items():
            if col_name not in df.columns:
                print(f"  - Warning: Column '{col_name}' not found. Skipping re-categorization.")
                continue

            # Invert the user's mapping to create the final replacement dict
            # e.g., {'NewVal_A': [1, 2], 'NewVal_B': [3, 4]}
            # becomes: {1: 'NewVal_A', 2: 'NewVal_A', 3: 'NewVal_B', 4: 'NewVal_B'}

            # We must convert all items to strings for safe .replace()
            final_replace_map = {}
            for new_value, list_of_old_values in mapping.items():
                new_value_str = str(new_value)
                for old_value in list_of_old_values:
                    final_replace_map[str(old_value)] = new_value_str

            # Apply the replacement
            df[col_name] = df[col_name].replace(final_replace_map)
            print(f"  - Values re-categorized in column '{col_name}'.")

    # --- 7. Save the final filtered DataFrame ---
    rows_removed = initial_rows - len(df)
    print(f"\nRemoved {rows_removed} rows in total.")
    print(f"Saving {len(df)} rows to {output_csv_path}...")

    df.to_csv(output_csv_path, index=False)

    # 4. Print the head of the resulting DataFrame
    print("\n--- Head of the final, filtered data ---")
    print(df)

    return df
def feature_engineering(csv_file_path: str):
    # 1. Read the CSV file
    df = pd.read_csv(csv_file_path)

    # 2. Calculate HHSIZE
    # This will raise a KeyError if 'HH_ID' is not in df.columns
    hh_size_series = df.groupby('HH_ID')['HH_ID'].transform('count')
    ef_size_series = df.groupby('EF_ID')['EF_ID'].transform('count')
    cf_size_series = df.groupby('CF_ID')['CF_ID'].transform('count')

    # 3. Add or update the columns
    df['HHSIZE'] = hh_size_series
    df['EFSIZE'] = ef_size_series
    df['CFSIZE'] = cf_size_series

    # 4. Save the modified DataFrame back to the same file
    df.to_csv(csv_file_path, index=False)

    print(f"Successfully added 'HHSIZE', 'EFSIZE', 'CFSIZE' and saved to {csv_file_path}")

# BALANCE & CHECK
def print_column_info(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # 1. Print total row count
    print(f"--- Total Row Count: {len(df)} ---")

    # 2. Calculate and print empty (NaN) counts
    empty_counts = df.isnull().sum()
    print(f"\n--- Empty Row Counts for: {csv_file_path} ---")
    print(empty_counts)

    # 3. Find and print unique values for each column
    print("\n--- Unique Values for Each Column ---")
    for column in df.columns:
        # Get unique values
        unique_values = df[column].unique()

        # Check if number of unique values is large
        if len(unique_values) > 500:
            print(f"Column '{column}': (Too many unique values to display > 500)")
        else:
            print(f"Column '{column}': {unique_values}")

    # 4. Print the head of the resulting DataFrame
    print("\n--- Head of the final, filtered data ---")
    print(df)
# VISUALIZATION
def plot_value_counts(csv_file_path, columns_to_exclude: list, output_image_path: str):
    """
    Reads a CSV and generates one image with subplots for all columns,
    *except* for those specified in the exclusion list.

    - Plots columns with < 50 unique values as a bar chart.
    - Plots columns with > 50 unique values as a histogram,
      using the 1st to 99th percentile as the range to
      prevent outliers from skewing the view.
    """
    # 1. Ensure output directory exists
    output_file = pathlib.Path(output_image_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 2. Read the CSV data
    df = pd.read_csv(csv_file_path, dtype=str)

    # 3. Determine which columns to plot
    all_columns = df.columns.tolist()
    columns_to_plot = [
        col for col in all_columns if col not in columns_to_exclude
    ]

    if not columns_to_plot:
        print("No columns to plot. Aborting.")
        return

    # 4. Calculate subplot grid size
    n_plots = len(columns_to_plot)
    ncols = 7
    nrows = int(math.ceil(n_plots / ncols))

    # 5. Create the subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3.5 * nrows))
    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    print(f"--- Generating {n_plots} subplots in one image ---")

    # 6. Loop through each column
    for i, col in enumerate(columns_to_plot):
        ax = axes_flat[i]
        n_unique = df[col].nunique()

        if n_unique > 50:
            # --- PLOT AS HISTOGRAM (UPDATED LOGIC) ---
            print(f"Info: Column '{col}' has {n_unique} values. Plotting as histogram.")

            data_numeric = pd.to_numeric(df[col], errors='coerce').dropna()

            if data_numeric.empty:
                print(f"  - Warning: '{col}' is non-numeric. Skipping plot.")
                ax.set_title(f"'{col}' is non-numeric")
                ax.axis('off')
                continue

            # --- FIX IS HERE: Calculate percentile range ---
            # This clips extreme outliers for a better view.
            q01 = data_numeric.quantile(0.01)
            q99 = data_numeric.quantile(0.99)

            plot_range = None
            if q01 < q99:
                plot_range = (q01, q99)
            # --- END FIX ---

            # Plot the histogram with the new range
            data_numeric.plot(
                kind='hist',
                ax=ax,
                bins=30,
                range=plot_range  # Apply the sensible range
            )
            ax.set_title(f"Histogram for Column: {col}")
            ax.set_xlabel(f"{col} (1st-99th percentile)")
            ax.set_ylabel("Frequency / Count")

        else:
            # --- PLOT AS BAR CHART (Original logic) ---
            counts = df[col].value_counts()

            try:
                counts.index = pd.to_numeric(counts.index)
                counts = counts.sort_index()
            except ValueError:
                counts = counts.sort_index()

            counts.plot(kind='bar', ax=ax)
            ax.set_title(f"Value Counts for Column: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency / Count")
            ax.tick_params(axis='x', rotation=45)

    # 7. Turn off any unused subplots
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    # 8. Adjust layout and save the *entire figure*
    fig.tight_layout()
    fig.savefig(output_image_path)
    plt.close(fig)

    print(f"Successfully saved combined chart to: {output_image_path}")
def plot_comparison_by_column(csv_files_dict: Dict[str, str], columns_to_exclude: List[str], output_dir: str):
    """
    Creates one .png file per column.
    - Categorical: Bar Chart.
    - Continuous: Histogram with SHARED X-Axis, Y-Axis, and Bins.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    first_file_path = list(csv_files_dict.values())[0]
    df_temp = pd.read_csv(first_file_path, dtype=str, nrows=0)
    all_columns = df_temp.columns.tolist()
    columns_to_plot = [col for col in all_columns if col not in columns_to_exclude]

    print(f"--- Starting Comparison Plots (excluding {columns_to_exclude}) ---")

    for col in columns_to_plot:
        df_proxy = pd.read_csv(first_file_path, dtype=str, usecols=[col])
        n_unique = df_proxy[col].nunique()

        if n_unique > 500:
            print(f"Skipping '{col}' (>500 unique values)")
            continue

        is_continuous = n_unique > 20
        plot_type = "hist (shared axes)" if is_continuous else "bar"

        print(f"Processing {col} ({plot_type})")

        # 1. Load Data & Find Global Limits (X and Y)
        data_storage = {}
        global_y_max = 0

        # --- NEW: Track Global X Limits for Continuous Vars ---
        global_x_min = float('inf')
        global_x_max = float('-inf')

        for year, file_path in csv_files_dict.items():
            if col not in pd.read_csv(file_path, nrows=0).columns:
                data_storage[year] = None
                continue

            df = pd.read_csv(file_path, dtype=str, usecols=[col])

            if is_continuous:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                data_storage[year] = values

                # Update Global X Limits
                if not values.empty:
                    v_min = values.min()
                    v_max = values.max()
                    if v_min < global_x_min: global_x_min = v_min
                    if v_max > global_x_max: global_x_max = v_max

            else:
                counts = df[col].value_counts()
                try:
                    counts.index = pd.to_numeric(counts.index)
                    counts = counts.sort_index()
                except ValueError:
                    counts = counts.sort_index()

                current_max = counts.max() if not counts.empty else 0
                if current_max > global_y_max: global_y_max = current_max
                data_storage[year] = counts

        # --- NEW: Calculate Common Bins for Histograms ---
        common_bins = None
        if is_continuous and global_x_max > float('-inf'):
            # Create 30 bins spanning the full global range
            common_bins = np.linspace(global_x_min, global_x_max, 31)

            # Pre-calculate histograms with these FIXED bins to find Y-Max
            for year, values in data_storage.items():
                if values is not None and not values.empty:
                    counts, _ = np.histogram(values, bins=common_bins)
                    if counts.max() > global_y_max:
                        global_y_max = counts.max()

        y_axis_limit = global_y_max * 1.1

        # 2. Plotting
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30, 8))
        axes_flat = axes.flatten()
        fig.suptitle(f"Comparison of Column: {col} (2006-2021)", fontsize=22)

        plot_num = 0
        for year, data in data_storage.items():
            if plot_num >= 4: break
            ax = axes_flat[plot_num]

            if data is None or len(data) == 0:
                ax.set_title(f"{year} (No Data)")
                ax.axis('off')
            else:
                if is_continuous:
                    # --- CONTINUOUS PLOT (Shared Bins & Axes) ---
                    # Use the common_bins we calculated earlier
                    ax.hist(data, bins=common_bins, color='steelblue', edgecolor='white', alpha=0.7)
                    ax.set_ylabel("Frequency")

                    # Explicitly set X-Axis to match global range
                    ax.set_xlim(global_x_min, global_x_max)

                    stats_text = (
                        f"Min:  {data.min():.2f}\n"
                        f"Max:  {data.max():.2f}\n"
                        f"Mean: {data.mean():.2f}\n"
                        f"Std:  {data.std():.2f}"
                    )

                    ax.text(0.95, 0.95, stats_text,
                            transform=ax.transAxes,
                            fontsize=12,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                else:
                    # --- CATEGORICAL PLOT ---
                    data.plot(kind='bar', ax=ax, color='steelblue', width=0.8)
                    ax.set_ylabel("Count")
                    ax.tick_params(axis='x', rotation=45)

                # Shared Y-Axis
                ax.set_ylim(0, y_axis_limit)
                ax.set_title(f"Year: {year}", fontsize=16)
                ax.set_xlabel(col)
                ax.grid(axis='y', linestyle='--', alpha=0.5)

            plot_num += 1

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = output_path / f"{col}_comparison.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  -> Saved to {save_path}")

if __name__ == '__main__':
    #BASE_DIR = pathlib.Path("C:/Users/o_iseri/Desktop/2ndJournal")
    BASE_DIR = pathlib.Path("/Users/orcunkoraliseri/Desktop/Postdoc/eSim/Occupancy")

    DATA_DIR = BASE_DIR / "DataSources_CENSUS"
    OUTPUT_DIR = BASE_DIR / "Outputs_CENSUS"

    # --- 2006 Files ---
    cen06 = DATA_DIR / "cen06.dat"
    cen06_sps = DATA_DIR / "cen06.sps"
    cen06_filtered = OUTPUT_DIR / "cen06_filtered.csv"
    cen06_filtered2 = OUTPUT_DIR / "cen06_filtered2.csv"

    # --- 2011 Files ---
    cen11 = DATA_DIR / "cen11.dat"
    cen11_sps = DATA_DIR / "cen11.sps"
    cen11_filtered = OUTPUT_DIR / "cen11_filtered.csv"
    cen11_filtered2 = OUTPUT_DIR / "cen11_filtered2.csv"

    # --- 2016 Files ---
    cen16 = DATA_DIR / "cen16.dat"
    cen16_sps = DATA_DIR / "cen16.sps"
    cen16_filtered = OUTPUT_DIR / "cen16_filtered.csv"
    cen16_filtered2 = OUTPUT_DIR / "cen16_filtered2.csv"

    # --- 2021 Files ---
    cen21 = DATA_DIR / "cen21.dat"
    cen21_sps = DATA_DIR / "cen21.sps"
    cen21_filtered = OUTPUT_DIR / "cen21_filtered.csv"
    cen21_filtered2 = OUTPUT_DIR / "cen21_filtered2.csv"

    """
    #CENSUS2006
    read_select_and_save(cen06, cen06_sps,
                         ["HH_ID", "EF_ID", "CF_ID", "PP_ID", "CMA", "AgeGrp", "SEX", "KOL", "AttSch", "CIP", "NOCS",
                          "EmpIn", "TotInc", "BROOMH", "Room", "DType", "MarSt", "CFStat", "BuiltH", "CONDO",
                          "GENSTAT", "CITIZEN", "LFACT", "CF_RP", "COW", "POWST", "INCTAX","REGION", "VALUE", "REPAIR",
                          "HRSWRK","MODE"], cen06_filtered)
    filter_and_save(csv_file_path=cen06_filtered,
                    values_to_remove_dict={'AGEGRP': [88], 'CIP': [12, 88],'NOCS': [88], "BUILTH":[88], "CONDO":[8], "GENSTAT":[8],"CITIZEN":[8],  
                    "COW":[8], "POWST":[8], "HRSWRK":[999], "VALUE": [8888888], "EMPIN": [88888888,8888888], "TOTINC": [88888888,8888888],
                                           "INCTAX": [8888888]},
                    output_csv_path=cen06_filtered2,
                    column_rename_dict={"BROOMH": "BEDRM", "MARST": "MARSTH", "BUILTH": "BUILT", "LFACT":"LFTAG", "REGION":"PR"},
                    recategorize_dict={"ROOM": {1: [1,2,3], 2: [4, 5], 3: [6,7], 4:[8, 9, 10, 11]},
                                       #"DTYPE": {1: [1], 2: [4, 5, 6], 3: [2, 3, 7, 8]},
                                       "BEDRM": {1: [0, 1,], 2: [2,3], 3: [4,5]}, "MARSTH": {3: [1,3,5], 2: [2], 1:[4]}, "MODE": {4:[6], 6:[7], 7:[8]},
                                       "CFSTAT": {1: [1, 2, 3, 4], 2: [5, 6], 3: [7, 8], 4: [9, 10], 5: [12], 6: [13], 7: [11]},
                                       "BUILT": {1: [1,2,3], 2: [4,5], 3:[6,7,8,9], 4:[10]}, "CF_RP": {3: [0], 1: [1], 2:[2]},
                                       "COW": {1: [4], 2: [2, 5], 3: [3, 6], 4: [1]}, "PR":{1:[1],2:[24],3:[35],4:[46,47,48],5:[59],6:[63]},
                                       "EMPIN": {0: [99999999, 9999999]},"TOTINC": {0: [99999999, 9999999]}, "INCTAX": {0: [9999999]}, "VALUE": {0: [9999999]},
                                       "HRSWRK": {0:[0,999],1:list(range(1,10)),2:list(range(10,20)),3:list(range(20,30)),4:list(range(30,38)),5:list(range(38,41)),6:list(range(41,50)),7:list(range(50,60)),8:list(range(60,70)),9:list(range(70,80)),10:list(range(80,101))}},)
    feature_engineering(csv_file_path=cen06_filtered2)
    #print_column_info(cen06_filtered2) # unique values, total row count, empty rows
    #plot_value_counts(csv_file_path=cen06_filtered2, columns_to_exclude= ["HH_ID", "EF_ID", "CF_ID", "PP_ID",], output_image_path=OUTPUT_DIR / "plot06.png")
    
    #CENSUS2011
    read_select_and_save(cen11, cen11_sps, ["HH_ID", "EF_ID", "CF_ID", "PP_ID", "CMA", "AGEGRP", "SEX", "KOL", "ATTSCH",
                                            "CIP2011", "NOCS", "EMPIN", "TOTINC", "BEDRM", "ROOM", "DTYPE","MARSTH",
                                            "CFSTAT", "BUILT", "CONDO", "GENSTAT", "CITIZEN","LFTAG", "CF_RP",
                                            "COW", "POWST", "INCTAX","PR", "VALUE", "REPAIR", "HRSWRK","MODE"], cen11_filtered)
    
    filter_and_save(csv_file_path=cen11_filtered,
                    values_to_remove_dict={'AGEGRP': [88], 'CIP2011': [12, 88],'NOCS': [88],
                                           "BEDRM": [8], "ROOM": [88], "CONDO":[8], "GENSTAT":[8],"CITIZEN":[8],
                                           "COW":[8], "POWST":[8], "MODE":[8], "EMPIN": [88888888,8888888], "TOTINC": [88888888,8888888],
                                           "VALUE": [8888888],"INCTAX": [888888]},
                    output_csv_path=cen11_filtered2,
                    column_rename_dict={"CIP2011": "CIP"},
                    recategorize_dict={"DTYPE": {1: [1],  2: [4, 5, 6],   3: [2, 3, 7, 8]}, "ROOM": {1: [1,2,3],  2: [4, 5],   3: [6,7], 4:[8,9,10,11]},
                                       "BEDRM": {1: [0, 1,],  2: [2,3],   3: [4,5]}, "MARSTH": {1: [1, 3], 2: [2], 3:[4, 5, 6]},
                                       "CFSTAT": {1:[1,2],2:[3],3:[4],4:[5],5:[6],6:[7],7:[8]},
                                       "BUILT": {1: [1,2,3], 2: [4,5], 3:[6,7,8], 4:[9, 10]},
                                       "COW": {1: [1], 2: [3, 5], 3: [4, 6], 4: [2]},
                                       "HRSWRK": {0: [99]},
                                       "EMPIN": {0: [99999999, 9999999]},"TOTINC": {0: [99999999, 9999999]},"INCTAX": {0: [999999]}, "VALUE": {0: [9999999]},
                                       "PR":{1:[10,11,12,13],2:[24],3:[35],4:[46,47,48],5:[59],6:[63,70]}})
    feature_engineering(csv_file_path=cen11_filtered2)
    #print_column_info(cen11_filtered2) # unique values, total row count, empty rows
    #plot_value_counts(csv_file_path=cen11_filtered2, columns_to_exclude= ["HH_ID", "EF_ID", "CF_ID", "PP_ID",], output_image_path=OUTPUT_DIR / "plot11.png")
    
    #CENSUS2016
    read_select_and_save(cen16, cen16_sps, ["HH_ID", "EF_ID", "CF_ID", "PP_ID", "CMA", "AGEGRP", "SEX", "KOL", "ATTSCH",
                                            "CIP2011", "NOCS", "EMPIN", "TOTINC", "BEDRM", "ROOM", "DTYPE", "MarStH",
                                            "CFSTAT", "BUILT", "CONDO", "GENSTAT", "CITIZEN", "LFTAG", "CF_RP",
                                            "COW", "POWST", "INCTAX", "PR", "VALUE", "REPAIR","HRSWRK","MODE"], cen16_filtered)
    filter_and_save(csv_file_path=cen16_filtered,
                    values_to_remove_dict={'AGEGRP': [88], "BEDRM": [8], 'CIP2011': [88], "DTYPE": [8],
                                           "KOL": [8], 'NOCS': [88], "ROOM": [88], "SEX": [8],
                                           "MARSTH": [8], "CFSTAT":[88], "BUILT": [88],
                                           "CONDO":[8], "GENSTAT":[8],"CITIZEN":[8], "COW":[8], "POWST":[8],
                                           "EMPIN": [88888888,8888888], "TOTINC": [88888888,8888888],"INCTAX": [88888888], "VALUE": [88888888],
                                           },
                    output_csv_path=cen16_filtered2, column_rename_dict={"CIP2011": "CIP"},
                    recategorize_dict= {"ROOM": {1: [1,2,3],  2: [4, 5],   3: [6,7], 4:[8,9,10,11]},
                                        "BEDRM": {1: [0, 1,],  2: [2,3],   3: [4,5]},
                                        "MARSTH": {1: [1, 3], 2: [2], 3:[4]},
                                        "CFSTAT": {1:[1,2],2:[3],3:[4],4:[5],5:[6],6:[7],7:[8]},
                                        "BUILT": {1: [1, 2, 3], 2: [4, 5], 3: [6, 7, 8], 4: [9, 10,11]},
                                        "HRSWRK": {0: [99]},
                                        "EMPIN": {0: [99999999, 9999999]},"TOTINC": {0: [99999999, 9999999]},"INCTAX": {0: [99999999]},"VALUE": {0: [99999999]},
                                        "PR":{1:[10,11,12,13],2:[24],3:[35],4:[46,47,48],5:[59],6:[63,70]}},)
    feature_engineering(csv_file_path=cen16_filtered2)

    #print_column_info(cen16_filtered2) # unique values, total row count, empty rows
    #plot_value_counts(csv_file_path=cen16_filtered2, columns_to_exclude= ["HH_ID", "EF_ID", "CF_ID", "PP_ID",], output_image_path=OUTPUT_DIR / "plot16.png")

    #CENSUS2021
    read_select_and_save(cen21, cen21_sps, ["HH_ID", "EF_ID", "CF_ID", "PP_ID", "CMA", "AGEGRP", "GENDER", "KOL", "ATTSCH",
                                            "CIP2021", "NOC21", "EMPIN", "TOTINC", "BEDRM", "ROOM", "DTYPE", "MarStH",
                                            "CFSTAT", "BUILT", "CONDO", "GENSTAT", "CITIZEN", "LFACT", "CF_RP",
                                            "COW", "POWST", "INCTAX", "PR", "VALUE", "REPAIR","HRSWRK","MODE"], cen21_filtered)

    filter_and_save(csv_file_path=cen21_filtered,
                    values_to_remove_dict={
                        "AGEGRP": [88], "ATTSCH": [8], "BEDRM": [8], 'CIP2021': [12, 88], "DTYPE": [8], "GENDER": [8],"NOC21": [88],
                        "BUILT": [88], "CONDO":[8], "GENSTAT":[8], "CITIZEN":[8], "COW":[8], "POWST":[8],"KOL": [8],"CFSTAT": [88],"MARSTH": [8],
                        "EMPIN": [88888888,8888888], "TOTINC": [88888888,8888888], "INCTAX": [88888888], "VALUE": [88888888],
                    },
                    output_csv_path=cen21_filtered2, 
                    column_rename_dict={"NOC21": "NOCS", "CIP2021": "CIP", "GENDER": "SEX", "LFACT":"LFTAG",},
                    value_replace_dict= {"ATTSCH": {0: 1, 1: 2}},
                    recategorize_dict= {"ROOM": {1: [1,2,3], 2: [4, 5], 3: [6,7], 4:[8,9,10,11]},
                                        "BEDRM": {1: [0, 1,], 2: [2,3], 3: [4,5]}, "MARSTH": {1: [1,3], 2: [2], 3:[4]},
                                        "CFSTAT": {1:[1,2],2:[3],3:[4],4:[5],5:[6],6:[7],7:[8]},
                                        "EMPIN": {0: [99999999, 9999999]},"TOTINC": {0: [99999999, 9999999]}, "INCTAX": {0: [99999999]}, "VALUE": {0: [99999999]},
                                        "BUILT": {1: [1, 2], 2: [3,4], 3: [5, 6, 7], 4: [8, 9, 10,11]},
                                        "PR":{1:[10,11,12,13],2:[24],3:[35],4:[46,47,48],5:[59],6:[70]}},)

    feature_engineering(csv_file_path=cen21_filtered2)
    
    print_column_info(cen21_filtered2) # unique values, total row count, empty rows
    #plot_value_counts(csv_file_path=cen21_filtered2, columns_to_exclude= ["HH_ID", "EF_ID", "CF_ID", "PP_ID",], output_image_path=OUTPUT_DIR / "plot21.png")
    """

    """
    #-------------------------------------------------------------------------------------------------------------------
    # 2. Define your inputs
    csv_paths = {'2006': cen06_filtered2, '2011': cen11_filtered2, '2016': cen16_filtered2, '2021': cen21_filtered2}
    # Define which columns you want to *skip*
    cols_to_exclude = ["HH_ID", "EF_ID", "CF_ID", "PP_ID"]
    # 3. Run the function
    plot_comparison_by_column(csv_paths, cols_to_exclude, OUTPUT_DIR)
    """

    #for DTYPE expansion during PROFILE-MATCHER:POST-PROCESSING
    feature_engineering(cen06_filtered)
    feature_engineering(cen11_filtered)


