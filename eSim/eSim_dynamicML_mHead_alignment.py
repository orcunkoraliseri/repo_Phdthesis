import pandas as pd
import pathlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

# --- 1. Define Paths ---
BASE_DIR = pathlib.Path("/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal")

# Output
OUTPUT_DIR_GSS = BASE_DIR / "Outputs_GSS"
OUTPUT_DIR_ALIGNED = BASE_DIR / "Outputs_Aligned"
OUTPUT_DIR_GSS.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_ALIGNED.mkdir(parents=True, exist_ok=True)

# Inputs
DATA_DIR_GSS = BASE_DIR / "DataSources_GSS/Canada_2022/Data_Données"
FILE_MAIN = DATA_DIR_GSS / "TU_ET_2022_Main-Principal_PUMF.sas7bdat"
FILE_EPISODE = OUTPUT_DIR_GSS / "out22EP_ACT_PRE_coPRE.csv"

CENSUS_FILE = BASE_DIR / "Outputs_CENSUS/forecasted_population_2025_LINKED.csv"

# --- 2. Define Columns ---
# A) MAIN FILE COLUMNS (Demographics - The "Bridge")
COLS_MAIN = [
    'PUMFID',  # Key for merging
    'PRV',  # Geography
    'REGION',  # Geography
    'DDAY',
    'HSDSIZEC',  # Household Size
    'AGEGR10',  # Age
    'GENDER2',  # Sex
    'MARSTAT',  # Marital Status
    'LAN_01',  # Language (KOL)
    'EDC_10',  # School Attendance (ATTSCH)
    'ED_05',  # Degree (HDGREE)
    'NOCLBR_Y',  # Occupation (NOCS)
    'NAIC22CY',  # Industry (NAICS)
    'ACT7DAYC',  # Labor Force (LFTAG Proxy)
    'WET_120',  # Job Type (COW Proxy)
    'WHWD140G',  # Hours Worked (HRSWRK)
    'ATT_150C',  # Commuting Mode (MODE)
    'CTW_140I',  # Place of Work (POWST Proxy)
    'INC_C',  # Total Income (TOTINC)
    'LUC_RST',  # Urban/Rural (CMA Proxy)
    'PHSDFLG', 'CXRFLAG', 'PARNUM'  # Reference Person Helpers
]

# C) RENAME MAP (Census Standards + Your Episode Names)
RENAME_MAP = {
    # --- DEMOGRAPHICS (Census Standard) ---
    'PRV': 'PR',
    'HSDSIZEC': 'HHSIZE',
    'AGEGR10': 'AGEGRP',
    'GENDER2': 'SEX',
    'MARSTAT': 'MARSTH',
    'LAN_01': 'KOL',
    'EDC_10': 'ATTSCH',
    'ED_05': 'HDGREE',
    'NOCLBR_Y': 'NOCS',
    'NAIC22CY': 'NAICS',
    'ACT7DAYC': 'LFTAG',
    'WET_120': 'COW',
    'WHWD140G': 'HRSWRK',
    'ATT_150C': 'MODE',
    'CTW_140I': 'POWST',
    'INC_C': 'TOTINC',
    'LUC_RST': 'CMA',
    "PUMFID": "occID",
}

# --- 3. The Logic Function --
def read_merge_save_gss(main_path, episode_path, cols_main, rename_dict, output_csv_path, chunk_size=100000):
    """
    1. Reads MAIN file fully (Demographics - SAS format).
    2. Reads EPISODE file in chunks (Pre-processed CSV format).
    3. Merges Demographics onto Episodes.
    4. Saves to CSV.
    5. Prints unique values for verification.
    """
    print(f"--- Starting GSS Processing ---")

    # --- STEP 1: Read Main File (Demographics) ---
    print(f"1. Reading Main File: {main_path.name}...")
    if not os.path.exists(main_path):
        print(f"❌ Error: Main file not found.")
        return None

    try:
        # Reading SAS file for Demographics
        df_main = pd.read_sas(main_path, encoding='latin-1')

        # Keep only the relevant demographic columns
        valid_main_cols = [c for c in cols_main if c in df_main.columns]
        df_main = df_main[valid_main_cols]
        print(f"   Loaded Main Data: {len(df_main)} people.")

    except Exception as e:
        print(f"❌ Error reading Main file: {e}")
        return None

    # --- STEP 2: Read Episode File (Schedules) & Merge ---
    print(f"2. Reading Episode File in chunks: {episode_path.name}...")
    if not os.path.exists(episode_path):
        print(f"❌ Error: Episode file not found.")
        return None

    merged_chunks = []

    try:
        # UPDATED: Read CSV instead of SAS
        # We do not filter columns here because the input CSV is already pre-processed
        reader = pd.read_csv(
            episode_path,
            chunksize=chunk_size,
            encoding='utf-8',  # Assuming standard CSV encoding
            low_memory=False
        )

        for i, chunk in enumerate(reader):
            # UPDATED: Handle 'occID' in pre-processed CSV vs 'PUMFID' in Main SAS
            if 'occID' in chunk.columns:
                # Rename to PUMFID to match the Demographics file key
                chunk = chunk.rename(columns={'occID': 'PUMFID'})
            elif 'PUMFID' not in chunk.columns:
                print("❌ Error: Merge key ('occID' or 'PUMFID') missing in Episode chunk. Cannot merge.")
                return None

            # Merge Main Data onto Episode Chunk
            chunk_merged = pd.merge(chunk, df_main, on='PUMFID', how='left')

            merged_chunks.append(chunk_merged)
            print(f"   Processed & Merged chunk {i + 1}...")

    except Exception as e:
        print(f"❌ Error reading Episode file: {e}")
        return None

    # --- STEP 3: Concatenate & Rename ---
    print("3. Concatenating all chunks...")
    full_df = pd.concat(merged_chunks, ignore_index=True)

    print("4. Renaming columns...")
    # rename_dict handles mapping Demographics (PRV->PR) and ID (PUMFID->occID)
    full_df = full_df.rename(columns=rename_dict)

    # --- STEP 4: Save ---
    print(f"5. Saving merged data to {output_csv_path}...")
    full_df.to_csv(output_csv_path, index=False)

    # --- STEP 5: Print Unique Values ---
    print("\n6. Unique Values Check:")
    print("-" * 40)
    for col in full_df.columns:
        # Basic check to avoid crashing on unhashable types
        try:
            unique_vals = full_df[col].unique()
            count = len(unique_vals)

            # If less than 20 unique values, print all. Otherwise, truncate.
            if count <= 20:
                print(f"[{col}] ({count}): {unique_vals}")
            else:
                print(f"[{col}] ({count}): {unique_vals[:5]} ... (truncated)")
        except Exception:
            print(f"[{col}] (Check skipped - likely unhashable)")

    print("-" * 40)

    print(f"✅ Success! Saved {len(full_df)} rows (Episodes).")
    return full_df

def data_alignment(census_csv_path: pathlib.Path, gss_csv_path: pathlib.Path,
                   output_dir: pathlib.Path = OUTPUT_DIR_ALIGNED):
    """
    Loads Census Forecast and GSS Library.
    Applies all harmonization functions to align columns and categories.
    Saves the resulting aligned DataFrames to CSV.
    Returns aligned DataFrames ready for matching.
    """
    print("--- Step 1: Loading Datasets for Alignment ---")

    # 1. Read Forecasted Census
    print(f"1. Reading Census Forecast: {census_csv_path.name}...")
    df_census = pd.read_csv(census_csv_path)
    print(f"   Census Loaded: {len(df_census)} rows.")

    # 2. Read Merged GSS
    print(f"2. Reading GSS Library: {gss_csv_path.name}...")
    # Use dtype=str to safely handle all codes before harmonization
    df_gss = pd.read_csv(gss_csv_path, dtype=str, low_memory=False)
    print(f"   GSS Loaded: {len(df_gss)} rows.")

    print("\n--- Step 2: Running Harmonization Pipeline ---")

    # Call each function sequentially
    df_census, df_gss = harmonize_agegrp(df_census, df_gss)
    df_census, df_gss = harmonize_hhsize(df_census, df_gss)
    df_census, df_gss = harmonize_hrswrk(df_census, df_gss)
    df_census, df_gss = harmonize_marsth(df_census, df_gss)
    df_census, df_gss = harmonize_sex(df_census, df_gss)
    df_census, df_gss = harmonize_kol(df_census, df_gss)
    df_census, df_gss = harmonize_nocs(df_census, df_gss)
    df_census, df_gss = harmonize_pr(df_census, df_gss)
    df_census, df_gss = harmonize_cow(df_census, df_gss)
    df_census, df_gss = harmonize_mode(df_census, df_gss)

    print(f"--- Alignment Complete. ---")
    print(f"   Census Shape: {df_census.shape}")
    print(f"   GSS Shape:    {df_gss.shape}")

    # --- Step 3: Save Aligned Data to CSV ---
    print("\n--- Step 3: Saving Aligned DataFrames ---")

    # Define file names
    census_out_file = output_dir / "Aligned_Census_2025.csv"
    gss_out_file = output_dir / "Aligned_GSS_2022.csv"

    print(f"   Saving Census to: {census_out_file}...")
    df_census.to_csv(census_out_file, index=False)

    print(f"   Saving GSS to:    {gss_out_file}...")
    df_gss.to_csv(gss_out_file, index=False)

    return df_census, df_gss

# HARMONIZE SUB-FUNCTIONS-----------------------------------------------------------------------------------------------
def harmonize_agegrp(df_census, df_gss):
    """
    Harmonizes AGEGRP.
    Assumes both DFs have a column named 'AGEGRP'.
    """
    # --- GSS Preparation ---
    # Filter bad values (97, 98, 99) from GSS 'AGEGRP'
    # Keep 96 (Valid Skip) as it maps to children
    df_gss = df_gss[~df_gss['AGEGRP'].isin(['97', '98', '99', 97, 98, 99])].copy()
    df_gss['AGEGRP'] = pd.to_numeric(df_gss['AGEGRP'], errors='coerce').fillna(96).astype(int)

    # --- Census Mapping ---
    def map_census_age_to_gss(x):
        try:
            x = int(float(x))
        except:
            return 96

        if x <= 2: return 96  # 0-14 -> Skip
        if x in [3, 4]: return 1  # 15-24
        if x in [5, 6]: return 2  # 25-34
        if x in [7, 8]: return 3  # 35-44
        if x in [9, 10]: return 4  # 45-54
        if x == 11: return 5  # 55-64
        if x == 12: return 6  # 65-74
        if x >= 13: return 7  # 75+
        return 96

    df_census['AGEGRP'] = df_census['AGEGRP'].apply(map_census_age_to_gss).astype(int)
    df_census = df_census[~df_census['AGEGRP'].isin([96])].copy()
    return df_census, df_gss

def harmonize_hhsize(df_census, df_gss):
    """
    Harmonizes HHSIZE.
    Assumes both DFs have a column named 'HHSIZE'.
    """
    # --- GSS Preparation ---

    # 1. Convert to numeric first (Handles '2.0', '2', etc.)
    # errors='coerce' turns non-numbers into NaN
    df_gss['HHSIZE'] = pd.to_numeric(df_gss['HHSIZE'], errors='coerce')

    # 2. Filter: Valid values are 1, 2, 3, 4, 5.
    # This automatically removes 6, 7, 8, 9, and NaNs.
    df_gss = df_gss[df_gss['HHSIZE'].isin([1, 2, 3, 4, 5])].copy()

    # 3. Convert to clean Integer
    df_gss['HHSIZE'] = df_gss['HHSIZE'].astype(int)

    # --- Census Mapping ---
    # Cap at 5 (matches GSS max category "5 or more")
    df_census['HHSIZE'] = df_census['HHSIZE'].apply(lambda x: 5 if int(float(x)) >= 5 else int(float(x)))

    return df_census, df_gss

def harmonize_hrswrk(df_census, df_gss):
    """
    Harmonizes HRSWRK.
    Assumes both DFs have a column named 'HRSWRK'.
    """
    # --- GSS Preparation ---
    # Remove 97, 98, 99 (Don't know/Refusal/Not stated)
    df_gss = df_gss[~df_gss['HRSWRK'].isin(['97', '98', '99', 97, 98, 99])].copy()

    def map_gss_hours(x):
        try:
            x = int(float(x))
        except:
            return 99

        if x == 96: return 99  # Valid Skip -> 99 (Not Applicable)
        if x == 1: return 1  # Under 30
        if 2 <= x <= 7: return 2  # 30-59
        if x == 8: return 3  # 60+
        return 99

    df_gss['HRSWRK'] = df_gss['HRSWRK'].apply(map_gss_hours).astype(int)

    # --- Census Mapping ---
    def map_census_hours(x):
        try:
            x = int(float(x))
        except:
            return 99

        if x == 99: return 99  # Not Applicable
        if x == 0: return 99  # No hours
        if 1 <= x <= 3: return 1  # 1-29 hours -> 1
        if 4 <= x <= 7: return 2  # 30-59 hours -> 2
        if x >= 8: return 3  # 60+ hours -> 3
        return 99

    df_census['HRSWRK'] = df_census['HRSWRK'].apply(map_census_hours).astype(int)

    return df_census, df_gss

def harmonize_marsth(df_census, df_gss):
    # GSS Preparation (Remove 96-99)
    df_gss = df_gss[~df_gss['MARSTH'].isin(['96', '97', '98', '99', 96, 97, 98, 99])].copy()

    def map_gss_marsth(x):
        try:
            x = int(float(x))
        except:
            return 99

        if x == 3: return 1  # Never married
        if x in [1, 2]: return 2  # Married / Common-law
        if x in [4, 5, 6]: return 3  # Sep / Div / Wid
        return 99

    df_gss['MARSTH'] = df_gss['MARSTH'].apply(map_gss_marsth).astype(int)
    df_gss = df_gss[df_gss['MARSTH'] != 99]  # Cleanup

    # Census Ensure Type
    df_census['MARSTH'] = df_census['MARSTH'].astype(int)

    return df_census, df_gss

def harmonize_sex(df_census, df_gss):
    """
    Harmonizes SEX.
    Census: 1=Female, 2=Male
    GSS: 1=Male, 2=Female
    Action: Filter valid (1,2) and SWAP values (1->2, 2->1).
    """
    # --- GSS Preparation ---

    # 1. Convert to numeric (Handling '1.0', '2.0', etc.)
    df_gss['SEX'] = pd.to_numeric(df_gss['SEX'], errors='coerce')

    # 2. Filter: Keep only 1 and 2 (Removes 6, 7, 8, 9, NaN)
    df_gss = df_gss[df_gss['SEX'].isin([1, 2])].copy()

    # 3. Swap values to match Census
    # 1 (Men) -> 2 (Male)
    # 2 (Women) -> 1 (Female)
    df_gss['SEX'] = df_gss['SEX'].map({1: 2, 2: 1}).astype(int)

    # --- Census Preparation ---
    df_census['SEX'] = df_census['SEX'].astype(int)

    return df_census, df_gss

def harmonize_kol(df_census, df_gss):
    # --- GSS Preparation ---
    # Convert to numeric first to handle '1.0', '2.0' etc.
    df_gss['KOL'] = pd.to_numeric(df_gss['KOL'], errors='coerce')

    # Filter: Keep 1, 2, 3, 4 (Removes 6, 7, 8, 9 and NaN)
    df_gss = df_gss[df_gss['KOL'].isin([1, 2, 3, 4])].copy()
    df_gss['KOL'] = df_gss['KOL'].astype(int)

    # --- Census Preparation ---
    df_census['KOL'] = df_census['KOL'].astype(int)

    return df_census, df_gss

def harmonize_nocs(df_census, df_gss):
    # --- GSS Preparation ---
    # Convert to numeric
    df_gss['NOCS'] = pd.to_numeric(df_gss['NOCS'], errors='coerce')

    # Filter: Remove 97, 98, 99 (Keep 1-10, 95, 96)
    df_gss = df_gss[~df_gss['NOCS'].isin([97, 98, 99]) & df_gss['NOCS'].notna()].copy()

    # Map 96 (Skip) and 95 (Uncodable) to 99 (Not Applicable)
    df_gss['NOCS'] = df_gss['NOCS'].replace({96: 99, 95: 99}).astype(int)

    # --- Census Preparation ---
    df_census['NOCS'] = df_census['NOCS'].astype(float).astype(int)

    return df_census, df_gss

def harmonize_pr(df_census, df_gss):
    # --- GSS Preparation ---
    # Convert to numeric
    df_gss['PR'] = pd.to_numeric(df_gss['PR'], errors='coerce')

    # Filter: Remove 96, 97, 98, 99
    df_gss = df_gss[~df_gss['PR'].isin([96, 97, 98, 99]) & df_gss['PR'].notna()].copy()

    def map_gss_region(x):
        if x in [10, 11, 12, 13]: return 1  # Eastern Canada
        if x == 24: return 2  # Quebec
        if x == 35: return 3  # Ontario
        if x in [46, 47, 48]: return 4  # Prairies
        if x == 59: return 5  # British Columbia
        if x in [60, 61, 62]: return 6  # Northern Canada (if present)
        return 99

    df_gss['PR'] = df_gss['PR'].apply(map_gss_region).astype(int)
    # Filter out unmapped
    df_gss = df_gss[df_gss['PR'] != 99]

    # --- Census Preparation ---
    df_census['PR'] = df_census['PR'].astype(int)

    return df_census, df_gss

def harmonize_cow(df_census, df_gss):
    """
    Harmonizes the 'COW' (Class of Worker) column.
    Target Alignment:
        1: Employee
        2: Self-employed
        3: Unpaid family worker
    """

    # --- GSS Preparation ---
    # 1. Convert to numeric, coercing errors to NaN
    df_gss['COW'] = pd.to_numeric(df_gss['COW'], errors='coerce')

    # 2. Filter: Keep only 1, 2, 3.
    # Instruction: "delete 6, 7, 8, 9" (and drop NaNs)
    df_gss = df_gss[df_gss['COW'].isin([1, 2, 3])].copy()

    # 3. Ensure integer type
    df_gss['COW'] = df_gss['COW'].astype(int)

    # --- Census Mapping ---
    # Census Raw:
    # 1=Employee, 2=Self-emp(no help), 3=Self-emp(with help), 4=Unpaid family worker

    # 1. Convert to numeric first
    df_census['COW'] = pd.to_numeric(df_census['COW'], errors='coerce')

    # 2. Define Mapping Dictionary
    # Combine 2 & 3 -> 2 (Self-employed)
    # Convert 4 -> 3 (Unpaid family worker)
    census_map = {
        1: 1,
        2: 2,
        3: 2,
        4: 3
    }

    # 3. Apply Map
    df_census['COW'] = df_census['COW'].map(census_map)

    # 4. Filter: Drop rows that didn't map (if any) and ensure Int
    df_census = df_census.dropna(subset=['COW']).copy()
    df_census['COW'] = df_census['COW'].astype(int)

    return df_census, df_gss

def harmonize_mode(df_census, df_gss):
    """
    Harmonizes MODE (Commuting).
    Target Categories: 1=Bike, 2=Driver, 3=Walk, 4=Transit, 5=Other, 9=NA
    """
    # --- GSS Preparation ---
    # Convert to numeric
    df_gss['MODE'] = pd.to_numeric(df_gss['MODE'], errors='coerce')

    # Filter: Remove 97, 98, 99
    df_gss = df_gss[~df_gss['MODE'].isin([97, 98, 99]) & df_gss['MODE'].notna()].copy()

    def map_gss_mode(x):
        if x == 5: return 1  # Bicycle
        if x == 1: return 2  # Driver
        if x == 4: return 3  # Walk
        if x == 3: return 4  # Transit
        if x in [2, 6]: return 5  # Passenger(2), Other(6) -> Other
        if x == 96: return 9  # Valid Skip -> NA
        return 9

    df_gss['MODE'] = df_gss['MODE'].apply(map_gss_mode).astype(int)

    # --- Census Mapping ---
    def map_census_mode(x):
        try:
            x = int(float(x))
        except:
            return 9

        if x == 1: return 1  # Bicycle
        if x == 2: return 2  # Driver
        if x == 7: return 3  # Walked -> Walk
        if x == 6: return 4  # Public Transit
        if x in [3, 4, 5]: return 5  # Motorcycle(3), Other(4), Passenger(5) -> Other
        if x == 9: return 9  # Not Applicable
        return 9

    df_census['MODE'] = df_census['MODE'].apply(map_census_mode).astype(int)
    # Filter: Remove 97, 98, 99
    df_census = df_census[~df_census['MODE'].isin([9]) & df_gss['MODE'].notna()].copy()

    return df_census, df_gss

# BALANCE & CHECK ------------------------------------------------------------------------------------------------------
def check_value_alignment(df1, df2, df1_name="Census", df2_name="GSS", target_cols=None):
    """
    Compares the unique values of SPECIFIC columns (Demographics) that exist in both DataFrames.
    Filters for: HHSIZE, HRSWRK, AGEGRP, MARSTH, SEX, KOL, NOCS, PR, LFTAG, MODE.
    Prints a summary table and a detailed report including ALL unique values.
    """
    # Filter df1 to only valid target columns
    cols1 = [c for c in target_cols if c in df1.columns]
    df1_sub = df1[cols1].copy()

    # Filter df2 to only valid target columns
    cols2 = [c for c in target_cols if c in df2.columns]
    df2_sub = df2[cols2].copy()

    print(f"\n{'=' * 80}")
    print(f"   DATA VALUE ALIGNMENT CHECK: {df1_name} vs {df2_name}")
    print(f"   (Restricted to {len(target_cols)} specific demographic columns)")
    print(f"{'=' * 80}")

    # Find common columns within the filtered subsets
    common_cols = sorted(list(set(df1_sub.columns).intersection(set(df2_sub.columns))))

    # Warn if targets are missing
    missing_targets = set(target_cols) - set(common_cols)
    if missing_targets:
        print(f"⚠️  Warning: The following target columns were NOT found in both datasets:")
        print(f"    {sorted(list(missing_targets))}")

    print(f"Analyzing {len(common_cols)} common columns...\n")

    results = []

    for col in common_cols:
        # Get unique values as sorted strings for robust comparison
        # Use the SUBSET dataframes
        u1 = sorted(df1_sub[col].dropna().astype(str).unique())
        u2 = sorted(df2_sub[col].dropna().astype(str).unique())

        set1 = set(u1)
        set2 = set(u2)

        match = set1 == set2

        # Calculate differences
        only_in_1 = sorted(list(set1 - set2))
        only_in_2 = sorted(list(set2 - set1))

        status = "✅ MATCH" if match else "❌ MISMATCH"

        results.append({
            "Column": col,
            "Status": status,
            f"Unique_{df1_name}": len(u1),
            f"Unique_{df2_name}": len(u2),
            f"Val_{df1_name}": u1,  # Store full list for detailed print
            f"Val_{df2_name}": u2,  # Store full list for detailed print
            f"Missing_in_{df2_name}": only_in_1,
            f"Missing_in_{df1_name}": only_in_2
        })

    if not results:
        print("❌ No common columns found from the target list.")
        return None

    # Convert to DataFrame for clean printing
    df_res = pd.DataFrame(results)

    # --- 1. Print Summary Table ---
    summary_cols = ["Column", "Status", f"Unique_{df1_name}", f"Unique_{df2_name}"]
    print(df_res[summary_cols].to_string(index=False))

    # --- 2. Print Detailed Report (Updated) ---
    print(f"\n\n{'=' * 80}")
    print(f"   🔍 DETAILED VALUE REPORT")
    print(f"{'=' * 80}")

    for _, row in df_res.iterrows():
        col = row['Column']
        status = row['Status']
        print(f"\nColumn: [{col}]  {status}")

        # Check if it looks like a continuous variable (too many values)
        if row[f"Unique_{df1_name}"] > 20 or row[f"Unique_{df2_name}"] > 20:
            print("   (Continuous/High-Cardinality variable detected)")
            # Handle potential string/float conversion for min/max
            try:
                # Use SUBSET dataframes
                min1, max1 = min(df1_sub[col]), max(df1_sub[col])
                min2, max2 = min(df2_sub[col]), max(df2_sub[col])
                print(f"   Range {df1_name}: {min1} to {max1}")
                print(f"   Range {df2_name}: {min2} to {max2}")
            except:
                print(f"   (Could not determine numeric range, likely string data)")
        else:
            # --- NEW: Always print unique values for visual check ---
            print(f"   {df1_name:<10} Values: {row[f'Val_{df1_name}']}")
            print(f"   {df2_name:<10} Values: {row[f'Val_{df2_name}']}")

            # Print specific differences if mismatch
            if status == "❌ MISMATCH":
                if row[f"Missing_in_{df2_name}"]:
                    print(f"   ⚠️ In {df1_name} ONLY: {row[f'Missing_in_{df2_name}']}")
                if row[f"Missing_in_{df1_name}"]:
                    print(f"   ⚠️ In {df2_name} ONLY: {row[f'Missing_in_{df1_name}']}")

    return df_res

def plot_distribution_comparison(df1, df2, df1_name="Census", df2_name="GSS", target_cols=None):
    """
    Plots side-by-side bar charts (normalized to percentage) for common columns.
    Useful for visually verifying if the distribution of GSS matches the Census.

    Parameters:
    - df1, df2: DataFrames to compare.
    - df1_name, df2_name: Labels for the legend (e.g., "Census", "GSS").
    - target_cols: List of column names to plot.
    """

    # --- SAFEGUARD: Check if data exists ---
    if df1 is None or df2 is None:
        print("❌ Error: One of the DataFrames is None. Cannot plot.")
        return

    # 1. Identify Common Columns from the Target List
    common_cols = sorted(list(set(target_cols).intersection(set(df1.columns)).intersection(set(df2.columns))))

    if not common_cols:
        print("❌ No common columns found to plot.")
        return

    print(f"📊 Plotting distributions for {len(common_cols)} columns...")

    # 2. Setup Plot Grid
    num_plots = len(common_cols)
    cols_per_row = 3
    rows = math.ceil(num_plots / cols_per_row)

    # Dynamic Figure Size: (Width, Height per row * rows)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 5 * rows))
    axes = axes.flatten()  # Flatten to 1D array for easy looping

    # 3. Loop through columns and plot
    for i, col in enumerate(common_cols):
        ax = axes[i]

        # Prepare temporary data for Seaborn
        # We merge just this column into a "Long Format" DF for easy plotting
        d1 = df1[[col]].dropna().copy()
        d1['Source'] = df1_name

        d2 = df2[[col]].dropna().copy()
        d2['Source'] = df2_name

        combined = pd.concat([d1, d2], ignore_index=True)

        # Determine if categorical (Discrete) or Continuous
        # Heuristic: If < 20 unique values, treat as distinct categories (bars)
        # If > 20 (like HRSWRK might be), let histogram bin it.
        n_unique = combined[col].nunique()
        is_discrete = n_unique < 25

        # PLOT
        # stat='percent' ensures we compare Proportions, not Raw Counts
        # common_norm=False ensures 100% is calculated per Group (Census=100%, GSS=100%)
        sns.histplot(
            data=combined,
            x=col,
            hue='Source',
            stat='percent',
            common_norm=False,
            multiple='dodge',  # 'dodge' puts bars side-by-side
            shrink=0.8,  # Adds valid spacing between bars
            discrete=is_discrete,
            ax=ax,
            palette={df1_name: "#1f77b4", df2_name: "#ff7f0e"}  # Blue vs Orange
        )

        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Percent (%)")
        ax.grid(axis='y', alpha=0.3)

    # 4. Hide empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# --- 4. Execution ---
if __name__ == "__main__":
    GSS_FILE = OUTPUT_DIR_GSS / "GSS_2022_Merged_Episodes.csv"
    """
    # read & merge GSS
    read_merge_save_gss(main_path=FILE_MAIN, episode_path=FILE_EPISODE, cols_main=COLS_MAIN, rename_dict=RENAME_MAP, output_csv_path=GSS_FILE)
    
    # Run Function
    df_census, df_gss = data_alignment(CENSUS_FILE, GSS_FILE)

    # --- STEP 0: Define Target Columns & Filter ---
    target_cols = ["HHSIZE", "HRSWRK", "AGEGRP", "MARSTH", "SEX", "KOL", "NOCS", "PR", "COW", "MODE"]
    check_value_alignment(df_census, df_gss, target_cols=target_cols)

    # --- Example Usage ---
    plot_distribution_comparison(df_census, df_gss, target_cols=target_cols)
    """
