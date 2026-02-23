import pandas as pd
import pyreadstat
import re, os
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjust as needed, or use None
#--------------------------------------- GEMINI - 2005
def load_spss_file(file_path, selected_columns=None, output_csv=None, printNan=False):
    print(f"Reading file: {file_path}...")

    if selected_columns is not None:
        df, meta = pyreadstat.read_sav(file_path, usecols=selected_columns)
    else:
        df, meta = pyreadstat.read_sav(file_path)
    # BALANCE - CHECK
    if printNan:
        print("Loaded shape:", df.shape)
        print("df_2005_episode", df.head(50))
        describe_unique_values(df, exclude_cols=["RECID", "PUMFID", "WGHT_PER"])
    else:
        pass
    save_df_to_csv(df, output_csv, num_rows=None)
    return df
#--------------------------------------- GEMINI - 2010
def load_dat_with_sps_layout(dat_file_path, sps_file_path, selected_columns=None, output_csv=None, printNan=False):
    """
    Reads a fixed-width .DAT file using SPSS .sps layout (DATA LIST),
    with optional column filtering via `selected_columns`.
    """
    var_regex = re.compile(r"^\s*/?\s*([a-zA-Z0-9_]+)\s+(\d+)\s+-\s+(\d+)")

    column_names = []
    col_specs = []

    print(f"Parsing syntax file: {sps_file_path}")

    with open(sps_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().upper().startswith('VARIABLE LABELS'):
                break  # Stop at label section

            match = var_regex.match(line)
            if match:
                name = match.group(1)
                start_pos = int(match.group(2))
                end_pos = int(match.group(3))

                # Append only if selection is not active OR name is in selection
                if selected_columns is None or name in selected_columns:
                    column_names.append(name)
                    col_specs.append((start_pos - 1, end_pos))

    print(f"Parsing complete. Loading {len(column_names)} column(s).")

    if not column_names:
        print("Error: No matching columns found. Check column names.")
        return None

    print(f"Loading data from: {dat_file_path}")
    df = pd.read_fwf(
        dat_file_path,
        colspecs=col_specs,
        names=column_names,
        dtype="str"
    )
    # BALANCE - CHECK
    if printNan:
        print("Data loaded successfully.")
        describe_unique_values(df, exclude_cols=["RECID"])
    else:
        pass

    save_df_to_csv(df, output_csv, num_rows=None)
    return df
#--------------------------------------- Claude - 2015
def parse_spss_syntax_selective(syntax_file, columns_to_keep=None):
    """
    Parse SPSS syntax file and extract only specified columns.

    Parameters:
    -----------
    syntax_file : str
        Path to the SPSS syntax (.sps) file
    columns_to_keep : list or None
        List of column names to extract. If None, extracts all columns.

    Returns:
    --------
    list : List of tuples (name, start, end, width, dtype) for selected columns
    """
    with open(syntax_file, 'r', encoding='latin-1') as f:
        content = f.read()

    # Extract DATA LIST section
    data_list_match = re.search(r'DATA LIST.*?/(.*?)(?:VARIABLE LABELS|VALUE LABELS|EXECUTE|\Z)',
                                content, re.DOTALL | re.IGNORECASE)

    if not data_list_match:
        raise ValueError("Could not find DATA LIST section in syntax file")

    data_list_section = data_list_match.group(1)

    # Parse variable definitions
    var_pattern = r'(\w+)\s+(\d+)\s*-\s*(\d+)(?:\s*\(([A\d]+)\))?'
    matches = re.findall(var_pattern, data_list_section)

    variables = []
    columns_to_keep_set = set(columns_to_keep) if columns_to_keep else None

    for var_name, start, end, format_spec in matches:
        # Skip if we're filtering and this column is not in the list
        if columns_to_keep_set and var_name not in columns_to_keep_set:
            continue

        start_pos = int(start) - 1  # Convert to 0-based
        end_pos = int(end)
        width = end_pos - start_pos

        # Determine dtype
        if format_spec == 'A':
            dtype = 'str'
        elif format_spec and format_spec.isdigit():
            dtype = 'float'
        else:
            dtype = 'int'

        variables.append((var_name, start_pos, end_pos, width, dtype))

    return variables
def read_gss_data_selective(data_file, syntax_file, columns_to_keep=None, chunksize=10000,output_csv=None, printNan=False):
    # Parse syntax file for selected columns only
    variables = parse_spss_syntax_selective(syntax_file, columns_to_keep)

    if columns_to_keep:
        print(f"Reading {len(variables)} out of requested {len(columns_to_keep)} columns")
        missing = set(columns_to_keep) - {v[0] for v in variables}
        if missing:
            print(f"Warning: Columns not found in syntax file: {missing}")
    else:
        print(f"Reading all {len(variables)} columns")

    print("Reading data in chunks...")

    # Prepare column specifications
    colspecs = [(var[1], var[2]) for var in variables]
    names = [var[0] for var in variables]

    # Read in chunks - much faster with fewer columns!
    chunks = []
    for i, chunk in enumerate(pd.read_fwf(data_file,
                                          colspecs=colspecs,
                                          names=names,
                                          encoding='latin-1',
                                          chunksize=chunksize,
                                          dtype_backend='numpy_nullable')):
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * chunksize} rows...")

    print("Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)

    # Convert dtypes based on syntax
    print("Converting data types...")
    for var_name, _, _, _, dtype in variables:
        if dtype == 'float':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')
        elif dtype == 'int':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce').astype('Int64')

    # Reorder columns to match the requested order if specified
    if columns_to_keep:
        # Only keep columns that exist in both lists
        final_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[final_columns]

    # BALANCE - CHECK
    if printNan:
        print("Data loaded successfully.")
        describe_unique_values(df, exclude_cols=["PUMFID"])
    else:
        pass
    save_df_to_csv(df, output_csv, num_rows=None)

    return df
#--------------------------------------- GEMINI - 2022
def read_SAS(sas_file_path, columns_to_keep, chunk_size=100000, encoding='utf-8', output_csv=None, printNan=False):
    print(f"Reading SAS file in chunks: {sas_file_path}...")
    if not os.path.exists(sas_file_path):
        print(f"❌ Error: File not found at {sas_file_path}")
        return None

    def process_reader(reader, columns_to_keep):
        chunks = []
        print(f"Processing chunks and keeping only {len(columns_to_keep)} columns...")
        for chunk in reader:
            # Check which of the desired columns actually exist in this chunk
            cols_exist_in_chunk = [col for col in columns_to_keep if col in chunk.columns]
            if not cols_exist_in_chunk:
                print("Warning: None of the desired columns found in a chunk. Skipping.")
                continue

            # Filter the chunk to keep only the existing desired columns
            filtered_chunk = chunk[cols_exist_in_chunk]
            chunks.append(filtered_chunk)
        return chunks

    # --- End of helper function ---

    try:
        # --- EDIT 1: Try reading with the primary encoding ---
        print(f"Attempting to read with encoding: '{encoding}'")
        reader = pd.read_sas(
            sas_file_path,
            chunksize=chunk_size,
            iterator=True,
            encoding=encoding  # <-- This is the key addition
        )
        filtered_chunks = process_reader(reader, columns_to_keep)

    except UnicodeDecodeError:
        # --- EDIT 2: Add fallback encoding (e.g., 'latin-1') ---
        print(f"⚠️ Warning: '{encoding}' failed. Trying 'latin-1'...")
        try:
            reader = pd.read_sas(
                sas_file_path,
                chunksize=chunk_size,
                iterator=True,
                encoding='latin-1'  # <-- The fallback
            )
            print("✅ Read iterator created successfully with 'latin-1'.")
            filtered_chunks = process_reader(reader, columns_to_keep)

        except Exception as e:
            print(f"❌ Error loading SAS file with fallback encoding: {e}")
            return None

    except Exception as e:
        print(f"❌ Error loading or processing SAS file in chunks: {e}")
        return None

    # --- Combine chunks (same as before) ---
    if not filtered_chunks:
        print("❌ Error: No data loaded. Check if column names are correct.")
        return None

    print("Concatenating filtered chunks...")
    full_df = pd.concat(filtered_chunks, ignore_index=True)
    print("✅ Data loaded and filtered successfully.")

    # BALANCE - CHECK
    if printNan:
        print(full_df.head(10))
        describe_unique_values(full_df, exclude_cols=["PUMFID"])
    else:
        pass

    save_df_to_csv(full_df, output_csv, num_rows=None)
    return full_df
#--------------------------------------- EDITING
def load_map_and_save(df, columns_to_map, mapping, output_csv_path=None, printNan=False):

    # --- 2. Normalize and Check Columns ---
    # Convert input to a list, whether it's a string or a list
    if isinstance(columns_to_map, str):
        columns_list = [columns_to_map]
    elif isinstance(columns_to_map, list):
        columns_list = columns_to_map
    else:
        print("❌ Error: 'columns_to_map' must be a string or a list of strings.")
        return None

    # Check if all specified columns exist in the DataFrame
    missing_cols = [col for col in columns_list if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The following columns were not found: {missing_cols}")
        print(f"Available columns are: {df.columns.tolist()}")
        return None

    # --- 3. Invert the mapping dictionary (Done once) ---
    print("Inverting activity mapping dictionary...")
    # Convert the 'old_val' (e.g., "450") from the map to a float (e.g., 450.0)
    # to match the data type of the column after pd.to_numeric.
    inverted_map = {
        float(old_val): new_val
        for new_val, old_val_list in mapping.items()
        for old_val in old_val_list
    }
    # --- 4. Loop Through Each Column to Map and Report ---
    print("\n--- Mapping Values & Reporting ---")

    for col_name in columns_list:
        print(f"\nProcessing column: '{col_name}'...")
        # Store original string values for *this* column
        original_values_str = df[col_name].copy()
        # Convert column to numeric (turns non-numeric strings into NaN)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        # Apply the map (values not in map become NaN)
        df[col_name] = df[col_name].map(inverted_map)
        # Get a mask of all rows that are now NaN
        nan_mask = df[col_name].isnull()
        print("✅ Mapping complete.")
        # Report on unmapped values for *this* column
        print("--- Unmapped Values Report ---")
        unmapped_original_values = original_values_str[nan_mask].unique()
        if len(unmapped_original_values) > 0:
            print(f"❌ Found {len(unmapped_original_values)} original values in '{col_name}' that became NaN:")
            print(list(unmapped_original_values))
        else:
            print(f"✅ All values in '{col_name}' were successfully mapped or were already blank.")

    print("\n--- All columns processed. ---")

    # --- 5. Save the modified DataFrame to a new CSV ---
    print(f"Attempting to save modified data to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Successfully saved modified file to: {output_csv_path}")

    # BALANCE - CHECK
    if printNan:
        print_nan_counts(df)
    else:
        pass

    # --- 6. Return the DataFrame for further use ---
    return df
#--------------------------------------- MERGING
def merge_coPresence(df, merge_map, output_csv_path, rename_map=None):
    print(f"Starting batch merge for {len(merge_map)} new columns...")
    all_cols_to_drop = []

    # --- 1. Iterate through each merge job in the map ---
    for output_col_name, cols_to_merge in merge_map.items():

        print(f"Processing: {cols_to_merge} -> {output_col_name}")

        # --- 1a. Check if all columns exist ---
        missing_cols = [col for col in cols_to_merge if col not in df.columns]
        if missing_cols:
            print(f"❌ Error: The following columns were not found: {missing_cols}")
            return None

        # --- 1b. Apply the merge logic ---
        try:
            df_subset = df[cols_to_merge].apply(pd.to_numeric, errors='coerce')
            conditions = [
                (df_subset == 1).any(axis=1),
                (df_subset == 2).any(axis=1),
                (df_subset == 9).any(axis=1)
            ]
            choices = [1, 2, 9]
            df[output_col_name] = np.select(conditions, choices, default=np.nan)
            all_cols_to_drop.extend(cols_to_merge)

        except Exception as e:
            print(f"❌ Error during merge logic for '{output_col_name}': {e}")
            return None

    # --- 2. Drop all original columns ---
    try:
        unique_cols_to_drop = list(set(all_cols_to_drop))
        df = df.drop(columns=unique_cols_to_drop)
        print(f"\n✅ All merge operations complete.")
        print(f"✅ Dropped {len(unique_cols_to_drop)} original columns.")

    except Exception as e:
        print(f"❌ Error dropping merged columns: {e}")
        return None

    # --- 3. (NEW) Rename columns ---
    if rename_map:
        print("Renaming specified columns...")
        try:
            df = df.rename(columns=rename_map)
            print(f"✅ Columns renamed: {list(rename_map.keys())} -> {list(rename_map.values())}")
        except Exception as e:
            print(f"❌ Error renaming columns: {e}")
            return None

    # --- 4. Save the final DataFrame to a new CSV ---
    print(f"Attempting to save final data to: {output_csv_path}")
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_csv_path, index=False)
        print(f"✅ Successfully saved file to: {output_csv_path}")
    except Exception as e:
        print(f"❌ Error saving new CSV: {e}")
        return None

    print(df.head(50))

    # --- 5. Return the DataFrame ---
    return df
#--------------------------------------- EXTRA
def describe_unique_values(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        uniques = df[col].unique()
        print(f"\n--- Column: {col} ---")
        print(f"Unique count: {len(uniques)}")
        print("Unique values:", uniques)
def save_df_to_csv(df, csv_file_path, num_rows=None):
    print(f"Attempting to save data to: {csv_file_path}")

    if num_rows is not None and isinstance(num_rows, int) and num_rows > 0:
        # Save only the first 'num_rows'
        rows_to_save = df.head(num_rows)
        rows_to_save.to_csv(csv_file_path, index=False)
        print(f"✅ Successfully saved the first {len(rows_to_save)} rows to CSV.")
    elif num_rows is None:
        # Save the entire DataFrame
        df.to_csv(csv_file_path, index=False)
        print(f"✅ Successfully saved all {len(df)} rows to CSV.")
    else:
         print("❌ Error: 'num_rows' must be a positive integer or None.")
         return
def print_nan_counts(df):
    nan_counts_all = df.isnull().sum()
    # Filter this list to get only columns that *have* missing values
    nan_counts_filtered = nan_counts_all[nan_counts_all > 0]
    # --- 3. Print the Results ---
    print("\n--- NaN (Missing) Value Counts ---")
    if nan_counts_filtered.empty:
        print("✅ No missing (NaN) values found in any column.")
    else:
        # Ensure pandas prints all columns/rows if the list is long
        original_max_rows = pd.get_option('display.max_rows')
        try:
            pd.set_option('display.max_rows', None)  # Temporarily allow unlimited rows
            print(nan_counts_filtered)
        finally:
            pd.set_option('display.max_rows', original_max_rows)  # Reset to default
    print("converted_dataframe")
    print(df.head(50))

    return df

if __name__ == '__main__':
    """
    C19PUMFM_NUM.SAV: This is the Main file containing the core socio-demographic data and the 24-hour time-use diary for all survey respondents.
    C19PUMFE_NUM.SAV: This is the Extended file containing the split-sample variables (e.g., culture, sports, social networks, transportation) that were asked of only a random subset of respondents.
    """
    GSS_2005_SPSS_full = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2005/Data Files SPSS/C19PUMFM_NUM.SAV"
    GSS_2005_SPSS_episode = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2005/Data Files SPSS/C19PUMFE_NUM.SAV"

    GSS_2010_SPSS_episode = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2010/Data_Donn‚es/C24EPISODE_withno_bootstrap.DAT"
    sps_syntax_2010 = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2010/Syntax_Syntaxe/SPSS/C24_Episode File_SPSS_withno_bootstrap.SPS"

    GSS_2015_SPSS_episode = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2015/c29_2015/Data_Donn‚es/GSS29PUMFE.txt"
    sps_syntax_2015 = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2015/c29_2015/Syntax_Syntaxe/Episode/SPSS/c29pumfe_e.sps"

    GSS_2022_SPSS_episode = "/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Data Sources/Canada_2022/Data_Données/TU_ET_2022_Episode_PUMF.sas7bdat"

    ####################################################################################################################
    # READING
    """
    # 2010 - GPT
    df_2005_episode = load_spss_file(GSS_2005_SPSS_episode,
                                     selected_columns=["RECID", "EPINO", "WGHT_EPI","ACTCODE", "STARTIME", "ENDTIME", "PLACE", "ALONE",
                                                       "SPOUSE", "CHILDHSD", "FRIENDS", "OTHFAM", "NHSDCL15", "NHSDC15P", "OTHERS", "PARHSD", "NHSDPAR", "MEMBHSD"],
                                     printNan=False)

    # EDITING - OCCUPANT ACTIVITY
    modified_df_2005 = load_map_and_save(df_2005_episode,
                                         columns_to_map="ACTCODE",
                                         mapping={1: [2, 11, 12, 21, 22, 23, 40, 50, 60, 70, 80, 600, 832, 842], 13: [30, 90, 190, 291, 292, 390, 491, 492, 590, 674, 691, 692, 791, 792, 793, 871, 872, 873, 891, 892, 893, 894, 990],
                                        2: [101, 102, 110, 120, 130, 140, 151, 152, 161, 162, 163, 164, 171, 172, 173, 181, 182, 183, 184, 185, 186],
                                        3: [200, 211, 212, 213, 220, 230, 240, 250, 260, 271, 272, 281, 282, 671, 672, 673, 675, 676, 677, 678],
                                        4: [301, 302, 303, 304, 310, 320, 331, 332, 340, 350, 361, 362, 370, 380], 7: [400, 410, 411, 480], 6: [430, 431], 9: [440, 751, 752, 753, 754, 760, 770, 780], 5: [450, 460, 470],
                                        8: [500, 511, 512, 520, 530, 540, 550, 560, 580], 12: [610, 620, 630, 640, 642, 651, 652, 660, 661, 680, 800],
                                        10: [701, 702, 711, 712, 713, 720, 730, 741, 742, 743, 831, 841, 850, 861, 862, 863, 864, 865, 866, 867, 880, 900, 911, 912, 913, 914, 920, 931, 932, 940, 950, 951, 961, 962, 980, 995],
                                        11: [801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 821, 822]},
                                         printNan=False)

    #----------------
    # EDITING - PRESENCE
    df05EP_ACT_PRE_convert = load_map_and_save(modified_df_2005,
                                               columns_to_map="PLACE",
                                               mapping={1: ["1"], 2: ["2", "8"], 3: ["3"], 4: ["9"], 5: ["6", "7"], 6: ["10"], 7: ["4"], 8: ["5"], 9: ["11"], 10: ["12"], 11: ["13"],12: ["14"], 13: ["15", "16", "18"], 14: ["20"], 15: ["17"], 16: ["19"], 17: ["21"], 18: ["97", "98", "99"]},
                                               printNan=False)

    # ----------------
    # EDITING - CO-PRESENCE
    # CONVERSION
    df05EP_ACT_PRE_coPRE_convert = load_map_and_save(df05EP_ACT_PRE_convert,
                                                     columns_to_map=["ALONE", "SPOUSE", "CHILDHSD", "FRIENDS", "OTHFAM", "NHSDCL15", "NHSDC15P", "OTHERS", "PARHSD", "NHSDPAR", "MEMBHSD"],
                                                     mapping={1:[1], 2:[2], 9: [7,8,9]},
                                                     printNan=False)

    # MERGING
    df05EP_ACT_PRE_coPRE_complete = merge_coPresence(df05EP_ACT_PRE_coPRE_convert,
                                                     merge_map={"otherHHs": ["OTHFAM", "NHSDCL15", "NHSDC15P"],"parents": ["PARHSD", "NHSDPAR"]},
                                                     output_csv_path="/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Outputs/out05EP_ACT_PRE_coPRE.csv",
                                                     rename_map={"RECID":"occID", "ACTCODE":"occACT", "STARTIME":"start", "ENDTIME":"end", "PLACE":"occPRE",
                                                                 "ALONE": "Alone","SPOUSE": "Spouse","CHILDHSD": "Children","MEMBHSD": "otherInFAMs", "FRIENDS": "Friends", "OTHERS": "Others"})

    ####################################################################################################################
    #2010 - gemini
    # READING
    df_2010_episode = load_dat_with_sps_layout(GSS_2010_SPSS_episode, sps_syntax_2010,
                                               selected_columns=["RECID", "EPINO", "WGHT_EPI","ACTCODE", "STARTIME", "ENDTIME", "PLACE", "ALONE", "SPOUSE", "CHILDHSD",
                                                                 "FRIENDS", "OTHFAM", "NHSDCL15", "NHSDC15P", "OTHERS", "PARHSD", "NHSDPAR", "MEMBHSD"],
                                               printNan=False)
    # ----------------
    # EDITING - OCC ACTIVITY
    modified_df_2010 = load_map_and_save(df_2010_episode,
                                         columns_to_map="ACTCODE",
                                         mapping={1: ["2", "11", "12", "21", "22", "23", "40", "50", "60", "70", "80.1", "80.2", "80.3", "80.9", "600", "832", "842"], 13: ["30", "90", "190", "291", "292", "390", "491", "492", "590", "674", "691", "692", "791", "792", "793", "871", "872", "873", "891", "892", "893", "894", "990"],
                                            2: ["101", "102", "110", "120", "130", "140", "151", "152", "161", "162", "163", "164", "171.1", "171.2", "172", "173", "181.1", "181.2", "181.3", "182", "183", "184", "185", "186", "671.2"],
                                            3: ["200.1", "200.2", "200.3", "211", "212", "213", "220", "230.1", "230.2", "240", "250.1", "250.2","260", "260.1", "271.1", "271.2", "271.3", "272.1", "272.2", "281.1", "281.2", "281.4", "281.5", "281.8", "281.9", "282.1", "282.2", "282.9", "671.1", "672", "673.1", "673.2", "673.3", "673.4", "673.5", "673.9", "675.1", "675.2", "675.3", "675.4", "675.9", "676", "677", "678"],
                                            4: ["301", "302.1", "302.2", "302.3", "302.4", "302.9", "303", "304", "310.1", "310.2", "310.3", "320", "331", "332.1", "332.2", "340.1", "340.2", "350.1", "350.2", "350.3", "350.9", "361", "362", "370", "380.1", "380.2", "380.3", "380.4", "380.9"],
                                            7: ["400", "410.1", "410.2", "410.3", "411", "480"], 6: ["430", "431"], 9: ["440", "751", "752", "753", "754", "760", "770", "780.1", "780.2"], 5: ["450", "460", "470"],
                                            8: ["500", "511", "512", "520", "530.1", "530.2", "540", "550", "560.1", "560.2", "580.1", "580.9"], 12: ["610", "620", "630", "640", "642", "651", "652", "660.1", "660.2", "660.3", "660.4", "660.5", "660.9", "661", "680.1", "680.2", "800"],
                                            10: ["701", "702", "711", "712", "713",  "720", "730", "741", "742", "743", "831", "841", "850.1", "850.2", "861", "862", "862.1", "862.2", "863", "864", "865", "866", "867.1", "867.9", "880", "900.1", "900.2", "911", "912", "913", "914.1", "914.9", "920", "931", "932.1", "932.2", "940.1", "940.2", "950", "951", "951.1",  "951.2", "951.3", "961", "962", "980.1", "980.9", "995"],
                                            11: ["801.1", "801.2", "801.3", "801.4", "801.5", "801.6", "801.7", "801.8", "802.1", "802.2", "803.1", "803.2", "804.1", "804.2", "805.1", "805.2", "805.3", "806.1", "806.2", "807.1", "807.2", "807.3", "807.4", "808", "809", "810", "810.9",'810.1', "811", "812", "813", "814", "815", "816", "821.1", "821.2", "821.3", "822"]},
                                         printNan=False)
    # ----------------
    # EDITING - PRESENCE
    df10EP_ACT_PRE_convert = load_map_and_save(modified_df_2010,
                                               columns_to_map="PLACE",
                                               mapping={1: ["1"], 2: ["2", "8"], 3: ["3"], 4: ["9"], 5: ["6", "7"], 6: ["10"], 7: ["4"], 8: ["5"], 9: ["11"], 10: ["12"],
                                                        11: ["13"], 12: ["14"], 13: ["15", "16", "18"], 14: ["20"], 15: ["17"], 16: ["19"], 17: ["21"], 18: ["97", "98", "99"]},
                                               printNan=False)
    # ----------------
    # EDITING - CO-PRESENCE
    # CONVERSION
    df10EP_ACT_PRE_coPRE_convert = load_map_and_save(df10EP_ACT_PRE_convert,
                                                     columns_to_map=["ALONE", "SPOUSE", "CHILDHSD", "FRIENDS", "OTHFAM", "NHSDCL15", "NHSDC15P", "OTHERS", "PARHSD", "NHSDPAR", "MEMBHSD"],
                                                     mapping={1:[1], 2:[2], 9: [7,8,9]},
                                                     printNan=False)

    # MERGING
    df10EP_ACT_PRE_coPRE_complete = merge_coPresence(df10EP_ACT_PRE_coPRE_convert,
                                                     merge_map={"otherHHs": ["OTHFAM", "NHSDCL15", "NHSDC15P"],"parents": ["PARHSD", "NHSDPAR"]},
                                                     output_csv_path="/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Outputs/out10EP_ACT_PRE_coPRE.csv",
                                                     rename_map={"RECID":"occID", "ACTCODE":"occACT", "STARTIME":"start", "ENDTIME":"end", "PLACE":"occPRE",
                                                         "ALONE": "Alone","SPOUSE": "Spouse","CHILDHSD": "Children","MEMBHSD": "otherInFAMs", "FRIENDS": "Friends", "OTHERS": "Others"})

    ####################################################################################################################
    # 2015 - Claude
    """"""
    # READING
    df_2015_episode = read_gss_data_selective(GSS_2015_SPSS_episode, sps_syntax_2015,
                                              columns_to_keep=['PUMFID', 'EPINO', 'WGHT_EPI', 'TOTEPISO', 'TUI_01', 'STARTIME', 'ENDTIME', 'LOCATION', 'TUI_06A', 'TUI_06B', 'TUI_06C', 'TUI_06D', 'TUI_06E', 'TUI_06F', 'TUI_06G', 'TUI_06H', 'TUI_06I','TUI_06J'],
                                              chunksize=10000,
                                              printNan=False)
    # ----------------
    # EDITING - OCCUPANT ACTIVITY
    modified_df_2015 = load_map_and_save(df_2015_episode,
                                         columns_to_map="TUI_01",
                                         mapping= { 5: ["1"], 7: ["2", "3", "4"], 2: ["5", "18", "19", "20", "21", "22", "23", "24", "25", "26"], 6: ["6"], 13: ["7"], 1: ["8", "9", "10", "11", "12", "40"], 8: ["13", "14", "15", "16", "17"],
                                                    3: ["27", "28", "29", "30", "31", "32", "33", "34", "35", "36"], 4: ["37", "38", "39"], 9: ["41", "42"], 12: ["43", "44", "45", "46", "52"], 11: ["47", "48", "49", "50", "51"],
                                                    10: ["53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63"], 14: ["95"]},
                                         printNan=False)
    # ----------------
    # EDITING - PRESENCE
    df15EP_ACT_PRE_convert = load_map_and_save(modified_df_2015,
                                               columns_to_map="LOCATION",
                                               mapping={1: ["300"], 2: ["301", "302"], 3: ["303"], 4: ["305", "304"], 5: ["306"], 6: ["307"], 7: ["309"], 8: ["310"], 9: ["312", "308", "311"], 10: ["313"], 11: ["314"], 12: ["315"], 13: ["316", "320"], 14: ["317"], 15: ["318"],
                                                        16: ["319"], 17: ["321"], 18: ["996", "997", "998", "999"]},
                                               printNan=False)
    # ----------------
    # EDITING - CO-PRESENCE
    # CONVERSION
    df15EP_ACT_PRE_coPRE_convert = load_map_and_save(df15EP_ACT_PRE_convert,
                                                     columns_to_map=["TUI_06A", "TUI_06B", "TUI_06C", "TUI_06H", "TUI_06I", "TUI_06G", "TUI_06J", "TUI_06E", "TUI_06D", "TUI_06F"],
                                                     mapping={1:[1], 2:[2], 9: [9]},
                                                     printNan=False)

    # MERGING
    df15EP_ACT_PRE_coPRE_complete = merge_coPresence(df15EP_ACT_PRE_coPRE_convert,
                                                     merge_map={"otherInFAMs": ["TUI_06D", "TUI_06F"], "friends": ["TUI_06H", "TUI_06I"]},
                                                     output_csv_path="/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Outputs/out15EP_ACT_PRE_coPRE.csv",
                                                     rename_map={"PUMFID":"occID", "TUI_01":"occACT", "STARTIME":"start", "ENDTIME":"end", "LOCATION":"occPRE",
                                                         "TUI_06A": "Alone", "TUI_06B": "Spouse", "TUI_06C": "Children", "TUI_06G": "otherHHs", "TUI_06J":"others", "TUI_06E":"parents"})

    ####################################################################################################################
    
    """
    # 2022 - gemini
    """
    # READING
    df_2022_episode = read_SAS(GSS_2022_SPSS_episode, columns_to_keep= ['PUMFID', 'INSTANCE', 'WGHT_EPI', 'ENDTIME', 'LOCATION', 'STARTIME', 'TUI_01', 'TUI_06A', 'TUI_06B', 'TUI_06C', 'TUI_06D', 'TUI_06E','TUI_06F', 'TUI_06G', 'TUI_06H', 'TUI_06I', 'TUI_06J'],
                                 chunk_size=100000,
                               printNan=False)
    

    # ----------------
    # EDITING - OCCUPANT ACTIVITY
    modified_df_2022 = load_map_and_save(df_2022_episode,
                                         columns_to_map="TUI_01",
                                         mapping={1: ["153", "501", "502", "503", "504", "505", "506", "507", "599"],  2: ["353", "201", "202", "203", "204", "205", "206", "207", "208", "209", "299", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241"],
                                        3: ["301", "302", "303", "304", "305", "306", "307", "399", "351", "352", "359"], 4: ["261", "262", "263", "264", "269"], 5: ["101", "102", "103", "104", "109"], 6: ["151", "152", "159"], 7: ["126", "127", "128", "129", "130", "199"],
                                        8: ["154", "601", "602", "603", "604", "699"], 9: ["701", "702", "799"], 10: ["1101", "1102", "1103", "1104", "1199", "1201", "1202", "1203", "1204", "1299"], 11: ["1001", "1002", "1003", "1004", "1005", "1105", "1099", "1106"],
                                        12: ["801", "802", "803", "804", "805", "806", "807", "808", "899", "901", "902", "903", "999"], 13: ["401", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "499"],
                                        14: ["1301", "1302", "1303", "1304", "9999"]},
                                         printNan=False)
    # ----------------
    # EDITING - PRESENCE
    df22EP_ACT_PRE_convert = load_map_and_save(modified_df_2022,
                                               columns_to_map="LOCATION",
                                               mapping={1: ["3300"], 2: ["3301", "3302"], 3: ["3303"], 4: ["3305", "3304"], 5: ["3306"], 6: ["3307"], 7: ["3309"], 8: ["3310"],
                                                9: ["3312", "3308", "3311"], 10: ["3313"], 11: ["3314"], 12: ["3315"], 13: ["3316"], 14: ["3317"], 15: ["3318"],
                                                16: ["3320"], 17: ["3323", "3399", "3319"], 18: ["9996", "9997", "9998", "9999"]},
                                               printNan=False)
                                               
    

    # ----------------
    # EDITING - CO-PRESENCE
    # CONVERSION
    df22EP_ACT_PRE_coPRE_convert = load_map_and_save(df22EP_ACT_PRE_convert,
                                                     columns_to_map= ["TUI_06A", "TUI_06B", "TUI_06C", "TUI_06H", "TUI_06I", "TUI_06G", "TUI_06J", "TUI_06E", "TUI_06D", "TUI_06F"],
                                                     mapping={1:[1], 2:[2], 9: [9]},
                                                     printNan=False)

    # MERGING
    df22EP_ACT_PRE_coPRE_complete = merge_coPresence(df22EP_ACT_PRE_coPRE_convert, merge_map={"otherInFAMs": ["TUI_06D", "TUI_06F"], "friends": ["TUI_06H", "TUI_06I"]},
                                                     output_csv_path="/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Outputs/out22EP_ACT_PRE_coPRE.csv",
                                                     rename_map={"PUMFID":"occID", "TUI_01":"occACT", "STARTIME":"start", "ENDTIME":"end", "LOCATION":"occPRE", "INSTANCE": "EPINO",
                                                     "TUI_06A": "Alone", "TUI_06B": "Spouse", "TUI_06C": "Children", "TUI_06G": "otherHHs", "TUI_06J":"others", "TUI_06E":"parents"})
    
    """
    df = pd.read_csv("/Users/orcunkoraliseri/Desktop/Postdoc/2ndJournal/Outputs_GSS/out22EP_ACT_PRE_coPRE.csv")
    print(df.nunique())
    # ----------------
    # EDITING - OTHER COLUMNS






