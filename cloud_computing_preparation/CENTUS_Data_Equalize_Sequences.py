import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# PADDING & TIME RESOLUTION INCREASE ---------------------------
def filter_activities(df):
    # Identify rows where hourStart is different from hourEnd
    df['diff_start_end'] = df['hourStart_Activity'] != df['hourEnd_Activity']

    # Sort by 'diff_start_end' so that these rows come first, then by random order
    df = df.sample(frac=1).sort_values(by=['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity', 'diff_start_end'],
                                       ascending=[True, True, True, False])

    # Drop duplicates keeping the first two for each group
    df = df.drop_duplicates(subset=['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity'], keep='first').drop(
        columns='diff_start_end')

    return df
def increase_time_resolution(df):
    df_half = df.copy()
    df['hourEnd_Activity'] = df['hourStart_Activity'] + 0.5
    df_half['hourStart_Activity'] = df_half['hourStart_Activity'] + 0.5
    return pd.concat([df, df_half])
def process_dataset(input_path, output_path):
    # Read dataset
    df = pd.read_csv(input_path)

    # Filter activities
    df_filtered = filter_activities(df)

    # Increase time resolution
    df_final = increase_time_resolution(df_filtered)

    # Sort and write to CSV
    df_final.sort_values(by=['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity'], inplace=True)
    df_final.to_csv(output_path, index=False)
def process_dataset_hourly(input_path, output_path):
    # Read dataset
    df = pd.read_csv(input_path)

    # Filter activities
    df_filtered = filter_activities(df)

    # Sort and write to CSV
    df_filtered.sort_values(by=['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity'], inplace=True)
    df_filtered.to_csv(output_path, index=False)
def equalize_sequences(input_path, output_path): # 30-minute intervals
    # Read the dataset from the given path
    df = pd.read_csv(input_path)

    # Adjust the start and end times to span the full day
    df.loc[df.groupby(['Household_ID', 'Occupant_ID_in_HH']).head(1).index, 'hourStart_Activity'] = 0
    df.loc[df.groupby(['Household_ID', 'Occupant_ID_in_HH']).tail(1).index, 'hourEnd_Activity'] = 24

    # Calculate the number of 30-minute intervals each activity spans
    df['intervals'] = ((df['hourEnd_Activity'] - df['hourStart_Activity']) * 2).astype(int)

    # Use .repeat() to expand rows by their intervals
    df_repeated = df.loc[df.index.repeat(df['intervals'])].copy()

    # Adjust the start and end times for each interval
    df_repeated['hourStart_Activity'] = df_repeated.groupby(
        ['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity']).cumcount() * 0.5 + df_repeated[
                                            'hourStart_Activity']
    df_repeated['hourEnd_Activity'] = df_repeated['hourStart_Activity'] + 0.5

    # Drop the intervals column and reset index
    df_repeated = df_repeated.drop('intervals', axis=1).reset_index(drop=True)

    # Save the processed dataset to the given output path
    df_repeated.to_csv(output_path, index=False)
    print('sequence equalizing is done')

    return df_repeated
def equalize_sequences_hourly(input_path, output_path):
    # Read the dataset from the given path
    df = pd.read_csv(input_path)

    # Adjust the start and end times to span the full day
    df.loc[df.groupby(['Household_ID', 'Occupant_ID_in_HH']).head(1).index, 'hourStart_Activity'] = 0
    df.loc[df.groupby(['Household_ID', 'Occupant_ID_in_HH']).tail(1).index, 'hourEnd_Activity'] = 24

    # Calculate the number of 1-hour intervals each activity spans
    df['intervals'] = (df['hourEnd_Activity'] - df['hourStart_Activity']).astype(int)

    # Use .repeat() to expand rows by their intervals
    df_repeated = df.loc[df.index.repeat(df['intervals'])].copy()

    # Adjust the start and end times for each interval to reflect hourly sequences
    df_repeated['hourStart_Activity'] = df_repeated.groupby(
        ['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity']
    ).cumcount() * 1 + df_repeated['hourStart_Activity']
    df_repeated['hourEnd_Activity'] = df_repeated['hourStart_Activity'] + 1

    # Drop the intervals column and reset index
    df_repeated = df_repeated.drop('intervals', axis=1).reset_index(drop=True)

    # Save the processed dataset to the given output path
    df_repeated.to_csv(output_path, index=False)
    print('Hourly sequence equalizing is done')

    return df_repeated
#VISUALIZATION: ANALYSIS________________________________________________________________________________________________
import matplotlib.pyplot as plt
def visualize_centus_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Group the data by 'months_season' and 'week_or_weekend'
    grouped_data = df.groupby(['months_season', 'week_or_weekend']).size().reset_index(name='count')

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Labels for weekday categories
    weekday_labels = {
        1: 'Weekday',
        2: 'Saturday',
        3: 'Sunday'
    }

    # Create a bar plot for the grouped data
    for week_category in grouped_data['week_or_weekend'].unique():
        subset = grouped_data[grouped_data['week_or_weekend'] == week_category]
        plt.bar(subset['months_season'] + 0.2 * (week_category - 2), subset['count'], width=0.2,
                label=weekday_labels[week_category])

    # Labeling and visual adjustments
    plt.xlabel('Season')
    plt.ylabel('Number of Entries')
    plt.title('Data Distribution Over Different Seasons')
    plt.xticks([1, 2, 3, 4], ['Winter', 'Spring', 'Summer', 'Autumn'])
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
#EXTRA  -------------------------------------------------
def select_and_save_rows(input_path, output_path):
    import pandas as pd
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_path)

    # Select the first 100 rows
    selected_data = data.head(100)

    # Save the selected data to another CSV file
    selected_data.to_csv(output_path, index=False)
def select_samples_householdID(tus_daily_augmented_path, tus_daily_augmented_sample_path, size=1, specific_household_id=None):
    import pandas as pd
    import numpy as np

    # Read the datasets
    df = pd.read_csv(tus_daily_augmented_path)

    if specific_household_id:
        # Ensure the specific Household ID exists in the data
        if specific_household_id in df['Household_ID'].unique():
            # Filter the DataFrame to only rows with the specific 'Household_ID'
            sample_df = df[df['Household_ID'] == specific_household_id]
            print(f"Sample dataset for specific Household_ID {specific_household_id} saved to {tus_daily_augmented_sample_path}")
        else:
            print(f"Household_ID {specific_household_id} not found in the dataset.")
            return
    else:
        # Select x random 'Household_ID'
        random_ids = np.random.choice(df['Household_ID'].unique(), size=size, replace=False)

        # Filter the DataFrame to only rows with the selected 'Household_ID'
        sample_df = df[df['Household_ID'].isin(random_ids)]

        print(f"Sample dataset for {size} random Household_IDs saved to {tus_daily_augmented_sample_path}")

    # Save the sample_df to a new CSV
    sample_df.to_csv(tus_daily_augmented_sample_path, index=False)
def select_samples_householdID_v3(csv_path, output_csv_path, initial_household_count, column_to_cover):
    """
    Improved function to ensure all unique activities are covered by selectively adding households.

    Parameters:
    csv_path (str): Path to the CSV file to read.
    output_csv_path (str): Path to save the sample dataset CSV.
    initial_household_count (int): Initial number of households to be selected.
    column_to_cover (str): Column name whose unique values need to be covered.

    Returns:
    str: Message indicating the number of households selected and path to saved CSV.
    """

    # Read the dataset from CSV
    df = pd.read_csv(csv_path)

    # Select initial set of random 'Household_ID'
    total_households = df['Household_ID'].unique()
    selected_households = np.random.choice(total_households, size=initial_household_count, replace=False)

    # Create a DataFrame with the selected households
    sample_df = df[df['Household_ID'].isin(selected_households)]

    # Initialize variables for tracking missing activities
    total_unique_activities = set(df[column_to_cover].unique())
    unique_activities_in_sample = set(sample_df[column_to_cover].unique())
    print("unique_activities_in_sample_before:", len(unique_activities_in_sample))
    print("total_unique_activities:", len(total_unique_activities))

    # Iteratively add households until all unique activities are covered
    while unique_activities_in_sample != total_unique_activities:
        missing_activities = total_unique_activities - unique_activities_in_sample
        for activity in missing_activities:
            # Find households with the missing activity
            households_with_activity = df[df[column_to_cover] == activity]['Household_ID'].unique()
            additional_household = np.setdiff1d(households_with_activity, selected_households)
            if additional_household.size:
                # Add the first household that includes the missing activity
                selected_households = np.append(selected_households, additional_household[0])
                sample_df = df[df['Household_ID'].isin(selected_households)]
                unique_activities_in_sample = set(sample_df[column_to_cover].unique())

    print("unique_activities_in_sample_after:", len(unique_activities_in_sample))
    print(sorted(unique_activities_in_sample))

    print(f"Total number of households in the dataset: {len(sample_df['Household_ID'].unique())}")
    # Save the sample_df to a new CSV
    sample_df.to_csv(output_csv_path, index=False)

    return f"Sample dataset for {len(selected_households)} households covering all activities saved to {output_csv_path}"
def select_samples_householdID_v4(csv_path, output_csv_path, initial_household_count, columns_to_cover):
    """
    Enhanced function to ensure all unique values across multiple columns are covered
    by selectively adding households.

    Parameters:
    csv_path (str): Path to the CSV file to read.
    output_csv_path (str): Path to save the sample dataset CSV.
    initial_household_count (int): Initial number of households to be selected.
    columns_to_cover (list): List of column names whose unique values need to be covered.

    Returns:
    str: Message indicating the number of households selected and path to saved CSV.
    """

    import pandas as pd
    import numpy as np

    # Read the dataset from CSV
    df = pd.read_csv(csv_path)

    # Select initial set of random 'Household_ID'
    total_households = df['Household_ID'].unique()
    selected_households = np.random.choice(total_households, size=initial_household_count, replace=False)

    # Create a DataFrame with the selected households
    sample_df = df[df['Household_ID'].isin(selected_households)]

    # Initialize variables for tracking missing values across all columns
    total_unique_values = {col: set(df[col].unique()) for col in columns_to_cover}
    unique_values_in_sample = {col: set(sample_df[col].unique()) for col in columns_to_cover}

    # Iteratively add households until all unique values in all columns are covered
    while any(unique_values_in_sample[col] != total_unique_values[col] for col in columns_to_cover):
        for col in columns_to_cover:
            missing_values = total_unique_values[col] - unique_values_in_sample[col]
            for value in missing_values:
                # Find households with the missing value in the current column
                households_with_value = df[df[col] == value]['Household_ID'].unique()
                additional_household = np.setdiff1d(households_with_value, selected_households)
                if additional_household.size:
                    # Add the first household that includes the missing value
                    selected_households = np.append(selected_households, additional_household[0])
                    sample_df = df[df['Household_ID'].isin(selected_households)]
                    unique_values_in_sample[col] = set(sample_df[col].unique())

    print(f"Total number of households in the dataset: {len(sample_df['Household_ID'].unique())}")

    # Save the sample_df to a new CSV
    sample_df.to_csv(output_csv_path, index=False)

    return f"Sample dataset for {len(selected_households)} households covering all unique values in {columns_to_cover} saved to {output_csv_path}"
def filter_occupants_by_row_count(df):
    """
    Filters the dataframe to retain only those occupants having rows greater than or equal to 48.

    Args:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: Filtered dataframe.
    """
    # Group by 'Household_ID' and 'Occupant_ID_in_HH' and filter groups with size < 48
    valid_occupants = df.groupby(['Household_ID', 'Occupant_ID_in_HH']).filter(lambda x: len(x) >= 48)
    return valid_occupants
def compute_unique_household_count(csv_path):
    """
    Compute the unique number of households needed to cover all activities in the given column.

    Parameters:
    csv_path (str): Path to the CSV file to read.
    initial_household_count (int): Initial number of households to be selected.
    column_to_cover (str): Column name whose unique values need to be covered.

    Returns:
    int: Number of unique households selected to cover all activities.
    """

    # Read the dataset from CSV
    df = pd.read_csv(csv_path)

    # Select initial set of random 'Household_ID'
    total_households = df['Household_ID'].unique()

    unique_household_count = len(total_households)
    print(f"Total number of unique households needed: {unique_household_count}")

    return unique_household_count
#___ADDING withNOBODY___________________________________________________________________________________________________
def add_copresence_csv(input_csv, output_csv):
    """
    Reads the CSV at `input_csv`, generates a 'Copresence' column, fills initial and trailing empty rows
    per occupant based on 'Marital Status', then forward‐fills any remaining NaNs within each occupant group,
    and overwrites the same CSV with the updated DataFrame.

    Steps for 'Copresence':
      1. Initialize Copresence = NaN.
      2. If withNOBODY == 1 → Copresence = 0.
      3. Else if any of [withMOTHER, withFATHER, withSPOUSE, withCHILD,
                        withBROTHER, withOTHERFAMILYMEMBER,
                        withOTHERPERSON, witness] > 0 → Copresence = 1.
      4. INITIAL FILL per occupant group (['Household_ID', 'Occupant_ID_in_HH']):
         • If there is at least one non‐NaN in Copresence:
             – Fill all rows before the first non‐NaN with (1 if Marital Status == 2 else 0).
         • If there are no non‐NaNs initially:
             – Fill the entire group with (1 if Marital Status == 2 else 0).
      5. TRAILING FILL per occupant group:
         • For groups with at least one non‐NaN, fill all rows after the last non‐NaN with (1 if Marital Status == 2 else 0).
         • Groups with no non‐NaNs were already fully filled in step 4.
      6. FORWARD‐FILL any remaining NaNs within each occupant group so that sequences like
         (1,1,1,NaN,NaN,NaN,0,0…) become (1,1,1,1,1,1,0,0…).
    """
    df = pd.read_csv(input_csv)

    other_cols = ["withMOTHER", "withFATHER", "withSPOUSE","withCHILD", "withBROTHER",
                  "withOTHERFAMILYMEMBER","withOTHERPERSON", "witness"]

    # Step 1 & 2 & 3: Initialize and apply basic rules
    df["withNOBODY"] = np.nan
    df.loc[df["withALONE"] == 1, "withNOBODY"] = 0
    mask_other = (df[other_cols] > 0).any(axis=1)
    df.loc[mask_other, "withNOBODY"] = 1

    grouping_cols = ["Household_ID", "Occupant_ID_in_HH"]
    # Step 4: INITIAL FILL per occupant group
    for _, group_indices in df.groupby(grouping_cols).groups.items():
        group_slice = df.loc[group_indices]
        ms = group_slice["Marital Status"].iloc[0]
        fill_value = 1 if ms == 2 else 0

        non_null_mask = group_slice["withNOBODY"].notna()
        if non_null_mask.any():
            first_valid_idx = group_slice[non_null_mask].index[0]
            idx_list = list(group_slice.index)
            first_pos = idx_list.index(first_valid_idx)
            to_fill_initial = idx_list[:first_pos]
        else:
            to_fill_initial = list(group_slice.index)

        df.loc[to_fill_initial, "withNOBODY"] = fill_value

    # Step 5: TRAILING FILL per occupant group
    for _, group_indices in df.groupby(grouping_cols).groups.items():
        group_slice = df.loc[group_indices]
        ms = group_slice["Marital Status"].iloc[0]
        fill_value = 1 if ms == 2 else 0

        non_null_mask = group_slice["withNOBODY"].notna()
        if non_null_mask.any():
            last_valid_idx = group_slice[non_null_mask].index[-1]
            idx_list = list(group_slice.index)
            last_pos = idx_list.index(last_valid_idx)
            to_fill_trailing = idx_list[last_pos + 1:]
            df.loc[to_fill_trailing, "withNOBODY"] = fill_value
        # If group had no non-NaNs, it's already filled fully in step 4

    # Step 6: FORWARD‐FILL any remaining NaNs within each occupant group
    df["withNOBODY"] = df.groupby(grouping_cols)["withNOBODY"].ffill()
    # ----- NEW STEP: Drop helper columns -----
    drop_cols = [
        "withALONE", "withMOTHER", "withFATHER", "withSPOUSE", "withCHILD",
        "withBROTHER", "withOTHERFAMILYMEMBER", "withOTHERPERSON", "witness"
    ]
    df = df.drop(columns=drop_cols)
    # Overwrite original CSV
    df.to_csv(output_csv, index=False)
#---SELECTION-BASIC---------------------------------------------------------------------------------------------------
def select_first_100(input_csv, output_csv):
    """
    Reads `input_csv`, selects the first 100 rows, and writes them to `output_csv`.
    """
    df = pd.read_csv(input_csv, nrows=100)
    df.to_csv(output_csv, index=False)
def select_by_household(input_csv, household_id, output_csv):
    """
    Reads `input_csv`, filters rows where 'Household_ID' equals `household_id`,
    and writes them to `output_csv`.
    """
    df = pd.read_csv(input_csv)
    filtered_df = df[df['Household_ID'] == household_id]
    filtered_df.to_csv(output_csv, index=False)
if __name__ == '__main__':
    #INPUT PATHS
    tus_main = r'dataset_TUS_main/TUS_main.csv'
    tus_main_RAWDATA_31 = r'dataset_TUS_main/tus_main_RAWDATA_31.csv'

    #INPUT PATHS - PROCESSED
    tus_main_processed = r'dataset_TUS_equalized/TUS_main_processed.csv'
    tus_main_RAWDATA_31_processed = r'dataset_TUS_equalized/tus_main_RAWDATA_31_processed.csv'

    #OUTPUT PATHS - EQUALIZED
    tus_mainEqualPadHHID = r'dataset_TUS_equalized/tus_mainEqualPadHHID.csv'
    tus_main_EqPadHHID_RAWDATA_31 = r'dataset_TUS_equalized/tus_main_EqPadHHID_RAWDATA_31.csv'

    # OUTPUT PATHS - TRIAL
    tus_main_EqPad100HHID_RAWDATA_31 = r'dataset_TUS_equalized/tus_main_EqPad100HHID_RAWDATA_31.csv'  # 60 minute intervals
    tus_main_EqPad100HHID_RAWDATA_31_100 = r'dataset_TUS_equalized/tus_main_EqPad100HHID_RAWDATA_31_100.csv'  # 60 minute intervals
    tus_main_EqPad500HHID_RAWDATA_31 = r'dataset_TUS_equalized/tus_main_EqPad500HHID_RAWDATA_31.csv'  # 60 minute intervals


    #___PRE-PROCESSING: 30-minute intervals
    #process_dataset(input_path=tus_main, output_path=tus_main_processed)
    #equalize_sequences(input_path=tus_main_processed, output_path=tus_mainEqualPadHHID)

    #___PRE-PROCESSING: 60-minute intervals
    #process_dataset_hourly(input_path=tus_main_EqPadHHID_RAWDATA_31, output_path=tus_main_RAWDATA_31_processed)
    #equalize_sequences_hourly(input_path=tus_main_RAWDATA_31_processed, output_path=tus_main_EqPadHHID_RAWDATA_22_v2)

    #___ADDING WITHNOBODY_______________________________________________________________________________________________
    #add_copresence_csv(tus_main_EqPadHHID_RAWDATA_31, tus_main_EqPadHHID_RAWDATA_22_v2)
    #select_first_100(tus_main_EqPadHHID_RAWDATA_31, r'dataset_TUS_main/tus_main_EqPadHHID_RAWDATA_31_copresence_sample.csv')
    #select_by_household(tus_main_EqPadHHID_RAWDATA_31, 7, r'dataset_raw/tus_main_EqPadHHID_RAWDATA_31_HHID7.csv')

    #___DATA SELECTION__________________________________________________________________________________________________
    #select_and_save_rows(input_path=tus_indiv_d_path, output_path=tus_indiv_d_firstRows)
    #select_samples_householdID(tus_main, tus_mainEqualPad1HHID, 1) # select number of household
    #select_samples_householdID(tus_main_simplified, tus_main_1HHID, specific_household_id=1) # select number of household
    # this can select all the unique activities in the sample
    #select_samples_householdID_v3(tus_main_EqPadHHID_DATA5,  tus_main_EqPad1000HHID_DATA5, 1000, 'Occupant_Activity')  # this oen covers one column
    select_samples_householdID_v4(tus_main_EqPadHHID_RAWDATA_31,  tus_main_EqPad500HHID_RAWDATA_31, 500, columns_to_cover=['Occupant_Activity', 'Family Typology', "Number Family Members", "Occupant_ID_in_HH"])

    #VISUALIZATION: EXTRA_______________________________________________________________________________________________
    #visualize_centus_data(tus_mainEqualPad)
    #compute_unique_household_count(tus_mainEqualPad)

    #VISUALIZATION: GENERAL_____________________________________________________________________________________________
    from preProcessing_Func import analysis_func as dfppaf
    input_Data = tus_main_EqPad500HHID_RAWDATA_31
    #ID_DROP = ["Household_ID", 'Occupant_ID_in_HH']
    ID_DROP = ["Household_ID"]
    #ID_DROP = ["Household_ID", 'OCC_order_HH']

    binary = True
    dfppaf.analysis(input_path=input_Data, columns=binary)
    #dfppaf.analysis(input_path=input_Data, data_len=binary)
    #dfppaf.analysis(input_path=input_Data, unique=binary, uniqueIDcolstoDrop=ID_DROP)
    dfppaf.analysis(input_path=input_Data, missingness=binary)
    #dfppaf.analysis(input_path=input_Data, headPrint=binary)
    #dfppaf.analysis(input_path=input_Data, describe=binary)
    #dfppaf.analysis(input_path=input_Data, unique_visual_byCols=binary, uniqueIDcolstoDrop=ID_DROP)
    #dfppaf.analysis(input_path=input_Data, count_unique_values=binary, uniqueIDcolstoDrop=ID_DROP)
