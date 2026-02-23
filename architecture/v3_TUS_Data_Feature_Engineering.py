import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#FEATURE ENGINEERING-------------------------------------------------
import pandas as pd
import numpy as np
def compute_activity_durations(df):
    # Activity_Duration for each distinct activity
    activity_duration = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'Occupant_Activity']).size().reset_index(name='Activity_Count')
    activity_duration['Activity_Duration'] = activity_duration['Activity_Count'] * 0.5

    # Integrate into the original dataset
    df = pd.merge(df, activity_duration[['Household_ID', 'Occupant_ID_in_HH', 'Occupant_Activity', 'Activity_Duration']], on=['Household_ID', 'Occupant_ID_in_HH', 'Occupant_Activity'], how='left')

    # Average_Activity_Duration for each occupant
    average_activity_duration = activity_duration.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Activity_Duration'].mean().reset_index()
    average_activity_duration.columns = ['Household_ID', 'Occupant_ID_in_HH', 'Average_Activity_Duration']

    # Integrate into the original dataset
    df = pd.merge(df, average_activity_duration, on=['Household_ID', 'Occupant_ID_in_HH'], how='left')

    return df
def compute_activity_stats(df):
    # --- Total number of distinct activities
    """counts the number of unique activities an occupant has performed, regardless of how many times they switched between them"""
    df['distinct_activities'] = df.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Occupant_Activity'].transform('nunique')

    # --- most_frequent_activity ---
    # Most frequent activity for each occupant
    df['most_frequent_activity'] = df.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Occupant_Activity'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

    # --- least_frequent_activity ---
    # Least frequent activity for each occupant
    def least_frequent_activity(x):
        return x.value_counts().idxmin() if not x.empty else np.nan

    df['least_frequent_activity'] = df.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Occupant_Activity'].transform(least_frequent_activity)

    return df
def compute_all_aggregative_features(df):
    df = compute_activity_durations(df)
    df = compute_activity_stats(df)
    return df
def temporal_patterns(df):
    # Sort the data to ensure it's in order
    df_sorted = df.sort_values(by=['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity'])

    # --- Calculate Max_Consecutive ---
    # Detect changes in the activity to define sequence breaks
    df_sorted['Activity_Change'] = df_sorted.groupby(['Household_ID', 'Occupant_ID_in_HH'])[
                                       'Occupant_Activity'].shift() != df_sorted['Occupant_Activity']

    # Create a group identifier for each sequence of consecutive same activities
    df_sorted['Group_ID'] = df_sorted.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Activity_Change'].cumsum().fillna(
        0)

    # Calculate the length of each sequence
    df_sorted['Max_Consecutive'] = df_sorted.groupby(['Household_ID', 'Occupant_ID_in_HH', 'Group_ID'])[
        'Occupant_Activity'].transform('size')

    # Start and end of day computation
    wake_up_mask = (df_sorted['Occupant_Activity'] != 11) & (df_sorted['Activity_Change'])
    sleep_mask = (df_sorted['Occupant_Activity'] == 11) & (df_sorted['Activity_Change'])

    # --- Calculate'start_of_day' and 'end_of_day' ---
    df_sorted['start_of_day'] = df_sorted.loc[wake_up_mask, 'hourStart_Activity']
    df_sorted['end_of_day'] = df_sorted.loc[sleep_mask, 'hourEnd_Activity']

    df_sorted['start_of_day'] = df_sorted.groupby(['Household_ID', 'Occupant_ID_in_HH'])['start_of_day'].transform(
        'first')
    df_sorted['end_of_day'] = df_sorted.groupby(['Household_ID', 'Occupant_ID_in_HH'])['end_of_day'].transform('last')

    # Drop temporary columns
    df_sorted = df_sorted.drop(columns=['Activity_Change', 'Group_ID'])

    # Merge these new features back to the original dataframe
    df = pd.merge(df, df_sorted[['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity',
                                 'Max_Consecutive', 'start_of_day', 'end_of_day']],
                  on=['Household_ID', 'Occupant_ID_in_HH', 'hourStart_Activity'], how="left")

    return df
def changing_points(df):
    # Changing Points: Activity Change Points
    """counts the number of times an occupant switches from one activity to another.
    """
    # Assuming df is your dataframe

    # Identify points where activity changes
    df['Activity_Changed'] = df.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Occupant_Activity'].diff().ne(0)

    # Sum these change points for each group
    df['activity_changes'] = df.groupby(['Household_ID', 'Occupant_ID_in_HH'])['Activity_Changed'].transform('sum')

    # Drop the temporary 'Activity_Changed' column
    df = df.drop(columns=['Activity_Changed'])

    return df
def transformation(df):

    # --- Activity Entropy ---
    # Transformation Techniques: Activity Entropy
    """Measure the unpredictability in activity patterns to differentiate between random and predictable behaviors.
    1- Grouping and getting value counts in one go.
    2- Calculating entropy directly from these counts.
    3- Merging the result back to the dataframe.
    """
    def calculate_entropy(counts):
        total = counts.sum()
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs))
        return entropy


    # 1. Pre-compute value counts
    value_counts = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'Occupant_Activity']).size().reset_index(
        name='counts')
    # 2. Calculate entropy using these pre-computed value counts
    entropies = value_counts.groupby(['Household_ID', 'Occupant_ID_in_HH']).apply(
        lambda group: calculate_entropy(group['counts'])).reset_index(name='Activity_Entropy')
    # Merge back with the original dataframe
    df = df.merge(entropies, on=['Household_ID', 'Occupant_ID_in_HH'], how='left')

    # --- Decomposition ---
    from statsmodels.tsa.seasonal import seasonal_decompose
    for _, occupant_df in df.groupby(['Household_ID', 'Occupant_ID_in_HH']):
        occupant_df.loc[:, 'Activity_Changed'] = occupant_df['Occupant_Activity'].diff().ne(0)
        changes_by_hour = occupant_df[occupant_df['Activity_Changed']].groupby('hourStart_Activity').size()

        # Reindex to ensure complete time series
        changes_by_hour = changes_by_hour.reindex(np.arange(0, 24, 0.5), fill_value=0)

        # Duplicate the series to simulate two days
        changes_by_hour = pd.concat([changes_by_hour, changes_by_hour])

        decomposition = seasonal_decompose(changes_by_hour, model='additive', period=48)
        seasonal = decomposition.seasonal

        # Take only the first half (i.e., original day's data) for merging back
        seasonal = seasonal.iloc[:48]

        # Merging back with the main dataframe
        # This component represents the recurring patterns in the activity changes.
        df.loc[occupant_df.index, 'seasonal'] = seasonal.values

    # Cleaning up
    if 'Activity_Changed' in df.columns:
        df.drop(columns=['Activity_Changed'], inplace=True)

    return df

#FEATURE ENGINEERING - ENCODING
def drop_id_columns(df):
    return df.drop(columns=['Household_ID', 'Occupant_ID_in_HH'])
def binary_encode_column(series):
    # Convert categories to integers
    cat_to_int = {cat: i for i, cat in enumerate(series.unique())}
    int_encoded = series.map(cat_to_int)

    # Find the number of binary columns needed
    max_value = int(int_encoded.max())
    max_bin_length = max_value.bit_length()

    # Convert integers to binary format and split into separate columns
    bin_encoded = int_encoded.apply(lambda x: pd.Series(list(f"{x:0{max_bin_length}b}")).astype(int))
    bin_encoded.columns = [f"{series.name}_bit{col}" for col in bin_encoded.columns]

    return bin_encoded

def custom_binary_encode(data, columns_to_encode):
    encoded_data = data.copy()
    for col in columns_to_encode:
        bin_encoded = binary_encode_column(data[col])
        encoded_data = encoded_data.drop(col, axis=1)
        encoded_data = pd.concat([encoded_data, bin_encoded], axis=1)
    return encoded_data

from sklearn.preprocessing import StandardScaler
def standardize_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)

#FEATURE ENGINEERING - MAIN
def feature_engineering(input_path, output_path, encoding=False, convertNontemporal=False):
    # Read the dataset from the given path
    df = pd.read_csv(input_path)

    # Filter occupants with less than 48 rows
    df = filter_occupants_by_row_count(df)

    df = compute_all_aggregative_features(df)
    df = temporal_patterns(df)
    df = changing_points(df)
    df = transformation(df)

    data = df[[
               'Household_ID', 'Occupant_ID_in_HH',
                'week_or_weekend',
               'most_frequent_activity', 'least_frequent_activity','distinct_activities',
               'Max_Consecutive','start_of_day', 'end_of_day',
               'activity_changes',
               'Activity_Entropy','seasonal',
               ]]

    # prepare the data
    # Dropping duplicates based on 'Household_ID' and 'Occupant_ID_in_HH' to ensure unique entries for each occupant
    if convertNontemporal == True:
        data = data.drop_duplicates(subset=['Household_ID', 'Occupant_ID_in_HH'])

    if encoding == True:
        # Assuming df is your original dataset
        data = custom_binary_encode(data, ['most_frequent_activity', 'least_frequent_activity'])
        data = standardize_data(data)
    else:
        pass

    # Dropping the columns
    #unique_data = unique_data.drop(columns=['Household_ID', 'Occupant_ID_in_HH'])

    # Save the processed dataset to the given output path
    data.to_csv(output_path, index=False)
    return df

#VISUALIZATION: ANALYSIS___________________________________________________________________________________________________________________________
import pandas as pd
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

def select_samples_householdID(tus_daily_augmented_path, tus_daily_augmented_sample_path, x):
    import pandas as pd
    import numpy as np

    # Read the datasets
    df = pd.read_csv(tus_daily_augmented_path)

    # Select x random 'Household_ID'
    random_ids = np.random.choice(df['Household_ID'].unique(), size=x, replace=False)

    # Filter the DataFrame to only rows with the selected 'Household_ID'
    sample_df = df[df['Household_ID'].isin(random_ids)]

    # Save the sample_df to a new CSV
    sample_df.to_csv(tus_daily_augmented_sample_path, index=False)

    print(f"Sample dataset for {x} random Household_IDs saved to {tus_daily_augmented_sample_path}")

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

    #print("unique_activities_in_sample_after:", len(unique_activities_in_sample))
    #print(sorted(unique_activities_in_sample))

    print(f"Total number of households in the dataset: {len(sample_df['Household_ID'].unique())}")
    # Save the sample_df to a new CSV
    sample_df.to_csv(output_csv_path, index=False)

    return f"Sample dataset for {len(selected_households)} households covering all activities saved to {output_csv_path}"

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

if __name__ == '__main__':
    '''
    K-shape clustering of TUS_indiv_d for pattern analysis
        - Filter Activities in Same Hour: The goal is to look for rows that have activities within the same starting hour and limit to two activities. 
            Priority is given to activities with different start and end hours, while the others are selected randomly.
        - Increase Time Resolution: Once the data is filtered, the next goal is to convert the hourly activity data to 30-minute intervals 
            by duplicating rows and adjusting time columns accordingly.
        - Data pre-processing: Cyclical encoding, normalization/standardization 
        - Dimension reduction: 
            - PCA: Retain components explaining, e.g., 95% variance.
            - t-SNE (Optional): Visualize using PCA-reduced data.
            - Distance Metric: Use Gower's distance for mixed datasets.
    Change Point Detection of TUS_indiv_d for shift analysis
    '''

    #INPUT PATHS
    tus_mainEqualPad = r'dataset_TUS_equalized/tus_mainEqualPad.csv'

    # OUTPUT PATHS - FEATURE ENGINEERING
    tus_indiv_d_featENG = r"dataset_step4/TUS_indiv_d_featENG.csv" #non-temporal features by feature engineering

    #___DATA SELECTION___________________________________________________________________________________________________________________________
    #select_and_save_rows(input_path=tus_indiv_d_path, output_path=tus_indiv_d_firstRows)
    #select_samples_householdID(tus_indiv_d_equal_path, CENTUS_sample_100random, 100) # select number of household
    #select_samples_householdID_v3(tus_indiv_d_equal_path,  TUS_sampleAllAct1000HHID, 1000, 'Occupant_Activity') # this can select all the unique activities in the sample

    #___FEATURE ENGINEERING___________________________________________________________________________________________________________________________
    #feature_engineering(input_path=tus_indiv_d_sample, output_path=tus_indiv_d_featENG, encoding=False, convertNontemporal=True) #test with sample data
    #feature_engineering(input_path=tus_indiv_d_sample, output_path=tus_indiv_d_featENG, encoding=False, convertNontemporal=False) #test with sample data
    #feature_engineering(input_path=tus_indiv_d_sample, output_path=tus_indiv_d_featENG, encoding=False) #with original data

    #VISUALIZATION: EXTRA___________________________________________________________________________________________________________________________
    #visualize_centus_data(tus_indiv_d_equal_path)

    #VISUALIZATION: GENERAL___________________________________________________________________________________________________________________________
    from preProcessing_Func import analysis_func as dfppaf
    input_Data= tus_mainEqualPad
    ID_DROP = ["Household_ID", 'Occupant_ID_in_HH']
    #ID_DROP = ["Household_ID", 'OCC_order_HH']

    binary=True
    dfppaf.analysis(input_path=input_Data, columns=binary)
    dfppaf.analysis(input_path=input_Data, data_len=binary)
    dfppaf.analysis(input_path=input_Data, unique=binary, uniqueIDcolstoDrop=ID_DROP)
    #dfppaf.analysis(input_path=input_Data, headPrint=binary)
    #dfppaf.analysis(input_path=input_Data, describe=binary)
    #dfppaf.analysis(input_path=input_Data, unique_visual_byCols=binary, uniqueIDcolstoDrop=ID_DROP)