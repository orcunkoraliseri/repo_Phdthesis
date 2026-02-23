
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

#FEATURE ENGINEERING-------------------------------------------------
import pandas as pd
import numpy as np

#BASED ON OCCUPANT_ACTIVITY COLUMN
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
        return round(entropy,2)

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
        df.loc[occupant_df.index, 'seasonal'] = round(seasonal,2).values

    # Cleaning up
    if 'Activity_Changed' in df.columns:
        df.drop(columns=['Activity_Changed'], inplace=True)

    return df
def calculate_unique_activity_counts(df):
    # Create masks for inside and outside locations
    inside_mask = df['location'] == 1
    outside_mask = df['location'] == 0

    # Group by Household_ID and Occupant_ID_in_HH and aggregate unique activities
    unique_activities = df.groupby(['Household_ID', 'Occupant_ID_in_HH']).agg({
        'Occupant_Activity': [lambda x: x[inside_mask].nunique(),
                              lambda x: x[outside_mask].nunique()]
    }).reset_index()

    # Flatten the MultiIndex and rename columns appropriately
    unique_activities.columns = ['Household_ID', 'Occupant_ID_in_HH', 'Unique_Activities_Inside',
                                 'Unique_Activities_Outside']

    # Merge the unique activity counts back into the original dataframe
    df = df.merge(unique_activities, on=['Household_ID', 'Occupant_ID_in_HH'], how='left')

    return df

#BASED ON LOCATION COLUMN
def calculate_inside_outside_hours(df):
    # Calculate duration of each row
    df['Duration'] = df['hourEnd_Activity'] - df['hourStart_Activity']

    # Group by Household_ID and Occupant_ID_in_HH and calculate total hours inside and outside
    inside_outside = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'location'])['Duration'].sum().unstack(
        fill_value=0)

    # Rename the columns for clarity
    inside_outside.columns = ['Total_Hours_Outside', 'Total_Hours_Inside']

    # Reset index to flatten the DataFrame
    inside_outside = inside_outside.reset_index()

    # Merge the calculated hours back into the original dataframe
    df = df.merge(inside_outside, on=['Household_ID', 'Occupant_ID_in_HH'], how='left')

    # Drop the temporary 'Duration' column as it is no longer needed
    df = df.drop(columns=['Duration'])

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

    # FEATURE ENGINEERING:Extract new features
    df = compute_all_aggregative_features(df)
    df = temporal_patterns(df)
    df = changing_points(df)
    df = transformation(df)
    df = calculate_inside_outside_hours(df)
    df = calculate_unique_activity_counts(df)
    data = df
    #print(data.columns)
    """
    data = df[[
               'Household_ID', 'Occupant_ID_in_HH',
               'months_season', 'week_or_weekend',
               'Region', 'Number Family Members', 'Family Typology', 'Age Classes', 'Employment status', 'Gender', 'Education Degree',
                'Gender',
                'Home Type',
               #'most_frequent_activity', 'least_frequent_activity',
               #'distinct_activities',
               #'Max_Consecutive',
               # 'Unique_Activities_Inside',
               # 'Unique_Activities_Outside',
               #'start_of_day', 'end_of_day',
               #'activity_changes',
               #'Activity_Entropy','seasonal',
               #'Total_Hours_Outside', 'Total_Hours_Inside'
               ]]
    """

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

#ANALYSIS - CORRELATION
import pandas as pd
def corr_analysis(input_path):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Read the dataset from the given path
    df = pd.read_csv(input_path)

    # Calculate the correlation matrix
    correlation_matrix = df[['Region', 'Number Family Members', 'Family Typology', 'Age Classes', 'Employment status', 'Gender', 'Education Degree',
                             'most_frequent_activity', 'least_frequent_activity', 'Max_Consecutive', 'Unique_Activities_Inside', 'Unique_Activities_Outside',
                             'start_of_day', 'end_of_day', 'activity_changes', 'Activity_Entropy', 'seasonal', 'Total_Hours_Outside', 'Total_Hours_Inside']].corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Return the correlation matrix
    return correlation_matrix
def corr_analysis_with_composite(input_path):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from prince import MCA  # Make sure to install prince with pip
    # Read the dataset
    df = pd.read_csv(input_path)
    demographic_features = ['Region', 'Number Family Members', 'Family Typology', 'Age Classes', 'Employment status', 'Gender', 'Education Degree']
    df = compositeScore(df)

    # Compute the correlation of the composite score with demographic variables
    correlation_matrix = df[demographic_features + ['Composite_Score']].corr(method='spearman')

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation with Composite Score')
    plt.show()

    # Return the correlation matrix
    return correlation_matrix
def compositeScore(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from prince import MCA  # Make sure to install prince with pip
    # Read the dataset
    # Define continuous, categorical, and demographic features
    continuous_features = ['Activity_Entropy', 'seasonal']
    categorical_features = ['most_frequent_activity', 'least_frequent_activity', 'Max_Consecutive',
                            'Unique_Activities_Inside', 'Unique_Activities_Outside',
                            #'start_of_day', 'end_of_day',
                            'activity_changes', 'Total_Hours_Outside', 'Total_Hours_Inside']
    # Standardize continuous features
    scaler = StandardScaler()
    continuous_scaled = scaler.fit_transform(df[continuous_features])

    # Apply PCA to continuous features
    pca = PCA(n_components=1)
    pca_score = pca.fit_transform(continuous_scaled)

    # Apply MCA to categorical features
    mca = MCA(n_components=1)
    mca_score = mca.fit_transform(df[categorical_features])

    # Combine PCA and MCA scores to form a composite score
    composite_score = pca_score.ravel() \
                      + mca_score.iloc[:, 0].values

    # Add the composite score to the dataframe
    df['Composite_Score'] = composite_score

    df.drop(columns=['most_frequent_activity', 'least_frequent_activity', 'Max_Consecutive',
                     'Unique_Activities_Inside', 'Unique_Activities_Outside', 'start_of_day',
                     'end_of_day', 'activity_changes', 'Total_Hours_Outside', 'Total_Hours_Inside',
                     'Activity_Entropy', 'seasonal'], inplace=True)

    # Return the correlation matrix
    return df

#ANALYSIS - FEATURE IMPORTANCE__________________________________________________________________________________________
def feature_importance(path, target="Cluster_Label", figsize=(14, 10), label_rotation=0):
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    """
    Calculates and plots the feature importances of a fitted Random Forest.

    Parameters:
    - path: str, the file path to the CSV file containing the data.
    - figsize: tuple, the size of the figure (width, height in inches).
    - label_rotation: int or float, the rotation angle of the y-axis labels.

    Returns:
    - A DataFrame containing the feature importances.
    """

    # Read the data
    data = pd.read_csv(path)

    #data = compositeScore(data)
    print(data.columns)

    data.drop(columns=['Household_ID', 'Occupant_ID_in_HH'], inplace=True)

    X = data.drop(target, axis=1)  # Features
    X = X[['Region', 'Number Family Members', 'Family Typology', 'Age Classes', 'Employment status', 'Gender', 'Education Degree', 'Home Type', 'months_season', 'week_or_weekend',]]
    y = data[target]  # Target variable

    clf = RandomForestRegressor()
    clf.fit(X, y)

    feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index=X.columns,
                                       columns=['Importance']).sort_values('Importance', ascending=False)

    # Plotting
    plt.figure(figsize=figsize)
    plt.title('Feature Importances')

    # Barh creates a horizontal bar plot
    plt.barh(np.arange(len(feature_importances)), feature_importances['Importance'], align='center')

    plt.yticks(np.arange(len(feature_importances)), feature_importances.index, rotation=label_rotation)
    plt.xlabel('Relative Importance')

    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

    return feature_importances

#SHAP________________________________________________________________________________________________
def calculate_shap_values(path, target="Composite_Score"):
    import shap
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt

    # Read the data
    data = pd.read_csv(path)

    # Assuming you've already created your composite score
    data = compositeScore(data)
    data.drop(columns=['Household_ID', 'Occupant_ID_in_HH'], inplace=True)

    # Prepare the data
    X = data.drop(target, axis=1)  # Features
    y = data[target]  # Target variable

    # Example usage
    model = RandomForestRegressor()  # or any other model
    # Fit the model
    model.fit(X, y)

    # Calculate SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Plot the SHAP values with specified figure size
    plt.figure(figsize=(24, 12))
    shap.summary_plot(shap_values, X, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('shap_plot.png', bbox_inches='tight')  # Saves the plot as a PNG file without the text.
    return shap_values

#permutation_feature_importance_________________________________________________________________________________________
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def permutation_feature_importance(path, target='Composite_Score'):
    # Read the dataset
    df = pd.read_csv(path)

    # Assuming you've already created your composite score
    df = compositeScore(df)
    df.drop(columns=['Household_ID', 'Occupant_ID_in_HH'], inplace=True)

    # Split the data into features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit a model (Random Forest in this case)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Perform permutation feature importance
    results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Organize the results
    feature_importances = pd.Series(results.importances_mean, index=X.columns)

    # Plotting
    feature_importances.sort_values().plot(kind='barh', figsize=(12, 8))
    plt.title('Permutation Feature Importance')
    plt.xlabel('Mean Decrease in Performance')
    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

    return feature_importances

if __name__ == '__main__':
    """
    Aim:To analyze the impact of the occupant demographcis for the activity and presence schedule columns 
    - feature engineering: extract new features from activity and presence schedule columns
    - combine features: define combined metric for all the target features 
    - implement sensitivity analysis: check if the global sensitivty analysis is possible
    """

    #INPUT PATHS
    #tus_indiv_d_equal_path = r"dataset_impactAnalysis/tus_indiv_d_euqal.csv"
    #tus_indiv_d_equal_path_100_sample = r"dataset_impactAnalysis/tus_indiv_d_equal_100sample.csv"

    #OUTPUT PATHS - PRE-PROCESSING
    tus_mainEqualPad = r'dataset_TUS_equalized/tus_mainEqualPad.csv'
    tus_mainEqualPad100HHID = r'dataset_TUS_equalized/tus_mainEqualPad100HHID.csv'

    # OUTPUT PATHS - FEATURE ENGINEERING
    tusFE = r"dataset_impactAnalysis/tusFE.csv" #non-temporal features by feature engineering

    #FeatureEngineering_________________________________________________________________________________________________
    feature_engineering(input_path=tus_mainEqualPad100HHID, output_path=tusFE, encoding=False, convertNontemporal=True) #test with sample data

    #CorrelationAnalysis________________________________________________________________________________________________
    #corr_analysis(input_path=tusFE)
    #corr_analysis_with_composite(input_path=tusFE)

    #FE________________________________________________________________________________________________
    feature_importance(path=tusFE, target="Composite_Score")

    #SHAP________________________________________________________________________________________________
    """SHAP (SHapley Additive exPlanations): SHAP values break down a prediction to show the impact of each feature. 
    It's based on game theory and provides a robust way to interpret complex models."""
    #shap_values = calculate_shap_values(path=tusFE, target="Composite_Score")

    # permutation_feature_importance_________________________________________________________________________________________
    #feature_importances = permutation_feature_importance(path=tusFE)

    #VISUALIZATION: GENERAL_____________________________________________________________________________________________
    from preProcessing_Func import analysis_func as dfppaf
    input_Data= tusFE
    ID_DROP = ["Household_ID", 'Occupant_ID_in_HH']
    #ID_DROP = ["Household_ID", 'OCC_order_HH']

    binary=True
    dfppaf.analysis(input_path=input_Data, columns=binary)
    dfppaf.analysis(input_path=input_Data, data_len=binary)
    #dfppaf.analysis(input_path=input_Data, unique=binary, uniqueIDcolstoDrop=ID_DROP)
    #dfppaf.analysis(input_path=input_Data, headPrint=binary)
    #dfppaf.analysis(input_path=input_Data, describe=binary)
    #dfppaf.analysis(input_path=input_Data, unique_visual_byCols=binary, uniqueIDcolstoDrop=ID_DROP)



