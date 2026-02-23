import pandas as pd
#PREPROCESS-------------------------------------------------------------------------------------------------------------
def stratified_split(df, test_size, random_state=42):
    from sklearn.model_selection import StratifiedShuffleSplit
    # Create a 'strata' column that combines 'months_season' and 'week_or_weekend'
    #df['strata'] = df['months_season'].astype(str) + "_" + df['week_or_weekend'].astype(str)
    df_copy = df.copy()
    df_copy['strata'] = df_copy['months_season'].astype(str) + "_" + df_copy['week_or_weekend'].astype(str)

    # Initialize the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in sss.split(df_copy, df_copy['strata']):
        train_data_ids = df_copy.iloc[train_index][['Household_ID', 'Occupant_ID_in_HH']].drop_duplicates()
        test_data_ids = df_copy.iloc[test_index][['Household_ID', 'Occupant_ID_in_HH']].drop_duplicates()

    # Drop the 'strata' column as it's no longer needed
    df_copy.drop('strata', axis=1, inplace=True)

    return train_data_ids, test_data_ids
def data_preprocess(df):
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import LabelEncoder

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 1. Initial Inspection
    grouped = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend'])
    to_remove = grouped.filter(lambda x: len(x) != 48).index
    df.drop(to_remove, inplace=True)

    # Integer encoding for the rest of the categorical columns
    # to keep the preprocessing consistent and simple, binary column of "gender" inserted in integer encoding
    one_hot_cols = ['Education Degree',
                    'Employment status',
                    'Family Typology',
                    'Number Family Members',
                    'Gender',
                    'Occupant_ID_in_HH',
                    'months_season',
                    'week_or_weekend',]

    for col in one_hot_cols:
        df[col] = df[col].astype('category').cat.codes


    # Desired column order based on impact analysis
    impactAnalysis_order = ['Household_ID', 'Education Degree','Employment status','Gender', 'Family Typology','Number Family Members',
                            'Occupant_ID_in_HH',
                            'months_season', 'week_or_weekend', 'hourStart_Activity', 'hourEnd_Activity', 'Occupant_Activity', "location", "withNOBODY",]
    df = df[impactAnalysis_order]

    # Suppress the SettingWithCopyWarning globally
    pd.options.mode.chained_assignment = None  # default='warn'

    # 3. Temporal Encoding
    time_cols = ['hourStart_Activity', 'hourEnd_Activity']
    df[[f'sin_{col}' for col in time_cols]] = np.sin(2 * np.pi * df[time_cols] / 24.0)
    df[[f'cos_{col}' for col in time_cols]] = np.cos(2 * np.pi * df[time_cols] / 24.0)
    # Optionally, you can reset the option back to the default warning state
    pd.options.mode.chained_assignment = 'warn'
    df = df.copy()
    df.drop(columns=time_cols, inplace=True)

    # 4. Normalization (except for the columns that will be used in embedding layers)
    scaler = RobustScaler() # if the data has outliers or is not normally distributed
    to_scale = df.columns.difference(one_hot_cols + ['Occupant_Activity', "location", "withNOBODY", "Gender"])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.3)
    valid_data_ids, test_data_ids = stratified_split(df.loc[df.index.isin(temp_data_ids.index)], test_size=0.5)

    train_data = df.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    valid_data = df.merge(valid_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    test_data = df.merge(test_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])

    # Drop 'Household_ID' column post splitting
    for dataset in [train_data, valid_data, test_data]:
        dataset.drop(columns=['Household_ID'], inplace=True)

    # TESTING DISTRIBUTIONS --------------------------------------------------------------------------------------------
    # Before splitting
    #activity_distribution_before, location_distribution_before, withnobody_distribution_before  = before_splitting(df)
    # After splitting
    #after_splitting(train_data, valid_data, test_data, activity_distribution_before, location_distribution_before, withnobody_distribution_before)

    #ENCODING TARGET VARIABLE - LABEL ENCODING -------------------------------------------------------------------------
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    #unique_activities = df['Occupant_Activity'].unique().ravel()
    unique_activities = df['Occupant_Activity'].unique().reshape(-1, 1)
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_activities)

    # Convert the 'Occupant_Activity' of train, valid and test to one-hot encoded
    y_activity_train = label_encoder.transform(train_data[['Occupant_Activity']].values.ravel())
    y_activity_valid = label_encoder.transform(valid_data[['Occupant_Activity']].values.ravel())
    y_activity_test = label_encoder.transform(test_data[['Occupant_Activity']].values.ravel())

    # Use the 'location' column directly as a binary target
    y_location_train = train_data[['location']].values.ravel()
    y_location_valid = valid_data[['location']].values.ravel()
    y_location_test = test_data[['location']].values.ravel()

    # Use the 'withNOBODY' column directly as a binary target
    y_withNOB_train = train_data[['withNOBODY']].values.ravel()
    y_withNOB_valid = valid_data[['withNOBODY']].values.ravel()
    y_withNOB_test = test_data[['withNOBODY']].values.ravel()

    # Remove 'Occupant_Activity' and 'location' from train, valid, and test after encoding
    train_data = train_data.drop(columns=['Occupant_Activity', 'location',"withNOBODY"])
    valid_data = valid_data.drop(columns=['Occupant_Activity', 'location',"withNOBODY"])
    test_data = test_data.drop(columns=['Occupant_Activity', 'location',"withNOBODY"])

    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 48, train_data.shape[1])
    X_valid = valid_data.values.reshape(-1, 48, valid_data.shape[1])
    X_test = test_data.values.reshape(-1, 48, test_data.shape[1])

    # Reshape y's as required for PyTorch model
    y_activity_train = y_activity_train.reshape(-1, 48)
    y_activity_valid = y_activity_valid.reshape(-1, 48)
    y_activity_test = y_activity_test.reshape(-1, 48)

    y_location_train = y_location_train.reshape(-1, 48)
    y_location_valid = y_location_valid.reshape(-1, 48)
    y_location_test = y_location_test.reshape(-1, 48)

    y_withNOB_train = y_withNOB_train.reshape(-1, 48)
    y_withNOB_valid = y_withNOB_valid.reshape(-1, 48)
    y_withNOB_test = y_withNOB_test.reshape(-1, 48)

    # TESTING COLUMN ORDER --------------------------------------------------------------------------------------------
    #print("Column Order in X_train:", "\n".join(train_data.columns.tolist())) # Print column order after preprocessing

    return X_train, y_activity_train, y_location_train, y_withNOB_train,\
        X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,\
        X_test, y_activity_test, y_location_test, y_withNOB_test, \
        label_encoder

def preprocessLessEmbed(df):
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import LabelEncoder

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Initial Inspection
    grouped = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend'])
    to_remove = grouped.filter(lambda x: len(x) != 48).index
    df.drop(to_remove, inplace=True)

    # Integer encoding for the rest of the categorical columns
    # to keep the preprocessing consistent and simple, binary column of "gender" inserted in integer encoding
    one_hot_cols = ['months_season', 'week_or_weekend',]

    for col in one_hot_cols:
        df[col] = df[col].astype('category').cat.codes

    # Desired column order based on impact analysis
    impactAnalysis_order = ['Household_ID', 'Occupant_ID_in_HH',
                            'months_season', 'week_or_weekend',
                            'hourStart_Activity', 'hourEnd_Activity',
                            'Occupant_Activity',]
    df = df[impactAnalysis_order]

    # Suppress the SettingWithCopyWarning globally
    pd.options.mode.chained_assignment = None  # default='warn'

    # 3. Temporal Encoding
    time_cols = ['hourStart_Activity', 'hourEnd_Activity']
    df[[f'sin_{col}' for col in time_cols]] = np.sin(2 * np.pi * df[time_cols] / 24.0)
    df[[f'cos_{col}' for col in time_cols]] = np.cos(2 * np.pi * df[time_cols] / 24.0)
    # Optionally, you can reset the option back to the default warning state
    pd.options.mode.chained_assignment = 'warn'
    df = df.copy()
    df.drop(columns=time_cols, inplace=True)

    # 4. Normalization (except for the columns that will be used in embedding layers)
    scaler = RobustScaler() # if the data has outliers or is not normally distributed
    to_scale = df.columns.difference(one_hot_cols + ['Occupant_Activity'])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.3)
    valid_data_ids, test_data_ids = stratified_split(df.loc[df.index.isin(temp_data_ids.index)], test_size=0.5)

    train_data = df.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    valid_data = df.merge(valid_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    test_data = df.merge(test_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])

    # Drop 'Household_ID' column post splitting
    for dataset in [train_data, valid_data, test_data]:
        dataset.drop(columns=['Household_ID','Occupant_ID_in_HH' ], inplace=True)

    #ENCODING TARGET VARIABLE - LABEL ENCODING -------------------------------------------------------------------------
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    #unique_activities = df['Occupant_Activity'].unique().ravel()
    unique_activities = df['Occupant_Activity'].unique().reshape(-1, 1)
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_activities)

    # Convert the 'Occupant_Activity' of train, valid and test to one-hot encoded
    y_activity_train = label_encoder.transform(train_data[['Occupant_Activity']].values.ravel())
    y_activity_valid = label_encoder.transform(valid_data[['Occupant_Activity']].values.ravel())
    y_activity_test = label_encoder.transform(test_data[['Occupant_Activity']].values.ravel())

    # Remove 'Occupant_Activity' and 'location' from train, valid, and test after encoding
    train_data = train_data.drop(columns=['Occupant_Activity',])
    valid_data = valid_data.drop(columns=['Occupant_Activity',])
    test_data = test_data.drop(columns=['Occupant_Activity', ])

    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 48, train_data.shape[1])
    X_valid = valid_data.values.reshape(-1, 48, valid_data.shape[1])
    X_test = test_data.values.reshape(-1, 48, test_data.shape[1])

    # Reshape y's as required for PyTorch model
    y_activity_train = y_activity_train.reshape(-1, 48)
    y_activity_valid = y_activity_valid.reshape(-1, 48)
    y_activity_test = y_activity_test.reshape(-1, 48)

    # TESTING COLUMN ORDER --------------------------------------------------------------------------------------------
    #print("Column Order in X_train:", "\n".join(train_data.columns.tolist())) # Print column order after preprocessing

    return X_train, y_activity_train, X_valid, y_activity_valid,  X_test, y_activity_test, label_encoder

def data_preprocessNoEmbedding(df):
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import LabelEncoder

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Initial Inspection
    grouped = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend'])
    to_remove = grouped.filter(lambda x: len(x) != 48).index
    df = df.copy()
    df.drop(to_remove, inplace=True)
    #print("hello:", df.columns)

    # Suppress the SettingWithCopyWarning globally
    pd.options.mode.chained_assignment = None  # default='warn'
    # 3. Temporal Encoding
    time_cols = ['hourStart_Activity', 'hourEnd_Activity']
    df[[f'sin_{col}' for col in time_cols]] = np.sin(2 * np.pi * df[time_cols] / 24.0)
    df[[f'cos_{col}' for col in time_cols]] = np.cos(2 * np.pi * df[time_cols] / 24.0)
    # Optionally, you can reset the option back to the default warning state
    pd.options.mode.chained_assignment = 'warn'
    df = df.copy()
    df.drop(columns=time_cols, inplace=True)

    # 4. Normalization (except for the columns that will be used in embedding layers)
    scaler = RobustScaler() # if the data has outliers or is not normally distributed
    to_scale = df.columns.difference(['Occupant_Activity', "location", "withNOBODY"])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.3)
    valid_data_ids, test_data_ids = stratified_split(df.loc[df.index.isin(temp_data_ids.index)], test_size=0.5)

    train_data = df.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    valid_data = df.merge(valid_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    test_data = df.merge(test_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])

    # Drop 'Household_ID' column post splitting
    for dataset in [train_data, valid_data, test_data]:
        dataset.drop(columns=['Household_ID', "Occupant_ID_in_HH", 'months_season', 'week_or_weekend'], inplace=True)

    # TESTING DISTRIBUTIONS --------------------------------------------------------------------------------------------
    # Before splitting
    #activity_distribution_before, location_distribution_before, withnobody_distribution_before  = before_splitting(df)
    # After splitting
    #after_splitting(train_data, valid_data, test_data, activity_distribution_before, location_distribution_before, withnobody_distribution_before)

    #ENCODING TARGET VARIABLE - LABEL ENCODING -------------------------------------------------------------------------
    #unique_activities = df['Occupant_Activity'].unique().reshape(-1, 1)
    unique_activities = df['Occupant_Activity'].unique().ravel()

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_activities)

    # Convert the 'Occupant_Activity' of train, valid and test to one-hot encoded
    y_activity_train = label_encoder.transform(train_data[['Occupant_Activity']].values.ravel())
    y_activity_valid = label_encoder.transform(valid_data[['Occupant_Activity']].values.ravel())
    y_activity_test = label_encoder.transform(test_data[['Occupant_Activity']].values.ravel())

    # Use the 'location' column directly as a binary target
    y_location_train = train_data[['location']].values.ravel()
    y_location_valid = valid_data[['location']].values.ravel()
    y_location_test = test_data[['location']].values.ravel()

    # Use the 'withNOBODY' column directly as a binary target
    y_withNOB_train = train_data[['withNOBODY']].values.ravel()
    y_withNOB_valid = valid_data[['withNOBODY']].values.ravel()
    y_withNOB_test = test_data[['withNOBODY']].values.ravel()


    # Remove 'Occupant_Activity' and 'location' from train, valid, and test after encoding
    train_data = train_data.drop(columns=['Occupant_Activity', 'location',"withNOBODY"])
    valid_data = valid_data.drop(columns=['Occupant_Activity', 'location',"withNOBODY"])
    test_data = test_data.drop(columns=['Occupant_Activity', 'location',"withNOBODY"])

    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 48, train_data.shape[1])
    X_valid = valid_data.values.reshape(-1, 48, valid_data.shape[1])
    X_test = test_data.values.reshape(-1, 48, test_data.shape[1])

    # Reshape y's as required for PyTorch model
    y_activity_train = y_activity_train.reshape(-1, 48)
    y_activity_valid = y_activity_valid.reshape(-1, 48)
    y_activity_test = y_activity_test.reshape(-1, 48)

    y_location_train = y_location_train.reshape(-1, 48)
    y_location_valid = y_location_valid.reshape(-1, 48)
    y_location_test = y_location_test.reshape(-1, 48)

    y_withNOB_train = y_withNOB_train.reshape(-1, 48)
    y_withNOB_valid = y_withNOB_valid.reshape(-1, 48)
    y_withNOB_test = y_withNOB_test.reshape(-1, 48)

    # TESTING COLUMN ORDER --------------------------------------------------------------------------------------------
    #print("Column Order in X_train:", "\n".join(train_data.columns.tolist())) # Print column order after preprocessing

    return X_train, y_activity_train, y_location_train, y_withNOB_train,\
        X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,\
        X_test, y_activity_test, y_location_test, y_withNOB_test, \
        label_encoder

def preprocessNoEmbed_Simpler(df):
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import LabelEncoder

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Initial Inspection
    grouped = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend'])
    to_remove = grouped.filter(lambda x: len(x) != 48).index
    df = df.copy()
    df.drop(to_remove, inplace=True)
    #print("hello:", df.columns)

    # Suppress the SettingWithCopyWarning globally
    pd.options.mode.chained_assignment = None  # default='warn'
    # 3. Temporal Encoding
    time_cols = ['hourStart_Activity', 'hourEnd_Activity']
    df[[f'sin_{col}' for col in time_cols]] = np.sin(2 * np.pi * df[time_cols] / 24.0)
    df[[f'cos_{col}' for col in time_cols]] = np.cos(2 * np.pi * df[time_cols] / 24.0)
    # Optionally, you can reset the option back to the default warning state
    pd.options.mode.chained_assignment = 'warn'
    df = df.copy()
    df.drop(columns=time_cols, inplace=True)

    # 4. Normalization (except for the columns that will be used in embedding layers)
    scaler = RobustScaler() # if the data has outliers or is not normally distributed
    to_scale = df.columns.difference(['Occupant_Activity', "location"])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.3)
    valid_data_ids, test_data_ids = stratified_split(df.loc[df.index.isin(temp_data_ids.index)], test_size=0.5)

    train_data = df.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    valid_data = df.merge(valid_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    test_data = df.merge(test_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])

    # Drop 'Household_ID' column post splitting
    for dataset in [train_data, valid_data, test_data]:
        dataset.drop(columns=['Household_ID', "Occupant_ID_in_HH", 'months_season', 'week_or_weekend'], inplace=True)

    #ENCODING TARGET VARIABLE - LABEL ENCODING -------------------------------------------------------------------------
    unique_activities = df['Occupant_Activity'].unique().ravel()

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_activities)

    # Convert the 'Occupant_Activity' of train, valid and test to one-hot encoded
    y_activity_train = label_encoder.transform(train_data[['Occupant_Activity']].values.ravel())
    y_activity_valid = label_encoder.transform(valid_data[['Occupant_Activity']].values.ravel())
    y_activity_test = label_encoder.transform(test_data[['Occupant_Activity']].values.ravel())

    # Use the 'location' column directly as a binary target
    y_location_train = train_data[['location']].values.ravel()
    y_location_valid = valid_data[['location']].values.ravel()
    y_location_test = test_data[['location']].values.ravel()

    # Remove 'Occupant_Activity' and 'location' from train, valid, and test after encoding
    train_data = train_data.drop(columns=['Occupant_Activity', 'location'])
    valid_data = valid_data.drop(columns=['Occupant_Activity', 'location'])
    test_data = test_data.drop(columns=['Occupant_Activity', 'location'])

    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 48, train_data.shape[1])
    X_valid = valid_data.values.reshape(-1, 48, valid_data.shape[1])
    X_test = test_data.values.reshape(-1, 48, test_data.shape[1])

    # Reshape y's as required for PyTorch model
    y_activity_train = y_activity_train.reshape(-1, 48)
    y_activity_valid = y_activity_valid.reshape(-1, 48)
    y_activity_test = y_activity_test.reshape(-1, 48)

    y_location_train = y_location_train.reshape(-1, 48)
    y_location_valid = y_location_valid.reshape(-1, 48)
    y_location_test = y_location_test.reshape(-1, 48)

    # TESTING COLUMN ORDER --------------------------------------------------------------------------------------------
    #print("Column Order in X_train:", "\n".join(train_data.columns.tolist())) # Print column order after preprocessing

    return X_train, y_activity_train, y_location_train, X_valid, y_activity_valid, y_location_valid, X_test, y_activity_test, y_location_test, label_encoder

def preprocessNoEmbed_Simplest(df):
    import numpy as np
    from sklearn.preprocessing import RobustScaler

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Initial Inspection
    grouped = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend'])
    to_remove = grouped.filter(lambda x: len(x) != 48).index
    df = df.copy()
    df.drop(to_remove, inplace=True)

    # Suppress the SettingWithCopyWarning globally
    pd.options.mode.chained_assignment = None  # default='warn'
    # 3. Temporal Encoding
    time_cols = ['hourStart_Activity', 'hourEnd_Activity']
    df[[f'sin_{col}' for col in time_cols]] = np.sin(2 * np.pi * df[time_cols] / 24.0)
    df[[f'cos_{col}' for col in time_cols]] = np.cos(2 * np.pi * df[time_cols] / 24.0)
    pd.options.mode.chained_assignment = 'warn'     #reset the option back to the default warning state
    df = df.copy()
    df.drop(columns=time_cols, inplace=True)

    # 4. Normalization (except for the columns that will be used in embedding layers)
    scaler = RobustScaler()
    to_scale = df.columns.difference(["location"])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.3)
    valid_data_ids, test_data_ids = stratified_split(df.loc[df.index.isin(temp_data_ids.index)], test_size=0.5)

    train_data = df.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    valid_data = df.merge(valid_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    test_data = df.merge(test_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])

    # Drop 'Household_ID' column post splitting
    for dataset in [train_data, valid_data, test_data]:
        dataset.drop(columns=['Household_ID', "Occupant_ID_in_HH", 'months_season', 'week_or_weekend'], inplace=True)

    # Use the 'location' column directly as a binary target
    y_location_train = train_data[['location']].values.ravel()
    y_location_valid = valid_data[['location']].values.ravel()
    y_location_test = test_data[['location']].values.ravel()

    # Remove 'Occupant_Activity' and 'location' from train, valid, and test after encoding
    train_data = train_data.drop(columns=['location'])
    valid_data = valid_data.drop(columns=['location'])
    test_data = test_data.drop(columns=['location'])

    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 48, train_data.shape[1])
    X_valid = valid_data.values.reshape(-1, 48, valid_data.shape[1])
    X_test = test_data.values.reshape(-1, 48, test_data.shape[1])

    y_location_train = y_location_train.reshape(-1, 48)
    y_location_valid = y_location_valid.reshape(-1, 48)
    y_location_test = y_location_test.reshape(-1, 48)

    # TESTING COLUMN ORDER --------------------------------------------------------------------------------------------
    #print("Column Order in X_train:", "\n".join(train_data.columns.tolist())) # Print column order after preprocessing

    return X_train, y_location_train, X_valid, y_location_valid, X_test, y_location_test


#PREPROCESS FOR TUNING -------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder

def stratified_k_fold_split(df, n_splits=5, random_state=42):
    df_copy = df.copy()
    df_copy['strata'] = df_copy['months_season'].astype(str) + "_" + df_copy['week_or_weekend'].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for train_index, test_index in skf.split(df_copy, df_copy['strata']):
        train_data_ids = df_copy.iloc[train_index][['Household_ID', 'Occupant_ID_in_HH']].drop_duplicates()
        test_data_ids = df_copy.iloc[test_index][['Household_ID', 'Occupant_ID_in_HH']].drop_duplicates()
        folds.append((train_data_ids, test_data_ids))

    df_copy.drop('strata', axis=1, inplace=True)
    return folds

def data_preprocess_k_fold_split(df, n_splits=3):
    # Suppress the DataConversionWarning, RuntimeWarning
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 1. Initial Inspection
    grouped = df.groupby(['Household_ID', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend'])
    to_remove = grouped.filter(lambda x: len(x) != 48).index
    df.drop(to_remove, inplace=True)

    # Integer encoding for the rest of the categorical columns
    one_hot_cols = ['Education Degree', 'Employment status', 'Family Typology',  "Gender",
                    'Number Family Members', 'Occupant_ID_in_HH', 'months_season', 'week_or_weekend']

    for col in one_hot_cols:
        df[col] = df[col].astype('category').cat.codes

    # Desired column order based on impact analysis
    impactAnalysis_order = ['Household_ID', 'Education Degree', 'Employment status', 'Gender',
                            'Family Typology','Number Family Members', 'Occupant_ID_in_HH',
                            'months_season', 'week_or_weekend', 'hourStart_Activity', 'hourEnd_Activity',
                            'Occupant_Activity', "location", "withNOBODY"]
    df = df[impactAnalysis_order]

    # 3. Temporal Encoding
    # Suppress the SettingWithCopyWarning globally
    pd.options.mode.chained_assignment = None  # default='warn'
    # 3. Temporal Encoding
    time_cols = ['hourStart_Activity', 'hourEnd_Activity']
    df[[f'sin_{col}' for col in time_cols]] = np.sin(2 * np.pi * df[time_cols] / 24.0)
    df[[f'cos_{col}' for col in time_cols]] = np.cos(2 * np.pi * df[time_cols] / 24.0)
    # Optionally, you can reset the option back to the default warning state
    pd.options.mode.chained_assignment = 'warn'
    # To avoid SettingWithCopyWarning, create a copy of the DataFrame
    df = df.copy()
    df.drop(columns=time_cols, inplace=True)

    # 4. Normalization (except for the columns that will be used in embedding layers)
    scaler = RobustScaler()
    to_scale = df.columns.difference(one_hot_cols + ['Occupant_Activity', "location", "withNOBODY", "Gender"])
    # Using .loc to avoid SettingWithCopyWarning
    df.loc[:, to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on k-fold cross-validation
    folds = stratified_k_fold_split(df, n_splits=n_splits)

    # Returning the folds instead of train, valid, test splits
    return folds, df

# TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING
def print_test(X_train,X_valid,X_test,
               y_activity_train,y_location_train,y_withNOB_train,
               y_activity_valid, y_location_valid, y_withNOB_valid,
               y_activity_test, y_location_test, y_withNOB_test ):
    #print("\n".join(map(str, X_train[:1])))  # pring first 1 sample
    #print(" ")  # pring first 10 sample

    #print("\n".join(map(str, y_activity_train[:3])))  # pring first 10 sample
    #print("\n".join(map(str, y_location_train[:3]))) #pring first 10 sample
    #print("\n".join(map(str, y_withNOB_train[:3])))  # pring first 10 sample

    print(X_train.shape, y_activity_train.shape, y_location_train.shape,  y_withNOB_train.shape)
    print(X_valid.shape, y_activity_valid.shape, y_location_valid.shape, y_withNOB_valid.shape)
    print(X_test.shape, y_activity_test.shape, y_location_test.shape, y_withNOB_test.shape)

#PREPROCESS - TESTING---------------------------------------------------------------------------------------------------
# Functions
def calculate_distribution(df, target_column):
    return df[target_column].value_counts(normalize=True)

def calculate_split_distributions(train_df, valid_df, test_df, target_column):
    train_distribution = calculate_distribution(train_df, target_column)
    valid_distribution = calculate_distribution(valid_df, target_column)
    test_distribution = calculate_distribution(test_df, target_column)

    return train_distribution, valid_distribution, test_distribution

def compare_distributions(before, after_train, after_valid, after_test, target_name):
    before_df = before.reset_index()
    before_df.columns = ['Category', 'Before_Split']

    train_df = after_train.reset_index()
    train_df.columns = ['Category', 'Train_Split']

    valid_df = after_valid.reset_index()
    valid_df.columns = ['Category', 'Valid_Split']

    test_df = after_test.reset_index()
    test_df.columns = ['Category', 'Test_Split']

    comparison_df = pd.merge(before_df, train_df, on='Category', how='outer')
    comparison_df = pd.merge(comparison_df, valid_df, on='Category', how='outer')
    comparison_df = pd.merge(comparison_df, test_df, on='Category', how='outer')

    comparison_df.to_csv(f'{target_name}_distribution_comparison.csv', index=False)
    print(f"{target_name} distribution comparison saved to {target_name}_distribution_comparison.csv")
#------------------------------------------------------
# Functions for Application
# Before splitting
def before_splitting(df):
    activity_distribution_before = calculate_distribution(df, 'Occupant_Activity')
    location_distribution_before = calculate_distribution(df, 'location')
    withnobody_distribution_before = calculate_distribution(df, 'withNOBODY')
    return activity_distribution_before, location_distribution_before, withnobody_distribution_before

# After splitting
def after_splitting(train_data, valid_data, test_data, activity_distribution_before,
                    location_distribution_before, withnobody_distribution_before):
    activity_train_distribution, activity_valid_distribution, activity_test_distribution = calculate_split_distributions(
        train_data, valid_data, test_data, 'Occupant_Activity')
    location_train_distribution, location_valid_distribution, location_test_distribution = calculate_split_distributions(
        train_data, valid_data, test_data, 'location')
    with_train_distribution, with_valid_distribution, with_test_distribution = calculate_split_distributions(train_data,
                                                                                                             valid_data,
                                                                                                             test_data,
                                                                                                             'withNOBODY')

    compare_distributions(activity_distribution_before, activity_train_distribution, activity_valid_distribution,
                          activity_test_distribution, 'Occupant_Activity')
    compare_distributions(location_distribution_before, location_train_distribution, location_valid_distribution,
                          location_test_distribution, 'Location')
    compare_distributions(withnobody_distribution_before, with_train_distribution, with_valid_distribution,
                          with_test_distribution, 'WithNOBODY')

    # Plot the distributions
    plot_distribution(activity_distribution_before, activity_train_distribution, activity_valid_distribution,
                      activity_test_distribution, 'Occupant_Activity')
    plot_distribution(location_distribution_before, location_train_distribution, location_valid_distribution,
                      location_test_distribution, 'Location')
    plot_distribution(withnobody_distribution_before, with_train_distribution, with_valid_distribution,
                      with_test_distribution, 'WithNOBODY')

# VISUALISATION OF THE RESULTS
def plot_distribution(before, after_train, after_valid, after_test, target_name):
    import matplotlib.pyplot as plt
    if target_name == 'Occupant_Activity':
        fig, axes = plt.subplots(4, 1, figsize=(25, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    #fig.suptitle(f'Distribution Comparison for {target_name}', fontsize=8)

    if target_name == 'Occupant_Activity':
        axes = axes.flatten()  # Flatten the 1D array of axes
        before.plot(kind='bar', ax=axes[0], title='Before Splitting', color='blue')
        after_train.plot(kind='bar', ax=axes[1], title='After Splitting - Train', color='green')
        after_valid.plot(kind='bar', ax=axes[2], title='After Splitting - Valid', color='orange')
        after_test.plot(kind='bar', ax=axes[3], title='After Splitting - Test', color='red')
    else:
        before.plot(kind='bar', ax=axes[0, 0], title='Before Splitting', color='blue')
        after_train.plot(kind='bar', ax=axes[0, 1], title='After Splitting - Train', color='green')
        after_valid.plot(kind='bar', ax=axes[1, 0], title='After Splitting - Valid', color='orange')
        after_test.plot(kind='bar', ax=axes[1, 1], title='After Splitting - Test', color='red')

    for ax in axes.flatten():
        ax.set_xlabel(target_name)
        ax.set_ylabel('Proportion')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)  # Adjust fontsize here

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
