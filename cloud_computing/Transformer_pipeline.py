# EXTRA_________________________________________________________________________________________________________________
def check_and_log_gradients(model, folder_path, epoch):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Extract the gradient mean and std
            gradient_mean = param.grad.mean().item()
            gradient_std = param.grad.std().item()

            # Write the gradient info to a text file
            filename = f"{name.replace('.', '_')}.txt"
            filepath = os.path.join(folder_path, filename)

            # Append the gradient info to the text file for each epoch
            with open(filepath, 'a') as f:
                f.write(f"Epoch {epoch} - Gradient for {name}: mean={gradient_mean}, std={gradient_std}\n")
def plot_gradients(folder_path):
    import re
    # List all gradient log files in the folder
    gradient_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for filename in gradient_files:
        filepath = os.path.join(folder_path, filename)
        epochs = []
        means = []
        stds = []

        # Read the gradient values from the file
        with open(filepath, 'r') as f:
            for line in f:
                match = re.match(r"Epoch (\d+) - Gradient for .*: mean=(-?[\d\.eE-]+), std=([\d\.eE-]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    means.append(float(match.group(2)))
                    stds.append(float(match.group(3)))

        # Plot the mean and std of gradients over epochs
        if epochs:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, means, label='Gradient Mean')
            plt.plot(epochs, stds, label='Gradient Std')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title(f'Gradient Progress for {filename.replace("_", ".").replace(".txt", "")}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(folder_path, f'{filename.replace(".txt", "_plot.png")}'))
            plt.close()
def record_trial_parameters(trial, model, hyperparameters, output_csv='Trans_tuning_params.csv'):
    import csv
    total_params = sum(p.numel() for p in model.parameters())

    # Combine hyperparameters with the trial data
    trial_data = {
        'trial_number': trial.number,
        'total_params': total_params,
        'hyperparameters': str(hyperparameters)
    }

    # Append the data to the CSV file
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=trial_data.keys())
        if file.tell() == 0:  # Write header only if file is empty
            writer.writeheader()
        writer.writerow(trial_data)
def save_trial_details(study, filename_prefix="TRresults"):
    import os
    import csv
    # Create directory to store the outputs
    folder_name = "TransTuningcsvOutputs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Prepare full paths for the .txt and .csv files
    txt_file_path = os.path.join(folder_name, f"{filename_prefix}_all_trials.txt")
    csv_file_path = os.path.join(folder_name, f"{filename_prefix}_all_trials.csv")

    # Save details of all trials to a .txt file
    with open(txt_file_path, 'w') as f:
        f.write('All trial results:\n')
        for trial in study.trials:
            f.write(f"Iter: {trial.number}, Target: {trial.value:.4f}, Params: {trial.params}, ")
            f.write(f"Activity Accuracy: {trial.user_attrs.get('activity_accuracy', 'N/A'):.4f}, ")
            f.write(f"Location Accuracy: {trial.user_attrs.get('location_accuracy', 'N/A'):.4f}, ")
            f.write(f"WithNOB Accuracy: {trial.user_attrs.get('withNOB_accuracy', 'N/A'):.4f}, ")
            f.write(f"Last epoch: {trial.user_attrs.get('last_epoch', 'N/A')}\n")

    # Collect all possible keys for fieldnames
    all_params_keys = set()
    for trial in study.trials:
        all_params_keys.update(trial.params.keys())

    # Save details of all trials to a .csv file
    # Define fieldnames for the CSV
    fieldnames = ['iter', 'target', 'activity_accuracy', 'location_accuracy', 'withNOB_accuracy', 'last_epoch']  + list(
        all_params_keys)

    # Save details of all trials to a .csv file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                'iter': trial.number,
                'target': trial.value,
                'activity_accuracy': f"{trial.user_attrs.get('activity_accuracy', 'N/A'):.2f}" if trial.user_attrs.get(
                    'activity_accuracy') is not None else 'N/A',
                'location_accuracy': f"{trial.user_attrs.get('location_accuracy', 'N/A'):.2f}" if trial.user_attrs.get(
                    'location_accuracy') is not None else 'N/A',
                'withNOB_accuracy': f"{trial.user_attrs.get('withNOB_accuracy', 'N/A'):.2f}" if trial.user_attrs.get(
                    'withNOB_accuracy') is not None else 'N/A',
                'last_epoch': trial.user_attrs.get('last_epoch', 'N/A')  # Add the last_epoch here
            }
            row.update(trial.params)
            writer.writerow(row)

    print(f"Results saved to {txt_file_path} and {csv_file_path}")

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
    to_remove = grouped.filter(lambda x: len(x) != 24).index
    df.drop(to_remove, inplace=True)

    # Integer encoding for the rest of the categorical columns
    # to keep the preprocessing consistent and simple, binary column of "gender" inserted in integer encoding
    one_hot_cols = ['Education Degree',
                    'Employment status',
                    'Age Classes',
                    'Region',
                    'Family Typology',
                    'Number Family Members',
                    "Marital Status",
                    "Kinship Relationship",
                    "Nuclear Family, Occupant Profile",
                    "Nuclear Family, Typology",
                    "Nuclear Family, Occupant Sequence Number",
                    "Citizenship",  # new additions
                    "Internet Access",
                    "Mobile Phone Ownership",
                    "Car Ownership",
                    'Gender',
                    'Family_Typology_Simple', 'Home Ownership', 'Room Count', 'Economic Sector, Profession', 'Job Type', # new alignment columns
                    'Occupant_ID_in_HH',
                    'months_season',
                    'week_or_weekend',]

    for col in one_hot_cols:
        df[col] = df[col].astype('category').cat.codes

    # Desired column order based on impact analysis
    impactAnalysis_order = ['Household_ID', 'Education Degree','Employment status', 'Gender', 'Family Typology','Number Family Members',
                            'Age Classes', 'Region',
                            "Marital Status",
                            "Kinship Relationship",
                            "Nuclear Family, Occupant Profile",
                            "Nuclear Family, Typology",
                            "Nuclear Family, Occupant Sequence Number",
                            "Citizenship",  # new additions
                            "Internet Access",
                            "Mobile Phone Ownership",
                            "Car Ownership",
                            'Family_Typology_Simple', 'Home Ownership', 'Room Count', 'Economic Sector, Profession', 'Job Type',  # new alignment columns
                            'Occupant_ID_in_HH',
                            'months_season', 'week_or_weekend', 'hourStart_Activity', 'hourEnd_Activity',
                            'Occupant_Activity',"location", "withNOBODY",]
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
    # no need to scale binary columns thus they are added in here
    scaler = RobustScaler() # if the data has outliers or is not normally distributed
    to_scale = df.columns.difference(one_hot_cols + ['Occupant_Activity', "location", "withNOBODY", "Gender", "Citizenship",  # new additions
                    "Internet Access",
                    "Mobile Phone Ownership",
                    "Car Ownership",])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.5) #0.3 default
    valid_data_ids, test_data_ids = stratified_split(df.loc[df.index.isin(temp_data_ids.index)], test_size=0.5) #0.5 default

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
    train_data = train_data.drop(columns=['Occupant_Activity', 'location', "withNOBODY"])
    valid_data = valid_data.drop(columns=['Occupant_Activity', 'location', "withNOBODY"])
    test_data = test_data.drop(columns=['Occupant_Activity', 'location', "withNOBODY"])

    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 24, train_data.shape[1])
    X_valid = valid_data.values.reshape(-1, 24, valid_data.shape[1])
    X_test = test_data.values.reshape(-1, 24, test_data.shape[1])

    # Reshape y's as required for PyTorch model
    y_activity_train = y_activity_train.reshape(-1, 24)
    y_activity_valid = y_activity_valid.reshape(-1, 24)
    y_activity_test = y_activity_test.reshape(-1, 24)

    y_location_train = y_location_train.reshape(-1, 24)
    y_location_valid = y_location_valid.reshape(-1, 24)
    y_location_test = y_location_test.reshape(-1, 24)

    y_withNOB_train = y_withNOB_train.reshape(-1, 24)
    y_withNOB_valid = y_withNOB_valid.reshape(-1, 24)
    y_withNOB_test = y_withNOB_test.reshape(-1, 24)

    # TESTING COLUMN ORDER --------------------------------------------------------------------------------------------
    #print("Column Order in X_train:", "\n".join(train_data.columns.tolist())) # Print column order after preprocessing

    return X_train, y_activity_train, y_location_train, y_withNOB_train, \
        X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
        X_test, y_activity_test, y_location_test, y_withNOB_test, \
        label_encoder

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
    to_remove = grouped.filter(lambda x: len(x) != 24).index
    df.drop(to_remove, inplace=True)

    # Integer encoding for the rest of the categorical columns
    one_hot_cols = ['Education Degree', 'Employment status', 'Family Typology',  "Gender",
                    'Number Family Members','Age Classes', 'Region',
                    "Marital Status",  "Kinship Relationship",
                    "Nuclear Family, Occupant Profile",
                    "Nuclear Family, Typology",
                    "Nuclear Family, Occupant Sequence Number",
                    "Citizenship",  # new additions
                    "Internet Access",
                    "Mobile Phone Ownership",
                    "Car Ownership",
                    'Family_Typology_Simple', 'Home Ownership', 'Room Count', 'Economic Sector, Profession', 'Job Type', # new alignment columns
                    'Occupant_ID_in_HH', 'months_season', 'week_or_weekend']

    for col in one_hot_cols:
        df[col] = df[col].astype('category').cat.codes

    # Desired column order based on impact analysis
    impactAnalysis_order = ['Household_ID', 'Education Degree', 'Employment status', 'Gender',
                            'Family Typology','Number Family Members',
                            'Age Classes', 'Region',
                            "Marital Status",
                            "Kinship Relationship",
                            "Nuclear Family, Occupant Profile",
                            "Nuclear Family, Typology",
                            "Nuclear Family, Occupant Sequence Number",
                            "Citizenship",  # new additions
                            "Internet Access",
                            "Mobile Phone Ownership",
                            "Car Ownership",
                            'Family_Typology_Simple', 'Home Ownership', 'Room Count', 'Economic Sector, Profession', 'Job Type',  # new alignment columns
                            'Occupant_ID_in_HH',
                            'months_season', 'week_or_weekend', 'hourStart_Activity', 'hourEnd_Activity',
                            'Occupant_Activity', "location","withNOBODY"]
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
    to_scale = df.columns.difference(one_hot_cols + ['Occupant_Activity', "location",  "withNOBODY", "Gender", "Citizenship",  # new additions
                    "Internet Access",
                    "Mobile Phone Ownership",
                    "Car Ownership",])
    # Using .loc to avoid SettingWithCopyWarning
    df.loc[:, to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on k-fold cross-validation
    folds = stratified_k_fold_split(df, n_splits=n_splits)

    # Returning the folds instead of train, valid, test splits
    return folds, df
# TRANSFORMER: EXTRA ---------------------------------------------------------------------------------------------------
import math
import torch.nn as nn
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(LearnablePositionalEncoding, self).__init__()
        # Trainable positional embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(seq_len, embed_dim))
        # Initialize with small random values to avoid zeros initially
        nn.init.normal_(self.positional_embedding, mean=0, std=0.02)

    def forward(self, x):
        # Add the trainable positional embeddings
        return x + self.positional_embedding[:x.size(1), :].unsqueeze(0)  # Batch-wise addition
#MODELING---------------------------------------------------------------------------------------------------------------
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TransformerModelTuning(nn.Module):
    def __init__(self,num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat,num_numFamMembCat,
                 num_ageClassCat, num_regionCat,
                 num_MartStatCat, num_KinsCat, num_OccProfCat, num_FamTypoCat, num_OccSeqNumCat,
                 num_CitizenCat,  num_InterOwnCat, num_MobPhoneOwnCat, num_CarOwnCat,
                 num_FamTypoSimpleCat, num_HomeOwnCat, num_RoomCountCat, num_EcoSectorCat, num_JobTypeCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers,
                 dropout_loc, dropout_withNOB, dropout_embedding, dropout_transformer,
                 embed_size,
                 nhead, d_feed):

        super(TransformerModelTuning, self).__init__()
        self.nhead = nhead
        self.num_continuous_features = num_continuous_features
        embed_size = embed_size

        self.activation = nn.ReLU()
        self.activation_act = nn.ReLU()
        self.activation_binary = nn.Tanh()

        # Define embedding dimensions for all categorical features
        # Occupant Demographics: Default
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 2)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 2)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 2)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 2)
        self.embedding_dim_ageClass = min(embed_size, num_ageClassCat // 2 + 2)
        self.embedding_dim_region = min(embed_size, num_regionCat // 2 + 2)
        # Occupant Demographics: Added
        self.embedding_dim_MartStat = min(embed_size, num_MartStatCat // 2 + 1)
        self.embedding_dim_Kins = min(embed_size, num_KinsCat // 2 + 1)
        self.embedding_dim_OccProf = min(embed_size, num_OccProfCat // 2 + 1)
        self.embedding_dim_FamTypo = min(embed_size, num_FamTypoCat // 2 + 1)
        self.embedding_dim_OccSeqNum = min(embed_size, num_OccSeqNumCat // 2 + 1)
        # Occupant Demographics: Added 2
        self.embedding_dim_Citizen = min(embed_size, num_CitizenCat // 2 + 1)
        self.embedding_dim_InterOwn = min(embed_size, num_InterOwnCat // 2 + 1)
        self.embedding_dim_MobPhoneOwn = min(embed_size, num_MobPhoneOwnCat // 2 + 1)
        self.embedding_dim_CarOwn = min(embed_size, num_CarOwnCat // 2 + 1)
        # Occupant Demographics: Added 3
        self.embedding_dim_FamTypoSimple = min(embed_size, num_FamTypoSimpleCat // 2 + 1)
        self.embedding_dim_HomeOwn = min(embed_size, num_HomeOwnCat // 2 + 1)
        self.embedding_dim_RoomCount = min(embed_size, num_RoomCountCat // 2 + 1)
        self.embedding_dim_EcoSector = min(embed_size, num_EcoSectorCat // 2 + 1)
        self.embedding_dim_JobType = min(embed_size, num_JobTypeCat // 2 + 1)
        # Order columns
        self.embedding_dim_OCCinHH = min(embed_size, num_OCCinHHCat // 2 + 1)
        # non-temporal TUS daily features
        self.embedding_dim_season = min(embed_size, num_seasonCat // 2 + 1)
        self.embedding_dim_weekend = min(embed_size, num_unique_weekCat // 2 + 1)

        # Embedding layers for each categorical feature
        # Occupant Demographics: Default
        self.education_embedding = nn.Embedding(num_educationCat, self.embedding_dim_education)
        self.employment_embedding = nn.Embedding(num_employmentCat, self.embedding_dim_employment)
        self.gender_embedding = nn.Embedding(num_genderCat, self.embedding_dim_gender)
        self.famTypology_embedding = nn.Embedding(num_famTypologyCat, self.embedding_dim_famTypology)
        self.numFamMemb_embedding = nn.Embedding(num_numFamMembCat, self.embedding_dim_numFamMemb)
        self.ageClass_embedding = nn.Embedding(num_ageClassCat, self.embedding_dim_ageClass)
        self.region_embedding = nn.Embedding(num_regionCat, self.embedding_dim_region)
        # Occupant Demographics: Added
        self.MartStat_embedding = nn.Embedding(num_MartStatCat, self.embedding_dim_MartStat)
        self.Kins_embedding = nn.Embedding(num_KinsCat, self.embedding_dim_Kins)
        self.OccProf_embedding = nn.Embedding(num_OccProfCat, self.embedding_dim_OccProf)
        self.FamTypo_embedding = nn.Embedding(num_FamTypoCat, self.embedding_dim_FamTypo)
        self.OccSeqNum_embedding = nn.Embedding(num_OccSeqNumCat, self.embedding_dim_OccSeqNum)
        # Occupant Demographics: Added 2
        self.Citizen_embedding = nn.Embedding(num_CitizenCat, self.embedding_dim_Citizen)
        self.InterOwn_embedding = nn.Embedding(num_InterOwnCat, self.embedding_dim_InterOwn)
        self.MobPhoneOwn_embedding = nn.Embedding(num_MobPhoneOwnCat, self.embedding_dim_MobPhoneOwn)
        self.CarOwn_embedding = nn.Embedding(num_CarOwnCat, self.embedding_dim_CarOwn)
        # Occupant Demographics: Added 3
        self.FamTypoSimple_embedding = nn.Embedding(num_FamTypoSimpleCat, self.embedding_dim_FamTypoSimple)
        self.HomeOwn_embedding = nn.Embedding(num_HomeOwnCat, self.embedding_dim_HomeOwn)
        self.RoomCount_embedding = nn.Embedding(num_RoomCountCat, self.embedding_dim_RoomCount)
        self.EcoSector_embedding = nn.Embedding(num_EcoSectorCat, self.embedding_dim_EcoSector)
        self.JobType_embedding = nn.Embedding(num_JobTypeCat, self.embedding_dim_JobType)
        # Order columns
        self.OCCinHH_embedding = nn.Embedding(num_OCCinHHCat, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        self.season_embedding = nn.Embedding(num_seasonCat, self.embedding_dim_season)
        self.weekend_embedding = nn.Embedding(num_unique_weekCat, self.embedding_dim_weekend)

        # Dropout layers for embeddings
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)

        # Calculate the total input size for the LSTM layers
        total_embedding_dim = sum([
            # Occupant Demographics
            self.embedding_dim_education,
            self.embedding_dim_employment,
            self.embedding_dim_gender,
            self.embedding_dim_famTypology,
            self.embedding_dim_numFamMemb,
            self.embedding_dim_ageClass,
            self.embedding_dim_region,
            self.embedding_dim_MartStat,
            self.embedding_dim_Kins,
            self.embedding_dim_OccProf,
            self.embedding_dim_FamTypo,
            self.embedding_dim_OccSeqNum,
            # Demographic columns: Added 2
            self.embedding_dim_Citizen,
            self.embedding_dim_InterOwn,
            self.embedding_dim_MobPhoneOwn,
            self.embedding_dim_CarOwn,
            # Demographic columns: Added 3
            self.embedding_dim_FamTypoSimple,
            self.embedding_dim_HomeOwn,
            self.embedding_dim_RoomCount,
            self.embedding_dim_EcoSector,
            self.embedding_dim_JobType,
            # Order columns
            self.embedding_dim_OCCinHH,
            # non-temporal TUS daily features
            self.embedding_dim_season,
            self.embedding_dim_weekend,
        ])

        input_size = total_embedding_dim + num_continuous_features
        #print("input_size:", input_size)

        # Ensure input_size is divisible by nhead
        if input_size % self.nhead != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by nhead ({self.nhead}).")

        #self.positional_encoding = PositionalEncoding(input_size)
        self.positional_encoding = LearnablePositionalEncoding(seq_len=20000, embed_dim=input_size)

        # Transformer encoder layer initialization
        self.encoder_layer = TransformerEncoderLayer(d_model=input_size,nhead=self.nhead,dropout=dropout_transformer,
                                                     activation=self.activation, batch_first=True, dim_feedforward=d_feed) # multi-head attention mechanism & residual connections & Feed-Forward Network

        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_hidden_layers,)

        # Batch Normalization
        #self.batch_norm = nn.BatchNorm1d(input_size)
        # Layer Normalization
        self.layer_norm = LayerNorm(input_size)

        # Output layers for each task
        self.activity_dense = nn.Linear(input_size, output_dim_activity)
        self.location_dense = nn.Linear(input_size, output_dim_location)
        self.withNOB_dense = nn.Linear(input_size, output_dim_withNOB)

        # Dropout layers for output
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)

    def forward(self,  education_input, employment_input, gender_input, famTypology_input, numFamMemb_input,
                ageClass_input, region_input,
                MartStat_input, Kins_input, OccProf_input, FamTypo_input, OccSeqNum_input,
                Citizen_input, InterOwn_input, MobPhoneOwn_input,CarOwn_input,
                FamTypoSimple_input, HomeOwn_input, RoomCount_input, EcoSector_input, JobType_input,
                OCCinHH_input,
                season_input, weekend_input,
                continuous_input):
        # Embeddings:Default Demographcis
        education_embedded = self.education_embedding(education_input).reshape(-1, 24, self.embedding_dim_education)
        employment_embedded = self.employment_embedding(employment_input).reshape(-1, 24, self.embedding_dim_employment)
        gender_embedded = self.gender_embedding(gender_input).reshape(-1, 24, self.embedding_dim_gender)
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 24,self.embedding_dim_famTypology)
        numFamMemb_embedded = self.numFamMemb_embedding(numFamMemb_input).reshape(-1, 24, self.embedding_dim_numFamMemb)
        ageClass_embedded = self.ageClass_embedding(ageClass_input).reshape(-1, 24, self.embedding_dim_ageClass)
        region_embedded = self.region_embedding(region_input).reshape(-1, 24, self.embedding_dim_region)
        # Embeddings:Added Demographics
        MartStat_embedded = self.MartStat_embedding(MartStat_input).reshape(-1, 24, self.embedding_dim_MartStat)
        Kins_embedded = self.Kins_embedding(Kins_input).reshape(-1, 24, self.embedding_dim_Kins)
        OccProf_embedded = self.OccProf_embedding(OccProf_input).reshape(-1, 24, self.embedding_dim_OccProf)
        FamTypo_embedded = self.FamTypo_embedding(FamTypo_input).reshape(-1, 24, self.embedding_dim_FamTypo)
        OccSeqNum_embedded = self.OccSeqNum_embedding(OccSeqNum_input).reshape(-1, 24, self.embedding_dim_OccSeqNum)
        # Embeddings:Added Demographics 2
        Citizen_embedded = self.Citizen_embedding(Citizen_input).reshape(-1, 24, self.embedding_dim_Citizen)
        InterOwn_embedded = self.InterOwn_embedding(InterOwn_input).reshape(-1, 24, self.embedding_dim_InterOwn)
        MobPhoneOwn_embedded = self.MobPhoneOwn_embedding(MobPhoneOwn_input).reshape(-1, 24, self.embedding_dim_MobPhoneOwn)
        CarOwn_embedded = self.CarOwn_embedding(CarOwn_input).reshape(-1, 24, self.embedding_dim_CarOwn)
        # Embeddings:Added Demographics 3
        FamTypoSimple_embedded = self.FamTypoSimple_embedding(FamTypoSimple_input).reshape(-1, 24, self.embedding_dim_FamTypoSimple)
        HomeOwn_embedded = self.HomeOwn_embedding(HomeOwn_input).reshape(-1, 24, self.embedding_dim_HomeOwn)
        RoomCount_embedded = self.RoomCount_embedding(RoomCount_input).reshape(-1, 24, self.embedding_dim_RoomCount)
        EcoSector_embedded = self.EcoSector_embedding(EcoSector_input).reshape(-1, 24, self.embedding_dim_EcoSector)
        JobType_embedded = self.JobType_embedding(JobType_input).reshape(-1, 24, self.embedding_dim_JobType)
        # Order columns
        OCCinHH_embedded = self.OCCinHH_embedding(OCCinHH_input).reshape(-1, 24, self.embedding_dim_OCCinHH)
        # non-temporal TUS daily features
        season_embedded = self.season_embedding(season_input).reshape(-1, 24, self.embedding_dim_season)
        weekend_embedded = self.weekend_embedding(weekend_input).reshape(-1, 24, self.embedding_dim_weekend)
        # Concatenate all features
        concatenated_features = torch.cat((education_embedded,employment_embedded,gender_embedded, famTypology_embedded, numFamMemb_embedded,
                                           ageClass_embedded, region_embedded,
                                           MartStat_embedded, 
                                           Kins_embedded, OccProf_embedded, FamTypo_embedded, OccSeqNum_embedded,
                                           Citizen_embedded, InterOwn_embedded, 
                                           MobPhoneOwn_embedded, CarOwn_embedded,
                                           FamTypoSimple_embedded, HomeOwn_embedded, RoomCount_embedded, EcoSector_embedded, JobType_embedded,
                                           OCCinHH_embedded, season_embedded, weekend_embedded, continuous_input), dim=2)

        concatenated_features = self.positional_encoding(concatenated_features)

        # Normalize concatenated features
        concatenated_features = self.layer_norm(concatenated_features)

        # Apply Transformer Encoder
        transformer_out = self.transformer_encoder(concatenated_features)

        # Layer Normalization
        transformer_out = self.layer_norm(transformer_out)

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, LOCATION, WITHNOBODY: The model has distinct output layers which allows for specialized processing for each target.
        # Activity Output with Activation
        activity_output = self.activity_dense(self.activation_act(transformer_out))

        # Location Output with Activation
        location_output = self.location_dropout(transformer_out)
        location_output = self.location_dense(self.activation_binary(location_output))

        # WithNOBODY Output with Activation
        withNOB_output = self.withNOB_dropout(transformer_out)
        withNOB_output = self.withNOB_dense(self.activation_binary(withNOB_output))

        return activity_output, location_output, withNOB_output

# TRAIN & EVALUATE: EXTRA ----------------------------------------------------------------------------------------------
def log_metrics_to_tensorboard(writer, epoch, metrics):
    """
    Logs metrics to TensorBoard.

    Parameters:
        writer (SummaryWriter): The TensorBoard summary writer.
        epoch (int): The current epoch number.
        metrics (dict): A dictionary containing metric names and their values.
    """
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(metric_name, metric_value, epoch)
# TRAIN & EVALUATE------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def create_dataloader(X, y_activity, y_location,y_withNOB, batch_size, shuffle=False, drop_last=False):
    # Replace these indices with the correct indices for according to exisiting data
    educationDegree_idx = 0
    employmentStatus_idx = 1
    gender_idx = 2
    famTypology_idx = 3
    numFamMembers_idx = 4
    ageClass_idx = 5
    region_idx = 6
    MartStat_idx = 7
    Kins_idx= 8
    OccProf_idx= 9
    FamTypo_idx= 10
    OccSeqNum_idx= 11
    Citizen_idx = 12
    InterOwn_idx=13
    MobPhoneOwn_idx=14
    CarOwn_idx=15
    FamTypoSimple_idx= 16
    HomeOwn_idx= 17
    RoomCount_idx= 18
    EcoSector_idx= 19
    JobType_idx= 20
    OCCinHH_idx = 21
    season_idx = 22
    weekend_idx = 23
    num_categorical_features = 24 # Update this to the total number of categorical features

    dataset = TensorDataset(
        torch.tensor(X[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X[:, :, numFamMembers_idx], dtype=torch.long),
        torch.tensor(X[:, :, ageClass_idx], dtype=torch.long),
        torch.tensor(X[:, :, region_idx], dtype=torch.long),
        torch.tensor(X[:, :, MartStat_idx], dtype=torch.long),
        torch.tensor(X[:, :, Kins_idx], dtype=torch.long),
        torch.tensor(X[:, :, OccProf_idx], dtype=torch.long),
        torch.tensor(X[:, :, FamTypo_idx], dtype=torch.long),
        torch.tensor(X[:, :, OccSeqNum_idx], dtype=torch.long),
        torch.tensor(X[:, :, Citizen_idx], dtype=torch.long),
        torch.tensor(X[:, :, InterOwn_idx], dtype=torch.long),
        torch.tensor(X[:, :, MobPhoneOwn_idx], dtype=torch.long),
        torch.tensor(X[:, :, CarOwn_idx], dtype=torch.long),
        torch.tensor(X[:, :, FamTypoSimple_idx], dtype=torch.long),
        torch.tensor(X[:, :, HomeOwn_idx], dtype=torch.long),
        torch.tensor(X[:, :, RoomCount_idx], dtype=torch.long),
        torch.tensor(X[:, :, EcoSector_idx], dtype=torch.long),
        torch.tensor(X[:, :, JobType_idx], dtype=torch.long),
        torch.tensor(X[:, :, OCCinHH_idx], dtype=torch.long),
        torch.tensor(X[:, :, season_idx], dtype=torch.long),
        torch.tensor(X[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X[:, :, num_categorical_features:], dtype=torch.float),  # Continuous data
        torch.tensor(y_activity, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location, dtype=torch.float),  # Location labels as floats (binary classification)
        torch.tensor(y_withNOB, dtype=torch.float),  # withNOBODY labels as floats (binary classification)
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_model(model, data_loader, cr_act, cr_loc, cr_NOB, opt, device, outDimAct, w_act, w_loc, w_NOB):
    model.train()

    train_loss = 0.0
    train_activity_loss = 0.0
    train_location_loss = 0.0
    train_withNOB_loss = 0.0

    correct_activity_predictions = 0
    total_activity_predictions = 0

    correct_location_predictions = 0
    total_location_predictions = 0

    correct_withNOB_predictions = 0
    total_withNOB_predictions = 0

    import csv
    log_file = 'embedding_logs.csv'
    # Open CSV file to log embedding details
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Batch Index', 'Parameter Name', 'Gradient Norm', 'Weight Norm'])

        for batch_idx, data in enumerate(data_loader):
            try:
                # Unpack all the data from the dataloader
                # !!! Adjust the number of unpacked variables based on how many there is
                education_data, employment_data, gender_data, famTypology_data, \
                numFamMemb_data, ageClass_data, region_data, \
                MartStatCat_data, KinsCat_data, OccProfCat_data, FamTypoCat_data, OccSeqNumCat_data, \
                Citizen_data, InterOwn_data, MobPhoneOwn_data, CarOwn_data, \
                FamTypoSimple_data, HomeOwn_data, RoomCount_data, EcoSector_data, JobType_data, \
                OCCinHH_data, season_data, \
                weekend_data, continuous_data, activity_target, location_target, withNOB_target  = [d.to(device) for d in data]

                opt.zero_grad()
                # Model's forward pass
                activity_output, location_output, withNOB_output = model(education_data, employment_data, gender_data, famTypology_data,
                                        numFamMemb_data, ageClass_data, region_data,
                                        MartStatCat_data, 
                                        KinsCat_data, OccProfCat_data, FamTypoCat_data, OccSeqNumCat_data,
                                        Citizen_data, InterOwn_data,
                                        MobPhoneOwn_data, CarOwn_data,
                                        FamTypoSimple_data, HomeOwn_data, RoomCount_data, EcoSector_data, JobType_data,
                                        OCCinHH_data, season_data,
                                        weekend_data, continuous_data, )

                # Compute the loss for both outputs
                loss_act = cr_act(activity_output.reshape(-1, outDimAct), activity_target.reshape(-1)) # criterion_activity
                loss_loc = cr_loc(location_output.reshape(-1), location_target.reshape(-1).float()) # criterion_location
                loss_NOB = cr_NOB(withNOB_output.reshape(-1), withNOB_target.reshape(-1).float()) # criterion_withNOBODY

                # Normalize the weights
                total_w = w_act + w_loc + w_NOB # calculation of total weight
                w_act /= total_w # w_act is weight_activity
                w_loc /= total_w # w_loc is weight_location
                w_NOB /= total_w # w_NOB is weight_withNOBODY

                total_loss = (w_act * loss_act + w_loc * loss_loc + w_NOB * loss_NOB)
                total_loss.backward()

                # Log embedding parameters after backward pass
                for name, param in model.named_parameters():
                    if 'embedding' in name:
                        gradient_norm = param.grad.norm().item() if param.grad is not None else None
                        weight_norm = param.norm().item()
                        writer.writerow([batch_idx, name, gradient_norm, weight_norm])

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25)  # to prevent vanishing gradient
                opt.step()

                train_loss += total_loss.item()
                train_activity_loss += loss_act.item()
                train_location_loss += loss_loc.item()
                train_withNOB_loss += loss_NOB.item()

                # Calculate activity accuracy
                _, predicted_activity_classes = torch.max(activity_output, dim=2)
                correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
                total_activity_predictions += activity_target.numel()

                # Calculate location accuracy
                predicted_location_classes = torch.sigmoid(location_output.reshape(-1)) > 0.5
                correct_location_predictions += (predicted_location_classes == location_target.reshape(-1).float()).float().sum()
                total_location_predictions += location_target.numel()

                # Calculate "withNOBODY" accuracy
                predicted_withNOB_classes = torch.sigmoid(withNOB_output.reshape(-1)) > 0.5
                correct_withNOB_predictions += (predicted_withNOB_classes == withNOB_target.reshape(-1).float()).float().sum()
                total_withNOB_predictions += withNOB_target.numel()

            except Exception as e:
                # Handle errors during the training loop
                print(f"An error occurred during training at batch {batch_idx}: {e}")
                continue  # Skip this batch and continue with the next

    # Avoid division by zero
    if total_activity_predictions == 0:
        raise ValueError("No predictions made for activity - check your data and model outputs.")

    # Check gradients
    #check_gradients(model)

    # Average loss and accuracy
    train_loss_avg = train_loss / len(data_loader)
    train_activity_loss = train_activity_loss / len(data_loader)
    train_location_loss = train_location_loss / len(data_loader)
    train_withNOB_loss = train_withNOB_loss / len(data_loader)

    train_activity_accuracy = correct_activity_predictions / total_activity_predictions
    train_location_accuracy = correct_location_predictions / total_location_predictions
    train_withNOB_accuracy = correct_withNOB_predictions / total_withNOB_predictions

    # Calculate total accuracy as a weighted average
    weight_total = w_act + w_loc + w_NOB
    train_total_accuracy = (
        w_act * train_activity_accuracy +
        w_loc * train_location_accuracy +
        w_NOB * train_withNOB_accuracy
    ) / weight_total

    return train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy, train_total_accuracy

def validate_model(model, data_loader, cr_act, cr_NOB, cr_loc,device, outDimAct, w_act, w_loc, w_NOB):
    model.eval()
    valid_loss = 0.0

    valid_activity_loss = 0.0
    valid_location_loss = 0.0
    valid_withNOB_loss = 0.0

    correct_activity_predictions = 0
    correct_location_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0

    correct_withNOB_predictions = 0
    total_withNOB_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            education_data, employment_data, gender_data, famTypology_data, \
                numFamMemb_data, ageClass_data, region_data, \
                MartStatCat_data, KinsCat_data, OccProfCat_data, FamTypoCat_data, OccSeqNumCat_data, \
                Citizen_data, InterOwn_data, MobPhoneOwn_data, CarOwn_data, \
                FamTypoSimple_data, HomeOwn_data, RoomCount_data, EcoSector_data, JobType_data, \
                OCCinHH_data, season_data, \
                weekend_data, continuous_data, activity_target, location_target, withNOB_target = [d.to(device) for d in data]

            activity_output, location_output, withNOB_output = model(education_data, employment_data, gender_data, famTypology_data,
                numFamMemb_data, ageClass_data, region_data,
                MartStatCat_data, KinsCat_data, OccProfCat_data, FamTypoCat_data, OccSeqNumCat_data,
                Citizen_data, InterOwn_data, MobPhoneOwn_data, CarOwn_data,
                FamTypoSimple_data, HomeOwn_data, RoomCount_data, EcoSector_data, JobType_data,
                OCCinHH_data, season_data,
                weekend_data, continuous_data,)

            loss_activity = cr_act(activity_output.reshape(-1, outDimAct), activity_target.reshape(-1))   # criterion_activity, output_dim_activity
            loss_location = cr_loc(location_output.reshape(-1), location_target.reshape(-1).float())      # criterion_location
            loss_withNOB = cr_NOB(withNOB_output.reshape(-1), withNOB_target.reshape(-1).float())         # criterion_withNOB

            # Normalize the weights
            total_w = w_act + w_loc + w_NOB # calculation of total weight
            w_act /= total_w                # w_act is weight_activity
            w_loc /= total_w                # w_loc is weight_location
            w_NOB /= total_w                # w_NOB is weight_withNOBODY

            weighted_loss = (w_act * loss_activity.item() + w_loc * loss_location.item() + w_NOB * loss_withNOB.item())
            valid_loss += weighted_loss

            valid_activity_loss += loss_activity.item()
            valid_location_loss += loss_location.item()
            valid_withNOB_loss += loss_withNOB.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.reshape(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.reshape(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

            # Calculate "withNOBODY" accuracy
            predicted_withNOB_classes = torch.sigmoid(withNOB_output.reshape(-1)) > 0.5
            correct_withNOB_predictions += (predicted_withNOB_classes == withNOB_target.reshape(-1).float()).float().sum()
            total_withNOB_predictions += withNOB_target.numel()

    valid_loss_avg = valid_loss / len(data_loader)
    valid_activity_loss = valid_activity_loss / len(data_loader)
    valid_location_loss = valid_location_loss / len(data_loader)
    valid_withNOB_loss = valid_withNOB_loss / len(data_loader)

    valid_activity_accuracy = correct_activity_predictions / total_activity_predictions
    valid_location_accuracy = correct_location_predictions / total_location_predictions
    valid_withNOB_accuracy = correct_withNOB_predictions / total_withNOB_predictions

    # Calculate total accuracy as a weighted average
    weight_total = w_act + w_loc + w_NOB
    valid_total_accuracy = (
        w_act * valid_activity_accuracy +
        w_loc * valid_location_accuracy +
        w_NOB * valid_withNOB_accuracy
    ) / weight_total

    return valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy, valid_total_accuracy


def train_and_evaluate_model_tuning(model,
                                 X_train, y_activity_train, y_location_train, y_withNOB_train,
                                 X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                 epochs, batch_size, learning_rate,
                                 device, w_act, w_loc, w_NOB,
                                 checkpoint_path='best_model_tuning.pth',
                                 verbose=False ,use_tensorboard=False, load_checkpoint=False,
                                 record_memory_usage=False, memory_log_path='memory_usage.txt',
                                 use_early_stopping=True,
                                 use_scheduler=False):
    import logging
    import psutil

    # Configure logging
    logging.basicConfig(filename='training_process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    outDimAct = len(set(y_activity_train.flatten())) #output_dim_activity

    writer = None
    if use_tensorboard:
        writer = SummaryWriter('runs/best_v3_model_tuning')

    # Create data loaders
    train_loader = create_dataloader(X_train, y_activity_train, y_location_train, y_withNOB_train, batch_size, shuffle=True,
                                     drop_last=False)
    valid_loader = create_dataloader(X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,  batch_size,  shuffle=False,
                                     drop_last=False)

    import torch
    import torch.nn.functional as F

    # Define loss functions for each task
    cr_act = torch.nn.CrossEntropyLoss(reduction='mean')  # 146
    cr_loc = torch.nn.BCEWithLogitsLoss(reduction='mean')  # Binary
    cr_NOB = torch.nn.BCEWithLogitsLoss(reduction='mean')  # Binary

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Optional learning rate scheduler
    if use_scheduler:
        import torch.optim.lr_scheduler as lr_scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)

    # Load checkpoint if it exists
    start_epoch = 0
    import os
    if load_checkpoint and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        best_valid_accuracy = checkpoint['best_valid_accuracy']
    else:
        history = {'train_loss': [], 'valid_loss': [],  'train_total_accuracy': [], 'valid_total_accuracy': [],
               'train_activity_loss': [], 'train_location_loss': [],'train_withNOB_loss': [],
               'valid_activity_loss': [], 'valid_location_loss': [], 'valid_withNOB_loss': [],
               'train_activity_accuracy': [],'train_location_accuracy': [], 'train_withNOB_accuracy': [],
               'valid_activity_accuracy': [], 'valid_location_accuracy': [],  'valid_withNOB_accuracy': []}
        best_valid_accuracy = 0.0

    if record_memory_usage:
        memory_log_file = open(memory_log_path, 'w')
        process = psutil.Process(os.getpid())
        memory_usages = []

    # Early stopping parameters
    patience = 50
    patience_counter = 0
    min_delta = 1e-4 # Minimum change to qualify as an improvement

    for epoch in range(start_epoch, epochs):
        if record_memory_usage:
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 ** 2)
            memory_usages.append(memory_usage_mb)
            memory_log_file.write(f'Epoch {epoch} - Memory Usage: {memory_usage_mb} MB\n')

        train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, \
            train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy,  train_total_accuracy  = train_model(model, train_loader,
            cr_act, cr_loc, cr_NOB, optimizer, device, outDimAct, w_act=w_act,w_loc=w_loc, w_NOB=w_NOB)

        valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, \
            valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy,  valid_total_accuracy = validate_model(
            model, valid_loader, cr_act, cr_loc, cr_NOB, device, outDimAct, w_act=w_act, w_loc=w_loc, w_NOB=w_NOB)

        scheduler.step(valid_loss_avg)
        history['train_loss'].append(train_loss_avg)
        history['train_activity_loss'].append(train_activity_loss)
        history['train_location_loss'].append(train_location_loss)
        history['train_withNOB_loss'].append(train_withNOB_loss)

        history['valid_loss'].append(valid_loss_avg)
        history['valid_activity_loss'].append(valid_activity_loss)
        history['valid_location_loss'].append(valid_location_loss)
        history['valid_withNOB_loss'].append(valid_withNOB_loss)

        history['train_total_accuracy'].append(train_total_accuracy.item())
        history['train_activity_accuracy'].append(train_activity_accuracy.item())
        history['train_location_accuracy'].append(train_location_accuracy.item())
        history['train_withNOB_accuracy'].append(train_withNOB_accuracy.item())

        history['valid_total_accuracy'].append(valid_total_accuracy.item())
        history['valid_activity_accuracy'].append(valid_activity_accuracy.item())
        history['valid_location_accuracy'].append(valid_location_accuracy.item())
        history['valid_withNOB_accuracy'].append(valid_withNOB_accuracy.item())

        # Save checkpoint after every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
            'best_valid_accuracy': best_valid_accuracy
        }
        torch.save(checkpoint, checkpoint_path)

        if use_tensorboard:
            # metrics for logging
            metrics_to_log = {
                'Loss/train': train_loss_avg,
                'Loss/train_activity': train_activity_loss,
                'Loss/train_location': train_location_loss,
                'Loss/train_withNOB': train_withNOB_loss,
                'Loss/valid': valid_loss_avg,
                'Loss/valid_activity': valid_activity_loss,
                'Loss/valid_location': valid_location_loss,
                'Loss/valid_withNOB': valid_withNOB_loss,
                'Accuracy/train': train_total_accuracy.item(),
                'Accuracy/train_activity': train_activity_accuracy.item(),
                'Accuracy/valid_activity': valid_activity_accuracy.item(),
                'Accuracy/train_location': train_location_accuracy.item(),
                'Accuracy/valid': valid_total_accuracy.item(),
                'Accuracy/valid_location': valid_location_accuracy.item(),
                'Accuracy/train_withNOB': train_withNOB_accuracy.item(),
                'Accuracy/valid_withNOB': valid_withNOB_accuracy.item(),
            }
            # Log metrics
            log_metrics_to_tensorboard(writer, epoch, metrics_to_log)

        # Log progress
        logging.info(f'Epoch {epoch}: '
                      f'Val_Act_Acc: {valid_activity_accuracy:.4f}, Val_Loc_Acc_: {valid_location_accuracy:.4f}, Val_withNOB_Acc: {valid_withNOB_accuracy:.4f}, '
                      f'Train_Act_Acc: {train_activity_accuracy:.4f}, Train_Loc_Acc: {train_location_accuracy:.4f}, Train_withNOB_Acc: {train_withNOB_accuracy:.4f}, '
                      f'Train Acc: {train_total_accuracy:.4f},  Valid Acc: {valid_total_accuracy:.4f},'
                      f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},'
                      f'Train Act Loss: {train_activity_loss:.4f}, Train Loc Loss: {train_location_loss:.4f}, Train withNOB Loss: {train_withNOB_loss:.4f},'
                      f'Val Act Loss: {valid_activity_loss:.4f}, Val Loc Loss: {valid_location_loss:.4f},  Val withNOB Loss: {valid_withNOB_loss:.4f},')

        # Conditionally print output
        if verbose:
            print(f'Epoch {epoch}: '
                  f'Val_Act_Acc: {valid_activity_accuracy:.4f}, Val_Loc_Acc_: {valid_location_accuracy:.4f}, Val_withNOB_Acc: {valid_withNOB_accuracy:.4f}, '
                  f'Train_Act_Acc: {train_activity_accuracy:.4f}, Train_Loc_Acc: {train_location_accuracy:.4f}, Train_withNOB_Acc: {train_withNOB_accuracy:.4f}, '
                  f'Train Acc: {train_total_accuracy:.4f},  Valid Acc: {valid_total_accuracy:.4f},'
                  f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},'
                  f'Train Act Loss: {train_activity_loss:.4f}, Train Loc Loss: {train_location_loss:.4f}, Train withNOB Loss: {train_withNOB_loss:.4f},'
                  f'Val Act Loss: {valid_activity_loss:.4f}, Val Loc Loss: {valid_location_loss:.4f},  Val withNOB Loss: {valid_withNOB_loss:.4f},')

        # Check if the model's validation accuracy for activity is improved
        if valid_activity_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_activity_accuracy
            # torch.save(model.state_dict(), checkpoint_path)
            torch.save(checkpoint, checkpoint_path)
            # print(f"Checkpoint saved to {checkpoint_path}")

        if use_early_stopping:
            # Early stopping based on validation loss stability
            if epoch > 0:
                if abs(history['valid_loss'][-1] - history['valid_loss'][-2]) < min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch} due to no improvement in validation loss for {patience} epochs.')
                    break

        check_and_log_gradients(model, folder_path='gradients_logs', epoch= epoch + 1)

        # Additional early stopping condition: Accuracy threshold
        accuracy_threshold = 0.96 #(0.01 tolerance for training data accuracy)
        if best_valid_accuracy >= accuracy_threshold:
            print(f'Early stopping triggered at epoch {epoch} due to reaching accuracy threshold of {accuracy_threshold:.2f}')
            break

    plot_gradients('gradients_logs')

    if use_tensorboard:
        # Close the TensorBoard writer
        writer.close()

    if record_memory_usage:
        average_memory_usage = sum(memory_usages) / len(memory_usages)
        memory_log_file.write(f'Average Memory Usage: {average_memory_usage} MB\n')
        print("memory_usage is recorded in the .txt file")
        memory_log_file.close()
    return history, model
# LSTM:TUNING CROSS-VALIDATION ---------------------------------------------------------------------------
def objectiveTransformer_kFold(trial, epochs, df, num_split):
    from sklearn.preprocessing import LabelEncoder
    import torch
    import matplotlib.pyplot as plt
    import os

    # Create folders for plots
    trial_plots_ACC_folder, trial_plots_LOSS_folder = create_plot_folders()

    # Embedding dimensions and other parameters
    num_features = {
        'num_educationCat': df['Education Degree'].nunique(),
        'num_employmentCat': df['Employment status'].nunique(),
        'num_genderCat': df['Gender'].nunique(),
        'num_famTypologyCat': df['Family Typology'].nunique(),
        'num_numFamMembCat': df['Number Family Members'].nunique(),
        'num_ageClassCat': df['Age Classes'].nunique(),
        'num_regionCat': df['Region'].nunique(),
        'num_MartStatCat': df["Marital Status"].nunique(),
        'num_KinsCat': df["Kinship Relationship"].nunique(),
        'num_OccProfCat': df["Nuclear Family, Occupant Profile"].nunique(),
        'num_FamTypoCat': df["Nuclear Family, Typology"].nunique(),
        'num_OccSeqNumCat': df["Nuclear Family, Occupant Sequence Number"].nunique(),
        'num_CitizenCat': df['Citizenship'].nunique(),  # new additions
        'num_InterOwnCat': df['Internet Access'].nunique(),
        'num_MobPhoneOwnCat': df['Mobile Phone Ownership'].nunique(),
        'num_CarOwnCat': df['Car Ownership'].nunique(),
        'num_OCCinHHCat': df['Occupant_ID_in_HH'].nunique(),
        'num_FamTypoSimpleCat': df['Family_Typology_Simple'].nunique(),
        'num_HomeOwnCat': df['Home Ownership'].nunique(),
        'num_RoomCountCat': df['Room Count'].nunique(),
        'num_EcoSectorCat': df['Economic Sector, Profession'].nunique(),
        'num_JobTypeCat': df['Job Type'].nunique(),
        'num_seasonCat': 4,
        'num_unique_weekCat': 3,
        'num_continuous_features': 4
    }

    # CONSTANT VALUES
    #mps_device = torch.device("mps")
    mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout_embedding = 0
    dropout_loc = 0.25
    dropout_withNOB = 0.25
    dropout_TFTs = 0
    weight_activity = 1
    weight_location = 1
    weight_withNOB = 1
    embed_size = 50
    nhead = 4
    num_hidden_layers = 2

    # MODEL HYPERPARAMETERS
    batch_size = trial.suggest_categorical('batch_size', [96, 128, 256])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 8e-4, 6e-4, 4e-4, 2e-4, 9e-5])
    d_feed = trial.suggest_categorical('d_feed', [7168, 8192, 9216, 10240])

    # TUNING HYPERPARAMETERS
    folds, df_preprocessed = data_preprocess_k_fold_split(df, n_splits=num_split)
    total_val_loss_activity = 0.0
    total_val_loss_location = 0.0
    total_activity_accuracy = 0.0
    total_location_accuracy = 0.0
    total_val_loss_withNOB = 0.0
    total_withNOB_accuracy = 0.0

    # PLOTTING: Track the history of training for plotting
    training_loss_history = []
    validation_loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    for fold_idx, (train_data_ids, val_data_ids) in enumerate(folds):
        train_data = df_preprocessed.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
        val_data = df_preprocessed.merge(val_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])

        for dataset in [train_data, val_data]:
            dataset.drop(columns=['Household_ID'], inplace=True)

        unique_activities = df_preprocessed['Occupant_Activity'].unique().reshape(-1, 1)
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_activities)

        y_activity_train = label_encoder.transform(train_data[['Occupant_Activity']].values.ravel())
        y_activity_valid = label_encoder.transform(val_data[['Occupant_Activity']].values.ravel())
        y_location_train = train_data[['location']].values.ravel()
        y_location_valid = val_data[['location']].values.ravel()
        y_withNOB_train = train_data[['withNOBODY']].values.ravel()
        y_withNOB_valid = val_data[['withNOBODY']].values.ravel()

        train_data = train_data.drop(columns=['Occupant_Activity', 'location', "withNOBODY"])
        val_data = val_data.drop(columns=['Occupant_Activity', 'location', "withNOBODY"])

        X_train = train_data.values.reshape(-1, 24, train_data.shape[1])
        X_valid = val_data.values.reshape(-1, 24, val_data.shape[1])

        y_activity_train = y_activity_train.reshape(-1, 24)
        y_activity_valid = y_activity_valid.reshape(-1, 24)
        y_location_train = y_location_train.reshape(-1, 24)
        y_location_valid = y_location_valid.reshape(-1, 24)
        y_withNOB_train = y_withNOB_train.reshape(-1, 24)
        y_withNOB_valid = y_withNOB_valid.reshape(-1, 24)

        output_dim_activity = len(set(y_activity_train.flatten()))
        output_dim_location = 1
        output_dim_withNOB = 1

        model = TransformerModelTuning(
            **num_features,
            output_dim_activity=output_dim_activity,
            output_dim_location=output_dim_location,
            output_dim_withNOB=output_dim_withNOB,
            num_hidden_layers=num_hidden_layers,
            dropout_loc=dropout_loc,
            dropout_withNOB=dropout_withNOB,
            dropout_embedding=dropout_embedding,
            dropout_transformer=dropout_TFTs,
            embed_size=embed_size,
            nhead=nhead,
            d_feed = d_feed,
        ).to(mps_device)

        history, trained_model = train_and_evaluate_model_tuning(
            model, X_train, y_activity_train, y_location_train, y_withNOB_train,
            X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
            epochs, batch_size, learning_rate, device=mps_device,
            w_act=weight_activity, w_loc=weight_location, w_NOB=weight_withNOB,
            checkpoint_path=f'best_modelLSTM_fold{fold_idx}.pth',
            verbose=False, use_tensorboard=False, load_checkpoint=False,
            use_scheduler=True
        )

        # Clear the CUDA cache after each fold to free up memory
        torch.cuda.empty_cache()

        # PLOTTING: Collect history data for plotting
        training_loss_history.extend(history['train_loss'])
        validation_loss_history.extend(history['valid_loss'])
        training_accuracy_history.extend(history['train_total_accuracy'])
        validation_accuracy_history.extend(history['valid_total_accuracy'])

        total_val_loss_activity += history['valid_activity_loss'][-1]
        total_val_loss_location += history['valid_location_loss'][-1]
        total_val_loss_withNOB += history['valid_withNOB_loss'][-1]

        total_activity_accuracy += history['valid_activity_accuracy'][-1]
        total_location_accuracy += history['valid_location_accuracy'][-1]
        total_withNOB_accuracy += history['valid_withNOB_accuracy'][-1]

        # After training the model
        last_epoch = len(
            history['valid_activity_loss'])  # Assuming the length of the loss list corresponds to the number of epochs

    hyperparameters = {
        'num_hidden_layers': num_hidden_layers,
        'nhead': nhead,
        'embed_size': embed_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'd_feed': d_feed,
    }

    # Call the function to record parameters and hyperparameters
    record_trial_parameters(trial=trial, model=model, hyperparameters=hyperparameters)

    avg_val_loss_activity = total_val_loss_activity / len(folds)
    avg_val_loss_location = total_val_loss_location / len(folds)
    avg_val_loss_withNOB = total_val_loss_withNOB / len(folds)

    avg_activity_accuracy = total_activity_accuracy / len(folds)
    avg_location_accuracy = total_location_accuracy / len(folds)
    avg_withNOB_accuracy = total_withNOB_accuracy / len(folds)

    # Call plotting functions
    plot_loss_history(training_loss_history, validation_loss_history, num_split, epochs, trial.number, trial_plots_LOSS_folder)
    plot_accuracy_history(training_accuracy_history, validation_accuracy_history, num_split, epochs, trial.number, trial_plots_ACC_folder)

    total_weight = weight_activity + weight_location + weight_withNOB
    weight_activity /= total_weight
    weight_location /= total_weight
    weight_withNOB /= total_weight

    weighted_loss = (
        weight_activity * avg_val_loss_activity +
        weight_location * avg_val_loss_location +
        weight_withNOB * avg_val_loss_withNOB
    )

    weighted_accuracy = (
        weight_activity * avg_activity_accuracy +
        weight_location * avg_location_accuracy +
        weight_withNOB * avg_withNOB_accuracy
    )

    return weighted_loss, weighted_accuracy, avg_activity_accuracy, avg_location_accuracy, avg_withNOB_accuracy, last_epoch
def main_tuningTransformer_kFold(csv_filepath, n_trials, epochs, num_split, df):
    import os
    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]

    # Create the Optuna study
    import optuna
    import json
    import torch
    # Define a wrapper function to pass the epochs parameter
    def objective_wrapper(trial):
        weighted_loss, weighted_accuracy, avg_activity_accuracy, avg_location_accuracy, avg_withNOB_accuracy, last_epoch = objectiveTransformer_kFold(trial, epochs=epochs, df=df, num_split=num_split)

        # Optionally store accuracy values in the trial's user attributes
        trial.set_user_attr('total_accuracy', weighted_accuracy)
        trial.set_user_attr('activity_accuracy', avg_activity_accuracy)
        trial.set_user_attr('location_accuracy', avg_location_accuracy)
        trial.set_user_attr('withNOB_accuracy', avg_withNOB_accuracy)
        trial.set_user_attr('last_epoch', last_epoch)

        # Clear the CUDA cache after each trial to free up memory
        torch.cuda.empty_cache()
        return weighted_loss

    # Define a callback function to save study progress after each trial
    def save_study_progress(study, trial):
        # Save the best trial's hyperparameters to a JSON file
        best_trial_params = study.best_trial.params
        with open(f"{filename_prefix}_best_TuningParams_Trans.json", 'w') as outfile:
            json.dump(best_trial_params, outfile)

        # Save trial details after each trial
        save_trial_details(study, "TResults")
        vis_PCP(study, target_name='Target (e.g., weighted_loss)')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[save_study_progress])

    # Save best trial's hyperparameters to a JSON file
    best_trial_params = study.best_trial.params
    with open(f"{filename_prefix}_best_TuningParams_Trans.json", 'w') as outfile:
        json.dump(best_trial_params, outfile)

    vis_Scatter_HyperCombin(study, 'ScatterPlots')
    vis_Hexbin_HyperCombin(study, 'HexbinPlots')
    vis_KDE_HyperCombin(study, 'KDEPlots')
    vis_Bubble_HyperCombin(study, 'bubblePlots')

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    print('Value:', study.best_trial.value)
    print('Params:', best_trial_params)
def trainEvaluate_afterTuning(df, csv_filepath, epochBase=500,):
    import torch
    import json
    import os

    X_train, y_activity_train, y_location_train, y_withNOB_train, \
    X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
    X_test, y_activity_test, y_location_test, y_withNOB_test, \
    label_encoder = data_preprocess(df)

    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    print("filename_prefix:", filename_prefix)

    # Load best hyperparameters
    with open(f"{filename_prefix}_best_TuningParams_Trans.json", 'r') as infile:
        best_params = json.load(infile)
    print(best_params)

    # input shape: (season_input, weekend_input, continuous_input), ((24,), (24,), (24, num_continuous_features))
    #mps_device = torch.device("mps")
    mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Embedding dimensions and other parameters
    num_features = {
        'num_educationCat': df['Education Degree'].nunique(),
        'num_employmentCat': df['Employment status'].nunique(),
        'num_genderCat': df['Gender'].nunique(),
        'num_famTypologyCat': df['Family Typology'].nunique(),
        'num_numFamMembCat': df['Number Family Members'].nunique(),
        'num_ageClassCat': df['Age Classes'].nunique(),
        'num_regionCat': df['Region'].nunique(),
        'num_MartStatCat': df["Marital Status"].nunique(),
        'num_KinsCat': df["Kinship Relationship"].nunique(),
        'num_OccProfCat': df["Nuclear Family, Occupant Profile"].nunique(),
        'num_FamTypoCat': df["Nuclear Family, Typology"].nunique(),
        'num_OccSeqNumCat': df["Nuclear Family, Occupant Sequence Number"].nunique(),
        'num_CitizenCat': df['Citizenship'].nunique(),  # new additions
        'num_InterOwnCat': df['Internet Access'].nunique(),
        'num_MobPhoneOwnCat': df['Mobile Phone Ownership'].nunique(),
        'num_CarOwnCat': df['Car Ownership'].nunique(),
        'num_FamTypoSimpleCat': df['Family_Typology_Simple'].nunique(),
        'num_HomeOwnCat': df['Home Ownership'].nunique(),
        'num_RoomCountCat': df['Room Count'].nunique(),
        'num_EcoSectorCat': df['Economic Sector, Profession'].nunique(),
        'num_JobTypeCat': df['Job Type'].nunique(),
        'num_OCCinHHCat': df['Occupant_ID_in_HH'].nunique(),
        'num_seasonCat': 4,
        'num_unique_weekCat': 3,
        'num_continuous_features': 4
    }


    print(
        num_features['num_educationCat'], num_features['num_employmentCat'], num_features['num_genderCat'],
        num_features['num_famTypologyCat'], num_features['num_numFamMembCat'], num_features['num_ageClassCat'],
        num_features['num_OCCinHHCat'], num_features['num_regionCat'], 
        num_features['num_MartStatCat'],
        num_features['num_KinsCat'], num_features['num_OccProfCat'],
        num_features['num_FamTypoCat'], num_features['num_OccSeqNumCat'], num_features['num_seasonCat'],
        num_features['num_unique_weekCat'], num_features['num_continuous_features'],
        num_features['num_CitizenCat'],
        num_features['num_InterOwnCat'], num_features['num_MobPhoneOwnCat'],
        num_features['num_CarOwnCat'],
        num_features['num_FamTypoSimpleCat'], num_features['num_HomeOwnCat'], num_features['num_RoomCountCat'],
        num_features['num_EcoSectorCat'], num_features['num_JobTypeCat']
    )

    dropout_embedding = 0
    dropout_loc = 0.25
    dropout_withNOB = 0.1
    dropout_TFTs = 0
    w_act = 1
    w_loc = 1
    w_NOB = 1
    embed_size = 50
    nhead = 4
    num_hidden_layers = 3

    # Unpack hyperparameters
    #num_hidden_layers = best_params['num_hidden_layers']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    #nhead = best_params['nhead']
    d_feed = best_params['d_feed']

    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary
    print("output_dim_activity:", output_dim_activity)

    model = TransformerModelTuning(
        **num_features,
        output_dim_activity=output_dim_activity,
        output_dim_location=output_dim_location,
        output_dim_withNOB=output_dim_withNOB,
        num_hidden_layers=num_hidden_layers,
        dropout_loc=dropout_loc,
        dropout_withNOB=dropout_withNOB,
        dropout_embedding=dropout_embedding,
        dropout_transformer=dropout_TFTs,
        embed_size=embed_size,
        nhead=nhead,
        d_feed=d_feed,
    ).to(mps_device)


    #print(model)

    # Train the model
    history, trained_model =  train_and_evaluate_model_tuning(model,
                                                             X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                             X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                             epochBase, batch_size, learning_rate,
                                                             device=mps_device, w_act=w_act, w_loc=w_loc,  w_NOB=w_NOB,
                                                             checkpoint_path=f"{filename_prefix}_best_modelTransformer.pth",
                                                             verbose=True ,use_tensorboard=True, load_checkpoint=False,
                                                             record_memory_usage=True, memory_log_path='memory_usage.txt',
                                                             use_early_stopping=False,
                                                             use_scheduler=True,)


    # After each fold, clear CUDA cache to free memory
    torch.cuda.empty_cache()

    # Optionally: Evaluate the trained model further and save
    # Example: Save the model if it's better than previous models
    torch.save(trained_model.state_dict(), f"{filename_prefix}_best_modelTransformer.pth")
    print("training_afterTuning()_Transformer is completed")

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

    # tensorboard --logdir=runs/End_to_End_full_embedding # from terminal

    model.load_state_dict(torch.load(f"{filename_prefix}_best_modelTransformer.pth"))
    evaluate_and_save_afterTuning(model, X_test, y_activity_test,y_location_test,y_withNOB_test,  label_encoder=label_encoder, device=mps_device, csv_filepath= csv_filepath, arch_type="Transformer")

    # EVALUATION for model performance for each activity categories ----------------------------------------------------
    # Assume `model`, `X_test`, `y_test`, and `label_encoder` are already defined and the model is trained
    evalACT_classify_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder=label_encoder, device=mps_device, save_csv=True, csv_filepath= csv_filepath,arch_type="Transformer")

# AFTER TUNING EVALUATE AND SAVE ---------------------------------------------------------------------------------------
def evaluate_and_save_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder, device, csv_filepath, arch_type):
    import os
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the new filename with the model predictions suffix
    file_name = f"{filename_prefix}_model_predictions_{arch_type}.csv"

    # Replace these indices with the correct indices for according to exisiting data
    educationDegree_idx = 0
    employmentStatus_idx = 1
    gender_idx = 2
    famTypology_idx = 3
    numFamMembers_idx = 4
    ageClass_idx = 5
    region_idx = 6
    MartStat_idx = 7
    Kins_idx= 8
    OccProf_idx= 9
    FamTypo_idx= 10
    OccSeqNum_idx= 11
    Citizen_idx = 12
    InterOwn_idx=13
    MobPhoneOwn_idx=14
    CarOwn_idx=15
    FamTypoSimple_idx= 16
    HomeOwn_idx= 17
    RoomCount_idx= 18
    EcoSector_idx= 19
    JobType_idx= 20
    OCCinHH_idx = 21
    season_idx = 22
    weekend_idx = 23
    num_categorical_features = 24 # Update this to the total number of categorical features

    test_dataset = TensorDataset(
        torch.tensor(X_test[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, numFamMembers_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, ageClass_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, region_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, MartStat_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, Kins_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OccProf_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, FamTypo_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OccSeqNum_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, Citizen_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, InterOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, MobPhoneOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, CarOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, FamTypoSimple_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, HomeOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, RoomCount_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, EcoSector_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, JobType_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OCCinHH_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, season_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, num_categorical_features:], dtype=torch.float),  # Continuous data
        torch.tensor(y_activity_test, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location_test, dtype=torch.float),  # Location labels as floats (binary classification)
        torch.tensor(y_withNOB_test, dtype=torch.float),  # withNOBODY labels as floats (binary classification)
    )

    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

    all_activity_preds = []
    all_location_preds = []
    all_withNOB_preds = []

    all_activity_actuals = []
    all_location_actuals = []
    all_withNOB_actuals = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # Unpack the data
            *features, activity_target, location_target, withNOB_target = [d.to(device) for d in data]

            # Make predictions
            activity_output, location_output, withNOB_output  = model(*features)
            _, predicted_activity = torch.max(activity_output, 2)
            predicted_location = torch.round(torch.sigmoid(location_output)).int()
            predicted_withNOB = torch.round(torch.sigmoid(withNOB_output)).int()

            # Append to lists
            all_activity_preds.extend(predicted_activity.reshape(-1).cpu().numpy())
            all_location_preds.extend(predicted_location.reshape(-1).cpu().numpy())
            all_withNOB_preds.extend(predicted_withNOB.reshape(-1).cpu().numpy())

            all_activity_actuals.extend(activity_target.reshape(-1).cpu().numpy())
            all_location_actuals.extend(location_target.reshape(-1).cpu().numpy())
            all_withNOB_actuals.extend(withNOB_target.reshape(-1).cpu().numpy())

    # Inverse transform to get categories from label encoded activity predictions
    all_actuals_categories = label_encoder.inverse_transform(all_activity_actuals)
    all_activity_preds_categories = label_encoder.inverse_transform(all_activity_preds)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Actual Activity': all_actuals_categories,
        'Predicted Activity Category': all_activity_preds_categories,
        'Actual Location': all_location_actuals,
        'Predicted Location': all_location_preds,
        'Actual WithNobody': all_withNOB_actuals,
        'Predicted WithNobody': all_withNOB_preds,
    })

    results_df.to_csv(file_name, index=False)
    print(f'Results saved to {file_name}')
    print(f"evaluate_and_save_afterTuning_{arch_type} is completed")
    return results_df  # Optional: return the results dataframe
# EVALUATION for model performance for each activity categories ----------------------------------------------------
def evalACT_classify_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder, device, csv_filepath, arch_type, save_csv=False, ):
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the new filename with the model predictions suffix
    file_name = f"{filename_prefix}_classificationReport_{arch_type}.csv"

    # Replace these indices with the correct indices for according to exisiting data
    educationDegree_idx = 0
    employmentStatus_idx = 1
    gender_idx = 2
    famTypology_idx = 3
    numFamMembers_idx = 4
    ageClass_idx = 5
    region_idx = 6
    MartStat_idx = 7
    Kins_idx= 8
    OccProf_idx= 9
    FamTypo_idx= 10
    OccSeqNum_idx= 11
    Citizen_idx = 12
    InterOwn_idx=13
    MobPhoneOwn_idx=14
    CarOwn_idx=15
    FamTypoSimple_idx= 16
    HomeOwn_idx= 17
    RoomCount_idx= 18
    EcoSector_idx= 19
    JobType_idx= 20
    OCCinHH_idx = 21
    season_idx = 22
    weekend_idx = 23
    num_categorical_features = 24 # Update this to the total number of categorical features

    test_dataset = TensorDataset(
        torch.tensor(X_test[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, numFamMembers_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, ageClass_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, region_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, MartStat_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, Kins_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OccProf_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, FamTypo_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OccSeqNum_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, Citizen_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, InterOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, MobPhoneOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, CarOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, FamTypoSimple_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, HomeOwn_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, RoomCount_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, EcoSector_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, JobType_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OCCinHH_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, season_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, num_categorical_features:], dtype=torch.float),  # Continuous data
        torch.tensor(y_activity_test, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location_test, dtype=torch.float),  # Location labels as floats (binary classification)
        torch.tensor(y_withNOB_test, dtype=torch.float),  # withNOBODY labels as floats (binary classification)
    )

    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

    all_activity_preds = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # Unpack the data
            *features, activity_target, location_target, withNOB_target = [d.to(device) for d in data]
            # Make predictions
            activity_output, location_output, withNOB_output = model(*features)
            _, predicted_activity = torch.max(activity_output, 2)
            all_activity_preds.extend(predicted_activity.reshape(-1).cpu().numpy())

    # Flatten y_activity_test if it's not already a 1D array
    y_activity_test_flat = y_activity_test.flatten() if y_activity_test.ndim > 1 else y_activity_test

    # Inverse transform to get categories
    all_activity_preds_labels = label_encoder.inverse_transform(all_activity_preds)
    all_activity_actuals_labels = label_encoder.inverse_transform(y_activity_test_flat)

    # Calculate classification report and confusion matrix
    class_report = classification_report(all_activity_actuals_labels, all_activity_preds_labels, zero_division=0,
                                         output_dict=True)
    conf_matrix = confusion_matrix(all_activity_actuals_labels, all_activity_preds_labels, labels=label_encoder.classes_)

    # Convert classification report to DataFrame
    class_report_df = pd.DataFrame.from_dict(class_report)
    class_report_df = class_report_df.transpose()


    # Save to CSV if required
    if save_csv:
        # Save classification report
        class_report_df.to_csv(file_name)
        print(f"Classification report saved to {file_name}")

        # Save confusion matrix
        conf_matrix_filename = filename_prefix + f'confusion_matrix_{arch_type}.csv'
        pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
            conf_matrix_filename)
        print(f"Confusion matrix saved to {conf_matrix_filename}")

    return class_report, conf_matrix
#VISUALIZE - AFTER TUNING ----------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import torch
def plot_history(history, filename='training_history.png'):
    plt.figure(figsize=(18, 12))

    # Activity Accuracy
    train_activity_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_activity_accuracy']]
    valid_activity_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_activity_accuracy']]

    # Location Accuracy
    train_location_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_location_accuracy']]
    valid_location_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_location_accuracy']]

    # withNOBODY Accuracy
    train_withNOB_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_withNOB_accuracy']]
    valid_withNOB_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_withNOB_accuracy']]

    # Activity Loss
    train_activity_loss = history['train_activity_loss']
    valid_activity_loss = history['valid_activity_loss']

    # Location Loss
    train_location_loss = history['train_location_loss']
    valid_location_loss = history['valid_location_loss']

    # WithNOBODY Loss
    train_withNOB_loss = history['train_withNOB_loss']
    valid_withNOB_loss = history['valid_withNOB_loss']

    # Plot Activity Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(train_activity_accuracy, label='Train Activity Accuracy')
    plt.plot(valid_activity_accuracy, label='Validation Activity Accuracy')
    plt.title('Activity (Categorical) Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel(' ')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot Location Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(train_location_accuracy, label='Train Location Accuracy')
    plt.plot(valid_location_accuracy, label='Validation Location Accuracy')
    plt.title('Location (Binary) Accuracy')
    plt.ylabel(' ')
    plt.xlabel(' ')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot WithNOBODY Accuracy
    plt.subplot(2, 3, 3)
    plt.plot(train_withNOB_accuracy, label='Train WithNOBODY Accuracy')
    plt.plot(valid_withNOB_accuracy, label='Validation WithNOBODY Accuracy')
    plt.title('WithNOBODY (Binary) Accuracy, Binary')
    plt.ylabel(' ')
    plt.xlabel(' ')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot Activity Loss
    plt.subplot(2, 3, 4)
    plt.plot(train_activity_loss, label='Train Activity Loss')
    plt.plot(valid_activity_loss, label='Validation Activity Loss')
    plt.title('Activity (Categorical) Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot Location Loss
    plt.subplot(2, 3, 5)
    plt.plot(train_location_loss, label='Train Location Loss')
    plt.plot(valid_location_loss, label='Validation Location Loss')
    plt.title('Location (Binary) Loss')
    plt.ylabel(' ')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot WithNOBODY Loss
    plt.subplot(2, 3, 6)
    plt.plot(train_withNOB_loss, label='Train WithNOBODY Loss')
    plt.plot(valid_withNOB_loss, label='Validation WithNOBODY Loss')
    plt.title('WithNOBODY (Binary) Loss')
    plt.ylabel(' ')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(filename)

    plt.tight_layout()
    plt.show()

#VISUALIZE FOR TUNING FUNCTION - DURING TUNING -------------------------------------------------------------------------
import os
def create_plot_folders():
    """Create folders to store the plots if they don't exist."""
    trial_plots_ACC_folder = "Tuning_ACCURACY_Trial_History_Plots"
    trial_plots_LOSS_folder = "Tuning_LOSS_Trial_History_Plots"

    if not os.path.exists(trial_plots_ACC_folder):
        os.makedirs(trial_plots_ACC_folder)

    if not os.path.exists(trial_plots_LOSS_folder):
        os.makedirs(trial_plots_LOSS_folder)

    return trial_plots_ACC_folder, trial_plots_LOSS_folder

def plot_loss_history(training_loss_history, validation_loss_history, num_folds, epochs, trial_number, trial_plots_LOSS_folder):
    """Plot and save the training and validation loss for each fold."""
    fig, axs = plt.subplots(1, num_folds, figsize=(15, 5), sharey=True)

    # Ensure axs is always treated as an array, even if there's only one fold
    if num_folds == 1:
        axs = [axs]

    for fold_idx in range(num_folds):
        # Extract relevant data for the current fold
        start_epoch = fold_idx * epochs
        end_epoch = start_epoch + epochs  # Assume epochs is the maximum length

        # Extract loss data for the current fold
        fold_training_loss = training_loss_history[start_epoch:end_epoch]
        fold_validation_loss = validation_loss_history[start_epoch:end_epoch]

        # Set x-axis values for actual epochs within each fold
        actual_epochs = range(1, len(fold_training_loss) + 1)

        # Plot the training and validation loss for the current fold
        if len(actual_epochs) > 0:
            axs[fold_idx].plot(
                actual_epochs,
                fold_training_loss,
                label="Training Loss"
            )
            axs[fold_idx].plot(
                actual_epochs,
                fold_validation_loss,
                label="Validation Loss"
            )

            axs[fold_idx].set_title(f"Fold {fold_idx + 1}")
            axs[fold_idx].set_xlabel("Epochs")
            axs[fold_idx].set_ylim(0, 0.1)  # Set y-axis limit for loss range
            axs[fold_idx].set_xlim(1, epochs)  # Ensure x-axis is set to the correct epoch range
            if fold_idx == 0:
                axs[fold_idx].set_ylabel("Loss")
            axs[fold_idx].legend()

    fig.suptitle(f"Training and Validation Loss per Fold (Trial {trial_number})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{trial_plots_LOSS_folder}/loss_trial_{trial_number}_subplots.png")
    plt.close()

def plot_accuracy_history(training_accuracy_history, validation_accuracy_history, num_folds, epochs, trial_number,
                          trial_plots_ACC_folder):
    """Plot and save the training and validation accuracy for each fold."""
    fig, axs = plt.subplots(1, num_folds, figsize=(15, 5), sharey=True)

    # Ensure axs is always treated as an array, even if there's only one fold
    if num_folds == 1:
        axs = [axs]

    for fold_idx in range(num_folds):
        # Extract relevant data for the current fold
        start_epoch = fold_idx * epochs
        end_epoch = start_epoch + epochs  # Assume epochs is the maximum length

        # Extract accuracy data for the current fold
        fold_training_accuracy = training_accuracy_history[start_epoch:end_epoch]
        fold_validation_accuracy = validation_accuracy_history[start_epoch:end_epoch]

        # Set x-axis values for actual epochs within each fold
        actual_epochs = range(1, len(fold_training_accuracy) + 1)

        # Plot the training and validation accuracy for the current fold
        if len(actual_epochs) > 0:
            axs[fold_idx].plot(
                actual_epochs,
                fold_training_accuracy,
                label="Training Accuracy"
            )
            axs[fold_idx].plot(
                actual_epochs,
                fold_validation_accuracy,
                label="Validation Accuracy"
            )

            axs[fold_idx].set_title(f"Fold {fold_idx + 1}")
            axs[fold_idx].set_xlabel("Epochs")
            axs[fold_idx].set_ylim(0, 1)  # Set y-axis limit to range from 0 to 1 for consistent accuracy scaling
            axs[fold_idx].set_xlim(1, epochs)  # Ensure x-axis is set to the correct epoch range
            if fold_idx == 0:
                axs[fold_idx].set_ylabel("Accuracy")
            axs[fold_idx].legend()

    fig.suptitle(f"Training and Validation Accuracy per Fold (Trial {trial_number})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{trial_plots_ACC_folder}/accuracy_trial_{trial_number}_subplots.png")
    plt.close()

#VISUALIZE FOR DETAIL - DURING TUNING ----------------------------------------------------------------------------------
def plot_hyperparameters_vs_target(num_hidden_layers_list, batch_size_list, learning_rate_list, nhead_list, d_feed_list, target_values_list):
    import matplotlib.pyplot as plt

    # Plot for num_hidden_layers
    plt.figure()
    plt.scatter(num_hidden_layers_list, target_values_list)
    plt.xlabel('num_hidden_layers')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('num_hidden_layers vs Target')
    plt.show()

    # Plot for batch_size
    plt.figure()
    plt.scatter(batch_size_list, target_values_list)
    plt.xlabel('batch_size')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('batch_size vs Target')
    plt.show()

    # Plot for learning_rate
    plt.figure()
    plt.scatter(learning_rate_list, target_values_list)
    plt.xlabel('learning_rate')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('learning_rate vs Target')
    plt.show()

    # Plot for nhead
    plt.figure()
    plt.scatter(nhead_list, target_values_list)
    plt.xlabel('nhead')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('nhead vs Target')
    plt.show()

    # Plot for d_feed
    plt.figure()
    plt.scatter(d_feed_list, target_values_list)
    plt.xlabel('d_feed')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('d_feed vs Target')
    plt.show()

def vis_PCP(study, target_name='Target (e.g., weighted_loss)', output_html='PCP_advanced.html'):
    import pandas as pd
    import plotly.express as px
    # Convert study trials to DataFrame
    all_trials_data = []
    for trial in study.trials:
        trial_data = trial.params.copy()
        trial_data['target'] = trial.value
        trial_data['iter'] = trial.number
        all_trials_data.append(trial_data)

    df_trials = pd.DataFrame(all_trials_data)

    # Get all possible hyperparameters used in the trials
    hyperparams = [col for col in df_trials.columns if col not in ['target', 'iter']]

    # Create a parallel coordinate plot with interactive filtering
    fig = px.parallel_coordinates(
        df_trials,
        dimensions=hyperparams + ['target'],  # Include target in the dimensions
        color='target',  # Color by target value
        labels={param: param for param in hyperparams},  # Assign labels to hyperparameters
        color_continuous_scale=px.colors.sequential.Viridis,  # Customize color scale
        range_color=[df_trials['target'].min(), df_trials['target'].max()],  # Set color range based on target
    )

    # Update layout for better interaction
    fig.update_layout(
        title='',
        coloraxis_colorbar=dict(
            title=target_name
        ),
        font=dict(family="Arial", size=14),
        hovermode='closest',  # Set hovermode to 'closest' for better interactivity
    )

    # Customize line colors: Selected lines are black, unselected lines are white
    fig.update_traces(
        line=dict(
            color='red',  # Make selected lines black
            colorscale=[(0, 'lightgray'), (1, 'red')],  # Unselected lines white, selected lines black
            showscale=True  # Optionally display a color scale
        )
    )

    # Save the plot as an HTML file
    fig.write_html(output_html)

    # Show the plot in the notebook/console
    #fig.show()
    return df_trials  # Optionally return the DataFrame for further analysis or export

def vis_Scatter_HyperCombin(study, filename_prefix):
    """
    visualize_hyperparameters_and_combinations:
    This function generates scatter plots for each hyperparameter vs target,
    and for each two-fold combination of hyperparameters vs the target metric.

    Parameters:
    study: Optuna study object containing trial data.
    filename_prefix: Prefix for saving plot files.
    """

    import matplotlib.pyplot as plt
    import itertools
    import os

    # Create directory to store the plots
    folder_name = "vis_Scatter_HyperCombin"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Collect all hyperparameters keys used in the trials
    all_params_keys = set()
    for trial in study.trials:
        all_params_keys.update(trial.params.keys())

    all_params_keys = list(all_params_keys)  # Convert to list to allow indexing

    # Plot each hyperparameter individually against the target (e.g., weighted loss)
    for param in all_params_keys:
        param_values = [trial.params.get(param) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Create scatter plot for each hyperparameter
        plt.figure()
        plt.scatter(param_values, target_values)
        plt.xlabel(param)
        plt.ylabel('Target (e.g., weighted_loss)')
        plt.title(f'{param} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param}_vs_target.png')
        plt.close()

    # Generate two-fold combinations of hyperparameters and plot them
    param_combinations = list(itertools.combinations(all_params_keys, 2))  # Create all 2-fold combinations

    for param1, param2 in param_combinations:
        param1_values = [trial.params.get(param1) for trial in study.trials]
        param2_values = [trial.params.get(param2) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Create scatter plot for the two-fold combinations
        plt.figure()
        scatter = plt.scatter(param1_values, param2_values, c=target_values, cmap='viridis', edgecolor='k')
        plt.colorbar(scatter, label='Target (e.g., weighted_loss)')
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'{param1} & {param2} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param1}_{param2}_vs_target.png')
        plt.close()

    print(f"Plots saved in folder: {folder_name} with prefix: {filename_prefix}")

def vis_Hexbin_HyperCombin(study, filename_prefix):
    """
    This function generates hexbin plots for each hyperparameter vs target,
    and for each two-fold combination of hyperparameters vs the target metric.

    Parameters:
    study: Optuna study object containing trial data.
    filename_prefix: Prefix for saving plot files.
    """

    import matplotlib.pyplot as plt
    import itertools
    import os

    # Create directory to store the hexbin plots
    folder_name = "vis_Hexbin_HyperCombin"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Collect all hyperparameters keys used in the trials
    all_params_keys = set()
    for trial in study.trials:
        all_params_keys.update(trial.params.keys())

    all_params_keys = list(all_params_keys)  # Convert to list to allow indexing

    # Plot each hyperparameter individually against the target (e.g., weighted loss)
    for param in all_params_keys:
        param_values = [trial.params.get(param) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Create hexbin plot for each hyperparameter
        plt.figure()
        plt.hexbin(param_values, target_values, gridsize=30, cmap='Blues')
        plt.colorbar(label='Counts')
        plt.xlabel(param)
        plt.ylabel('Target (e.g., weighted_loss)')
        plt.title(f'Hexbin Plot: {param} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param}_hexbin_vs_target.png')
        plt.close()

    # Generate two-fold combinations of hyperparameters and plot them
    param_combinations = list(itertools.combinations(all_params_keys, 2))  # Create all 2-fold combinations

    for param1, param2 in param_combinations:
        param1_values = [trial.params.get(param1) for trial in study.trials]
        param2_values = [trial.params.get(param2) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Create hexbin plot for the two-fold combinations
        plt.figure()
        plt.hexbin(param1_values, param2_values, gridsize=30, cmap='Blues')
        plt.colorbar(label='Counts')
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Hexbin Plot: {param1} & {param2} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param1}_{param2}_hexbin_vs_target.png')
        plt.close()

    print(f"Hexbin plots saved in folder: {folder_name} with prefix: {filename_prefix}")

def vis_KDE_HyperCombin(study, filename_prefix):
    """
    This function generates density (KDE) plots for each hyperparameter vs target,
    and for each two-fold combination of hyperparameters vs the target metric.

    Parameters:
    study: Optuna study object containing trial data.
    filename_prefix: Prefix for saving plot files.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    import os

    # Create directory to store the density plots
    folder_name = "vis_KDE_HyperCombin"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Collect all hyperparameters keys used in the trials
    all_params_keys = set()
    for trial in study.trials:
        all_params_keys.update(trial.params.keys())

    all_params_keys = list(all_params_keys)  # Convert to list to allow indexing

    # Plot each hyperparameter individually against the target (e.g., weighted loss)
    for param in all_params_keys:
        param_values = [trial.params.get(param) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Create density (KDE) plot for each hyperparameter
        plt.figure()
        sns.kdeplot(x=param_values, y=target_values, fill=True, cmap='Blues', thresh=0.1, warn_singular=False)
        plt.xlabel(param)
        plt.ylabel('Target (e.g., weighted_loss)')
        plt.title(f'Density Plot: {param} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param}_kde_vs_target.png')
        plt.close()

    # Generate two-fold combinations of hyperparameters and plot them
    param_combinations = list(itertools.combinations(all_params_keys, 2))  # Create all 2-fold combinations

    for param1, param2 in param_combinations:
        param1_values = [trial.params.get(param1) for trial in study.trials]
        param2_values = [trial.params.get(param2) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Create density (KDE) plot for the two-fold combinations
        plt.figure()
        sns.kdeplot(x=param1_values, y=param2_values, fill=True, cmap='Blues', thresh=0.1, warn_singular=False)
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Density Plot: {param1} & {param2} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param1}_{param2}_kde_vs_target.png')
        plt.close()

    print(f"Density plots (KDE) saved in folder: {folder_name} with prefix: {filename_prefix}")

def vis_Bubble_HyperCombin(study, filename_prefix):
    """
    This function generates bubble plots for each hyperparameter vs target,
    and for each two-fold combination of hyperparameters vs the target metric.

    Parameters:
    study: Optuna study object containing trial data.
    filename_prefix: Prefix for saving plot files.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import os
    from collections import Counter

    # Create directory to store the bubble plots
    folder_name = "vis_Bubble_HyperCombin"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Collect all hyperparameters keys used in the trials
    all_params_keys = set()
    for trial in study.trials:
        all_params_keys.update(trial.params.keys())

    all_params_keys = list(all_params_keys)  # Convert to list to allow indexing

    # Plot each hyperparameter individually against the target (e.g., weighted loss)
    for param in all_params_keys:
        param_values = [trial.params.get(param) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Count how many points overlap using Counter
        points_count = Counter(zip(param_values, target_values))

        # Separate the points and the count of occurrences
        unique_points = np.array(list(points_count.keys()))
        counts = np.array(list(points_count.values()))

        # Create bubble plot for each hyperparameter
        plt.figure()
        plt.scatter(unique_points[:, 0], unique_points[:, 1], s=counts * 50, alpha=0.6, edgecolor='black')
        plt.xlabel(param)
        plt.ylabel('Target (e.g., weighted_loss)')
        plt.title(f'Bubble Plot: {param} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param}_bubble_vs_target.png')
        plt.close()

    # Generate two-fold combinations of hyperparameters and plot them
    param_combinations = list(itertools.combinations(all_params_keys, 2))  # Create all 2-fold combinations

    for param1, param2 in param_combinations:
        param1_values = [trial.params.get(param1) for trial in study.trials]
        param2_values = [trial.params.get(param2) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        # Count how many points overlap using Counter for 2D combinations
        points_count = Counter(zip(param1_values, param2_values))

        # Separate the points and the count of occurrences
        unique_points = np.array(list(points_count.keys()))
        counts = np.array(list(points_count.values()))

        # Create bubble plot for the two-fold combinations
        plt.figure()
        plt.scatter(unique_points[:, 0], unique_points[:, 1], s=counts * 50, alpha=0.6, edgecolor='black', cmap='viridis')
        plt.colorbar(label='Target (e.g., weighted_loss)')
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Bubble Plot: {param1} & {param2} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param1}_{param2}_bubble_vs_target.png')
        plt.close()

    print(f"Bubble plots saved in folder: {folder_name} with prefix: {filename_prefix}")

if __name__ == '__main__':
    import pandas as pd
    # INPUT PATHS
    data = r"tus_main_EqPadHHID_RAWDATA_22.csv"
    csv_input = data
    df = pd.read_csv(csv_input)
    #print(df.columns)

    df = df[['Household_ID',
             'months_season', 'week_or_weekend',
             'Occupant_ID_in_HH',
             'Number Family Members',
             'Family Typology',
             'Employment status','Education Degree',
             'Age Classes', 'Region',
             "Marital Status",
             "Kinship Relationship",
             "Nuclear Family, Occupant Profile",
             "Nuclear Family, Typology",
             "Nuclear Family, Occupant Sequence Number",
             "Citizenship",  # new additions
             "Internet Access",
             "Mobile Phone Ownership",
             "Car Ownership",
             'Gender',
             'Family_Typology_Simple', 'Home Ownership', 'Room Count', 'Economic Sector, Profession', 'Job Type',
             'hourStart_Activity', 'hourEnd_Activity',
             'Occupant_Activity','location', 'withNOBODY']]

    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.max_rows', None)  # Display all rows
    #print(df.columns)

    # LSTM:TUNING CROSS-VALIDATION ---------------------------------------------------------------------------
    #main_tuningTransformer_kFold(csv_filepath=csv_input, epochs=125, n_trials=100, num_split=3, df=df) # defualt epochs= 20-30, default n_trials= 50-100, default n_split=3-5
    trainEvaluate_afterTuning(df=df, csv_filepath=csv_input, epochBase=300)
