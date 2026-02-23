import pandas as pd
import random
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
import dill, os, joblib
import torch.nn as nn

# 1st & 2nd step -------------------------------------------------------------------------------------------------------
def select_RES_HH(input, outRES, outRESOCC, num_members=None, residential_id=None):
    """
    Combined function for Step 1 and Step 2 of the pipeline.

    Step 1: Select Residential Data (RES)
      - If 'residential_id' is provided, select that household; otherwise, randomly select one,
        optionally filtering by "Number Family Members".
    Step 2: Assign Household Composition (Alignment Columns)
      - Filter rows where "Residential_ID" matches and ensure the number of rows equals "Number Family Members".
    """
    # Load the censusRES dataset (residential + household demographic data)
    res_df = pd.read_csv(input)

    # Optionally filter by specific number of family members
    if num_members is not None:
        res_df = res_df[res_df["Number Family Members"] == num_members]
        if res_df.empty:
            print(f"No residential units found with exactly {num_members} family members.")
            return

    # If a specific residential_id is provided, use it; otherwise, select randomly.
    if residential_id is not None:
        selected_residential = res_df[res_df["Residential_ID"] == residential_id]
        if selected_residential.empty:
            print(f"No residential unit found with Residential_ID {residential_id}.")
            return
        # Use the first matching residential unit
        selected_residential = selected_residential.iloc[[0]]
        num_family_members = selected_residential["Number Family Members"].values[0]
        selected_household_df = res_df[res_df["Residential_ID"] == residential_id]
        if len(selected_household_df) != num_family_members:
            print(f"Household data does not match for Residential_ID {residential_id}.")
            return
        valid_residential = selected_residential
        valid_household = selected_household_df
    else:
        while True:
            # Randomly select one residential unit
            selected_residential = res_df.sample(n=1, random_state=random.randint(1, 1000))
            residential_id = selected_residential["Residential_ID"].values[0]
            num_family_members = selected_residential["Number Family Members"].values[0]
            selected_household_df = res_df[res_df["Residential_ID"] == residential_id]
            if len(selected_household_df) == num_family_members:
                valid_residential = selected_residential
                valid_household = selected_household_df
                break

    # Save the selected residential and household data to separate CSV files
    valid_residential.to_csv(outRES, index=False)
    valid_household.to_csv(outRESOCC, index=False)

    print(valid_household.columns)
    print("Selected Residential Data:")
    print(valid_residential.to_string(index=False))
    print("\nSelected Household Data for Residential ID:", residential_id)
    print(valid_household.to_string(index=False))

# 3rd step -------------------------------------------------------------------------------------------------------------
def create_household_aggregated_features(household_data):
    """Create household-level aggregated features and return columns."""
    household_aggregated = household_data.groupby('Household_ID').agg({
        'Age Classes': ['mean', 'std'],
        'Education Degree': ['mean', 'std'],
        "Full_Part_time": ['mean', 'std'],
        'Permanent/fixed': ['mean', 'std'],
        "Mobile Phone Ownership": ['mean', 'std'],
        "Marital Status": ['mean', 'std'],
    })

    # Define column names once
    household_aggregated_columns = [
        'Avg_Age_Classes', 'Std_Age_Classes',
        'Avg_Education_Degree', 'Std_Education_Degree',
        'Avg_Full_Part_time', 'Std_Full_Part_time',
        'Avg_Permanent/fixed', 'Std_Permanent/fixed',
        'Avg_Mobile_Phone', 'Std_Mobile_Phone',
        'Avg_Marital_Status', 'Std_Marital_Status',
    ]

    # Assign column names
    household_aggregated.columns = household_aggregated_columns
    household_aggregated.reset_index(inplace=True)
    household_aggregated.fillna(0, inplace=True)

    return household_aggregated, household_aggregated_columns
def add_interaction_features(data):
    interaction_columns = [
        'Region_Employment', 'Kinship_Age', 'Job_Economic_Sector', 'Region_Family_Typology',
        'Employment_HH_Size', 'Education_Job_Type', 'Gender_Kinship', 'Citizenship_Employment',
        'Age_Region', 'Room_Ownership', 'FullPart_Gender', 'FullPart_AgeClass', 'FullPart_Region']
    """Add advanced interaction features."""
    data['Region_Employment'] = data['Region'].astype(str) + "_" + data['Employment status'].astype(str)
    data['Kinship_Age'] = data['Kinship Relationship'].astype(str) + "_" + data['Age Classes'].astype(str)
    data['Job_Economic_Sector'] = data['Job Type'].astype(str) + "_" + data['Economic Sector, Profession'].astype(str)
    data['Region_Family_Typology'] = data['Region'].astype(str) + "_" + data['Family_Typology_Simple'].astype(str)
    data['Employment_HH_Size'] = data['Employment status'].astype(str) + "_" + data['Number Family Members'].astype(str)
    data['Education_Job_Type'] = data['Education Degree'].astype(str) + "_" + data['Job Type'].astype(str)
    data['Gender_Kinship'] = data['Gender'].astype(str) + "_" + data['Kinship Relationship'].astype(str)
    data['Citizenship_Employment'] = data['Citizenship'].astype(str) + "_" + data['Employment status'].astype(str)
    data['Age_Region'] = data['Age Classes'].astype(str) + "_" + data['Region'].astype(str)
    data['Room_Ownership'] = data['Room Count'].astype(str) + "_" + data['Home Ownership'].astype(str)
    data['FullPart_Gender'] = data['Full_Part_time'].astype(str) + "_" + data['Gender'].astype(str)
    data['FullPart_AgeClass'] = data['Full_Part_time'].astype(str) + "_" + data['Age Classes'].astype(str)
    data['FullPart_Region'] = data['Full_Part_time'].astype(str) + "_" + data['Region'].astype(str)
    return data, interaction_columns
def add_feature_engineering_columns(data):
    feature_engineering_columns = [
        'Age_to_Household_Size', 'Education_Degree_to_Age', 'Room_Count_to_Family_Members',
        #'Mobile_Phone_to_Total_Devices',
        "Combined_MainIncomeSource_Feature", 'Stable_Family_Ratio', 'Primary_Kinship_Ratio',
     "Generational_Gap", 'Isolated_Ratio', 'Core_Count', 'Household_Size', 'Connectivity_Score', 'Asset_Mix_Score',
    "Marital_Kinship_Concordance",'Region_Citizenship', ]
    # Normalized Ratios
    data['Age_to_Household_Size'] = data.groupby('Household_ID')['Age Classes'].transform('mean') / data.groupby('Household_ID')['Occupant_ID_in_HH'].transform('count')
    data['Education_Degree_to_Age'] = data['Education Degree'] / (data['Age Classes'] + 1)
    data['Room_Count_to_Family_Members'] = data['Room Count'] / data.groupby('Household_ID')['Occupant_ID_in_HH'].transform('count')
    data['Stable_Family_Ratio'] = data.groupby('Household_ID')['Marital Status'].transform(lambda x: (x == 2).sum() / len(x))
    data['Primary_Kinship_Ratio'] = data.groupby('Household_ID')['Kinship Relationship'].transform(lambda x: (x == 2).sum() / len(x))
    data['Generational_Gap'] = data.groupby('Household_ID')['Age Classes'].transform(lambda x: x.max() - x.min())
    # isolated ratio
    data['Core_Count'] = data.groupby('Household_ID')['Kinship Relationship'].transform(
        lambda x: (x.isin([1, 2])).sum())
    data['Household_Size'] = data.groupby('Household_ID')['Household_ID'].transform('count')
    # Isolation ratio: the higher the ratio, the more members are non-core.
    data['Isolated_Ratio'] = 1 - (data['Core_Count'] / data['Household_Size'])

    data['Connectivity_Score'] = data[['Internet Access', 'Car Ownership',
                                       'Mobile Phone Ownership']].sum(axis=1)
    data['Asset_Mix_Score'] = data['Home Ownership'] + data['Connectivity_Score']
    data['Marital_Kinship_Concordance'] = data.groupby('Household_ID')['Kinship Relationship'].transform(lambda x: (x == 2).sum() / len(x))
    data['Region_Citizenship'] = data['Region'].astype(str) + "_" + data['Citizenship'].astype(str)
    data['Household_Diversity'] = data.groupby('Household_ID')['Kinship Relationship'].transform(lambda x: x.nunique() / x.count())

    data['Combined_MainIncomeSource_Feature'] = data['Marital Status'].astype(str) + "_" + \
                                                data['Employment status'].astype(str) + "_" + \
                                                data['Education Degree'].astype(str) + "_" + \
                                                data['Kinship Relationship'].astype(str) + "_" + \
                                                data['Economic Sector, Profession'].astype(str) + "_" + \
                                                data['Job Type'].astype(str) + "_" + \
                                                data['Age Classes'].astype(str) + "_" + \
                                                data['Full_Part_time'].astype(str)

    return data, feature_engineering_columns
def add_normalized_ratios(data):
    normalized_ratio_columns = [
        'Age_Diversity', 'Room_Per_Member', 'Education_Per_Member', 'Occupancy_Density',
        'Kinship_Diversity', 'PartTime_FullTime_Ratio', 'Employment_Intensity', 'FullTime_Per_Member'
    ]
    """Add normalized ratios for advanced feature engineering."""
    data['Age_Diversity'] = data['Std_Age_Classes'] / data['Number Family Members']
    data['Room_Per_Member'] = data['Room Count'] / data['Number Family Members']
    data['Education_Per_Member'] = data['Avg_Education_Degree'] / data['Number Family Members']
    data['Occupancy_Density'] = data['Number Family Members'] / data['Room Count']
    data['Kinship_Diversity'] = data.groupby('Household_ID')['Kinship Relationship'].transform('nunique') / data['Number Family Members']
    data['PartTime_FullTime_Ratio'] = data.groupby('Household_ID')['Full_Part_time'].transform(
        lambda x: (x == 1).sum() / (x == 2).sum() if (x == 2).sum() != 0 else 0)
    data['Employment_Intensity'] = data.groupby('Household_ID')['Full_Part_time'].transform(
        lambda x: (x == 2).sum() / ((x == 1).sum() + (x == 2).sum()) if ((x == 1).sum() + (x == 2).sum()) != 0 else 0)
    data['FullTime_Per_Member'] = data.groupby('Household_ID')['Full_Part_time'].transform(
        lambda x: (x == 2).sum() / len(x))
    return data, normalized_ratio_columns
def add_household_composition_features(data):
    household_composition_columns = [
        'Generational_Diversity', 'Gender_Diversity', 'Role_Concentration', 'Marital_Diversity'
    ]
    """Add household composition features."""
    data['Generational_Diversity'] = data.groupby('Household_ID')['Age Classes'].transform('nunique')
    data['Gender_Diversity'] = data.groupby('Household_ID')['Gender'].transform(
        lambda x: (x == 1).sum() / (x == 2).sum() if (x == 2).sum() != 0 else float('inf'))
    data['Role_Concentration'] = data.groupby('Household_ID')['Kinship Relationship'].transform(
        lambda x: x.value_counts(normalize=True).iloc[0])
    data['Marital_Diversity'] = data.groupby('Household_ID')['Marital Status'].transform('nunique')
    return data, household_composition_columns
def preprocess_data(data, output_columns, base_nominal_columns):
    # Perform feature engineering
    data, household_aggregated_columns, interaction_columns, normalized_ratio_columns, household_composition_columns, feature_engineering_columns = feature_engineering(data)

    # PREPROCESSING
    ordinal_raw_columns = ['Occupant_ID_in_HH', 'Number Family Members', 'Age Classes', 'Education Degree', "Room Count"]

    # Flattened list of nominal columns (including outputs to be dropped later)
    nominal_columns = base_nominal_columns + interaction_columns + normalized_ratio_columns + \
                      household_composition_columns + feature_engineering_columns + output_columns

    # Ordinal Encoding
    ordinal_encoder = OrdinalEncoder()
    data[ordinal_raw_columns] = ordinal_encoder.fit_transform(data[ordinal_raw_columns])

    # Scaling
    scaler = RobustScaler()
    data[ordinal_raw_columns + household_aggregated_columns] = scaler.fit_transform(
        data[ordinal_raw_columns + household_aggregated_columns])

    # Nominal Encoding
    nominal_encodings = {}
    for col in nominal_columns:
        unique_values = sorted(data[col].unique())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        data[col] = data[col].map(mapping)
        nominal_encodings[col] = mapping

    return data, nominal_encodings, nominal_columns, ordinal_raw_columns, household_aggregated_columns
class CustomDataset(Dataset):
    def __init__(self, X, y, nominal_columns, ordinal_columns, output_columns):
        self.nominal_data = X[nominal_columns].values
        self.ordinal_data = X[ordinal_columns].values
        self.targets = {col: y[col].values for col in output_columns}  # Dictionary of target columns

    def __len__(self):
        return len(next(iter(self.targets.values())))  # Length of any output column

    def __getitem__(self, idx):
        x_nominal = torch.tensor(self.nominal_data[idx], dtype=torch.long)  # Long for embeddings
        x_ordinal = torch.tensor(self.ordinal_data[idx], dtype=torch.float32)
        y = {col: torch.tensor(self.targets[col][idx], dtype=torch.long) for col in self.targets.keys()}  # Dictionary of targets
        return x_nominal, x_ordinal, y
def feature_engineering(data):
    """
    Perform all feature engineering steps and return updated data and feature column lists.
    """
    # Household-level aggregated features
    household_data = data[['Household_ID', 'Occupant_ID_in_HH', 'Age Classes', 'Education Degree', "Room Count",
                           "Full_Part_time", 'Mobile Phone Ownership', 'Marital Status', 'Permanent/fixed']].copy()

    household_aggregated, household_aggregated_columns = create_household_aggregated_features(household_data)
    data = data.merge(household_aggregated, on='Household_ID', how='left')

    # Advanced Feature Engineering
    data, interaction_columns = add_interaction_features(data)
    data, normalized_ratio_columns = add_normalized_ratios(data)
    data, household_composition_columns = add_household_composition_features(data)
    data, feature_engineering_columns = add_feature_engineering_columns(data)

    # Drop Household_ID after aggregation
    data = data.drop(columns=['Household_ID'])

    return data, household_aggregated_columns, interaction_columns, normalized_ratio_columns, household_composition_columns, feature_engineering_columns
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, output_columns, base_nominal_columns):
        self.output_columns = output_columns
        self.base_nominal_columns = base_nominal_columns

    def fit(self, X, y=None):
        # Perform feature engineering on training data.
        # Assume feature_engineering returns:
        #   data, household_aggregated_columns, interaction_columns,
        #   normalized_ratio_columns, household_composition_columns, feature_engineering_columns
        X_fe, self.household_aggregated_columns_, self.interaction_columns_, \
            self.normalized_ratio_columns_, self.household_composition_columns_, \
            self.feature_engineering_columns_ = feature_engineering(X.copy())

        # Define ordinal columns.
        self.ordinal_raw_columns_ = ['Occupant_ID_in_HH', 'Number Family Members',
                                     'Age Classes', 'Education Degree', "Room Count"]

        # Build nominal columns including engineered features and outputs.
        all_nominal_columns = (self.base_nominal_columns + self.interaction_columns_ +
                               self.normalized_ratio_columns_ + self.household_composition_columns_ +
                               self.feature_engineering_columns_ + self.output_columns)
        self.nominal_columns_ = all_nominal_columns

        # Create input nominal columns by removing outputs.
        self.input_nominal_columns_ = [col for col in self.nominal_columns_ if col not in self.output_columns]

        # Fit ordinal encoder on ordinal columns.
        self.ordinal_encoder_ = OrdinalEncoder()
        X_fe[self.ordinal_raw_columns_] = self.ordinal_encoder_.fit_transform(X_fe[self.ordinal_raw_columns_])

        # Fit scaler on ordinal + household aggregated columns.
        self.scaler_ = RobustScaler()
        scale_cols = self.ordinal_raw_columns_ + self.household_aggregated_columns_
        X_fe[scale_cols] = self.scaler_.fit_transform(X_fe[scale_cols])

        # Build nominal encoding mappings and apply mapping on training data.
        self.nominal_encodings_ = {}
        for col in self.nominal_columns_:
            unique_values = sorted(X_fe[col].unique())
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            X_fe[col] = X_fe[col].map(mapping)
            self.nominal_encodings_[col] = mapping

        # Define final input columns: use input_nominal_columns plus ordinal and aggregated columns.
        self.input_columns_ = self.input_nominal_columns_ + self.ordinal_raw_columns_ + self.household_aggregated_columns_
        return self

    def transform(self, X):
        import pandas as pd
        X_transformed = X.copy()

        # Apply feature engineering to new data.
        X_transformed, _, _, _, _, _ = feature_engineering(X_transformed)

        # Transform ordinal columns.
        X_transformed[self.ordinal_raw_columns_] = self.ordinal_encoder_.transform(
            X_transformed[self.ordinal_raw_columns_])

        # Scale ordinal and household aggregated columns.
        scale_cols = self.ordinal_raw_columns_ + self.household_aggregated_columns_
        X_transformed[scale_cols] = self.scaler_.transform(X_transformed[scale_cols])

        # Apply nominal encoding using training mappings.
        for col in self.nominal_columns_:
            if col in X_transformed.columns:
                mapping = self.nominal_encodings_.get(col, {})
                X_transformed[col] = X_transformed[col].map(mapping).fillna(0).astype(int)
        return X_transformed
def calculate_embedding_sizes(nominal_encodings):
    embedding_sizes = []
    for col, mapping in nominal_encodings.items():
        num_categories = len(mapping)
        embedding_dim = min(50, num_categories // 2)
        embedding_sizes.append((num_categories, embedding_dim))
    return embedding_sizes
class MultiOutputNeuralNetwork(nn.Module):
    def __init__(
        self,
        embedding_sizes,
        num_ordinal_features,
        output_sizes,  # Dictionary with output column names and their number of classes
        ordinal_hidden_size=64,
        fc_hidden_sizes=(128, 64),
        dropout_rate=0.3,
        use_batch_norm=True,
        output_dropout_rate=None,  # New dictionary for output-specific dropout
    ):
        super(MultiOutputNeuralNetwork, self).__init__()

        # Embedding layers for nominal columns
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories, embedding_dim in embedding_sizes]
        )

        # Dense layer for ordinal inputs
        ordinal_layers = [
            nn.Linear(num_ordinal_features, ordinal_hidden_size),
            nn.ReLU(),
        ]
        if use_batch_norm:
            ordinal_layers.append(nn.BatchNorm1d(ordinal_hidden_size))
        if dropout_rate > 0:
            ordinal_layers.append(nn.Dropout(dropout_rate))
        self.ordinal_layer = nn.Sequential(*ordinal_layers)

        # Fully connected layers for combined features
        fc_layers = []
        input_size = ordinal_hidden_size + sum([embedding_dim for _, embedding_dim in embedding_sizes])
        for hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(input_size, hidden_size))
            fc_layers.append(nn.ReLU())
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        self.fc = nn.Sequential(*fc_layers)

        # Output heads for multi-output classification
        self.output_layers = nn.ModuleDict({
            output_name: nn.Linear(fc_hidden_sizes[-1], num_classes)
            for output_name, num_classes in output_sizes.items()
        })

        # Dropout layers for specific outputs
        self.output_dropout_layers = nn.ModuleDict({
            output_name: nn.Dropout(output_dropout_rate.get(output_name, 0.0))
            for output_name in output_sizes.keys()
        })

    def forward(self, x_nominal, x_ordinal):
        # Pass nominal inputs through embeddings
        embedded = [embedding(x) for embedding, x in zip(self.embeddings, x_nominal)]
        embedded = torch.cat(embedded, dim=1)

        # Pass ordinal inputs through dense layer
        ordinal_out = self.ordinal_layer(x_ordinal)

        # Combine embedded and ordinal features
        combined = torch.cat([embedded, ordinal_out], dim=1)
        fc_out = self.fc(combined)

        # Compute outputs for each target
        outputs = {}
        for output_name, layer in self.output_layers.items():
            head_output = layer(fc_out)
            head_output = self.output_dropout_layers[output_name](head_output)  # Apply output-specific dropout
            outputs[output_name] = head_output
        return outputs
def classify_new_data(model_path, new_data, output_columns):
    """
    Classifies new data using the trained multi-output model.

    This function:
      1. Instantiates the custom preprocessor with the same parameters used during training.
      2. Transforms the new data to generate the input features using the preprocessor.
      3. Extracts the nominal input columns (preprocessor.input_nominal_columns_) and the ordinal features
         (a concatenation of preprocessor.ordinal_raw_columns_ and preprocessor.household_aggregated_columns_).
      4. Calculates the embedding sizes (from the preprocessor.nominal_encodings_ for the nominal input columns)
         and the output sizes (from preprocessor.nominal_encodings_ for each target column).
      5. Instantiates the MultiOutputNeuralNetwork model with the same hyperparameters as in training,
         loads the saved model weights from the provided .pth file, and sets the model to evaluation mode.
      6. Feeds the new data to the model to obtain predictions.
      7. Converts the predicted indices back to their original categorical labels using the reverse mappings.

    Parameters:
        model_path (str): Path to the saved .pth model file.
        new_data (DataFrame): New data to be classified.
        output_columns (list): List of target output columns.
        base_nominal_columns (list): List of base nominal columns used during training.

    Returns:
        dict: A dictionary mapping each output column to its predicted label(s).
    """

    # Transform new_data using the preprocessor.
    # (Assumes that the preprocessor has been fitted on training data and that its state
    #  is available; otherwise, you should load the fitted preprocessor.)
    preprocessor = joblib.load(r'dataset_CENTUS/inputs100HHID/preprocessor.pkl')
    new_data_processed = preprocessor.transform(new_data)

    # Define input features for the model:
    #   - Nominal features: the ones used during training (excludes target columns).
    #   - Ordinal features: the raw ordinal columns plus household aggregated features.
    input_nominal_cols = preprocessor.input_nominal_columns_
    ordinal_cols = preprocessor.ordinal_raw_columns_ + preprocessor.household_aggregated_columns_

    # Convert the processed features to tensors.
    x_nominal = torch.tensor(new_data_processed[input_nominal_cols].values, dtype=torch.long)
    x_ordinal = torch.tensor(new_data_processed[ordinal_cols].values, dtype=torch.float32)

    # Calculate embedding sizes for the nominal input features.
    embedding_sizes = calculate_embedding_sizes({
        col: preprocessor.nominal_encodings_[col] for col in input_nominal_cols
    })

    # Determine output sizes for each target column.
    output_sizes = {col: len(preprocessor.nominal_encodings_[col]) for col in output_columns}

    # Define output dropout rates as used during training.
    output_dropout_rate = {
        "Family Typology": 0.00,
        "Nuclear Family, Occupant Profile": 0.15,
        "Nuclear Family, Typology": 0.15,
        "Nuclear Family, Occupant Sequence Number": 0.21,
    }

    # Instantiate the model with the same hyperparameters as during training.
    model = MultiOutputNeuralNetwork(
        embedding_sizes=embedding_sizes,
        num_ordinal_features=len(ordinal_cols),
        output_sizes=output_sizes,
        ordinal_hidden_size=24,
        fc_hidden_sizes=(24,),
        dropout_rate=0.0,
        use_batch_norm=True,
        output_dropout_rate=output_dropout_rate
    )

    # Load the saved model weights.
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # Generate predictions.
    with torch.no_grad():
        # The model expects a list of tensors for the nominal features (one per column).
        outputs = model([x_nominal[:, i] for i in range(x_nominal.shape[1])], x_ordinal)

    # Convert raw outputs (logits) to predicted labels by selecting the index with the maximum value,
    # and then mapping that index back to the original category.
    predictions = {}
    for col in output_columns:
        predicted_indices = outputs[col].argmax(dim=1).cpu().numpy()
        reverse_mapping = {v: k for k, v in preprocessor.nominal_encodings_[col].items()}
        predictions[col] = [reverse_mapping[idx] for idx in predicted_indices]

    return predictions
def compute_accuracy(predictions, true_data, output_columns):
    """
    Computes accuracy for each output column by comparing predicted labels with true labels.

    Parameters:
        predictions (dict): Dictionary of predicted labels for each output column.
        true_data (DataFrame): DataFrame containing the ground truth labels.
        output_columns (list): List of target output column names.

    Returns:
        dict: A dictionary mapping each output column to its accuracy score.
    """
    accuracies = {}
    for col in output_columns:
        # Convert ground truth values to a list (or numpy array)
        y_true = true_data[col].tolist()
        y_pred = predictions[col]
        accuracies[col] = accuracy_score(y_true, y_pred)
    return accuracies
def main_classify_new_data(input_path, trained_model_AI_ML, output_path=None):
    import pandas as pd

    # Read the CSV file containing the full dataset.
    new_data = pd.read_csv(input_path)

    # Define the target output columns.
    output_columns = [
        'Family Typology',
        'Nuclear Family, Occupant Profile',
        'Nuclear Family, Typology',
        'Nuclear Family, Occupant Sequence Number'
    ]

    # Call classify_new_data() using the saved model weights.
    predictions = classify_new_data(trained_model_AI_ML, new_data, output_columns)

    # Print the predictions.
    print("Predictions:")
    for col, preds in predictions.items():
        print(f"{col}: {preds}")

    # Append the classified output columns to the original dataframe.
    for col in output_columns:
        if col in predictions:
            new_data[col] = predictions[col]

    # Save the combined dataframe as a CSV if an output path is provided.
    if output_path:
        new_data.to_csv(output_path, index=False)
        print(f"Classified data saved to {output_path}")

    # Print the selected household data corresponding to the residential unit
    print("\nSelected Household Data with all columns, aligned and non-aligned, CENTUS:", new_data["Residential_ID"].values[0])
    print(new_data.to_string(index=False))

    return new_data
#4th step---------------------------------------------------------------------------------------------------------------
#POSTPROCESS------------------------------------------------------------------------------------------------------------
def temporalConversion(input_path, output_path):
    import pandas as pd
    import numpy as np

    # Read dataset
    df = pd.read_csv(input_path)

    # Duplicate each row 24 times
    df_processed = df.loc[df.index.repeat(24)].reset_index(drop=True)

    n = len(df)  # Number of original rows

    # Add hourStart_Activity: for each block of 24 rows, values 0 to 23
    df_processed['hourStart_Activity'] = np.tile(np.arange(24), n)

    # Add hourEnd_Activity: for each block of 24 rows, values 1 to 24
    df_processed['hourEnd_Activity'] = np.tile(np.arange(1, 25), n)

    # Add months_season: randomly select an integer from 1 to 4 for each original row and repeat it 24 times
    months = np.random.randint(1, 5, size=n)  # 1 to 4
    df_processed['months_season'] = np.repeat(months, 24)

    # Add week_or_weekend: randomly select an integer from 1 to 3 for each original row and repeat it 24 times
    week = np.random.randint(1, 4, size=n)  # 1 to 3
    df_processed['week_or_weekend'] = np.repeat(week, 24)

    # Save the processed dataset
    df_processed.to_csv(output_path, index=False)
    print("Processed dataset saved to", output_path)

    return df_processed
# 5th step -------------------------------------------------------------------------------------------------------------
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
                    'Family_Typology_Simple',
                    'Home Ownership',
                    'Room Count',
                    'Economic Sector, Profession',
                    'Job Type', # new alignment columns
                    'Occupant_ID_in_HH',
                    'months_season',
                    'week_or_weekend',]

    for col in one_hot_cols:
        df[col] = df[col].astype('category').cat.codes

    # Desired column order based on impact analysis
    impactAnalysis_order = ['Household_ID',
                            'Education Degree',
                            'Employment status',
                            'Gender',
                            'Family Typology',
                            'Number Family Members',
                            'Age Classes',
                            'Region',
                            "Marital Status",
                            "Kinship Relationship",
                            "Nuclear Family, Occupant Profile",
                            "Nuclear Family, Typology",
                            "Nuclear Family, Occupant Sequence Number",
                            "Citizenship",  # new additions
                            "Internet Access",
                            "Mobile Phone Ownership",
                            "Car Ownership",
                            'Family_Typology_Simple',
                            'Home Ownership',
                            'Room Count',
                            'Economic Sector, Profession',
                            'Job Type',  # new alignment columns
                            'Occupant_ID_in_HH',
                            'months_season',
                            'week_or_weekend',
                            'hourStart_Activity',
                            'hourEnd_Activity',]
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
    to_scale = df.columns.difference(one_hot_cols + ["Gender", "Citizenship", "Internet Access", "Mobile Phone Ownership", "Car Ownership",])
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # 5. Data Splitting based on unique Household_ID and Occupant_ID_in_HH combinations, stratified sampling
    train_data_ids, temp_data_ids = stratified_split(df, test_size=0.5) #0.3 default
    train_data = df.merge(train_data_ids, on=['Household_ID', 'Occupant_ID_in_HH'])
    # Drop 'Household_ID' column post splitting
    train_data.drop(columns=['Household_ID'], inplace=True)
    #ENCODING TARGET VARIABLE - LABEL ENCODING -------------------------------------------------------------------------
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Reshape for LSTM
    X_train = train_data.values.reshape(-1, 24, train_data.shape[1])
    return X_train

#MODELING---------------------------------------------------------------------------------------------------------------
import torch.nn as nn
class RNNsModelTuning(nn.Module):
    def __init__(self,num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat,num_numFamMembCat,
                 num_ageClassCat, num_regionCat,
                 num_MartStatCat, num_KinsCat, num_OccProfCat, num_FamTypoCat, num_OccSeqNumCat,
                 num_CitizenCat,  num_InterOwnCat, num_MobPhoneOwnCat, num_CarOwnCat,
                 num_FamTypoSimpleCat, num_HomeOwnCat, num_RoomCountCat, num_EcoSectorCat, num_JobTypeCat,
                 num_OCCinHHCat,
                 num_seasonCat, num_unique_weekCat,
                 num_continuous_features,
                 output_dim_activity, output_dim_location, output_dim_withNOB,
                 num_hidden_layers, hidden_units,
                 rnn_type,
                 dropout_loc, dropout_withNOB, dropout_embedding, dropout_RNNs,
                 embed_size):

        super(RNNsModelTuning, self).__init__()
        embed_size = embed_size
        self.rnn_type = rnn_type
        self.num_continuous_features = num_continuous_features

        self.activation_act = nn.ReLU()
        self.activation_binary = nn.Tanh()

        # Define embedding dimensions for all categorical features
        # Occupant Demographics: Default
        self.embedding_dim_education = min(embed_size, num_educationCat // 2 + 2)
        self.embedding_dim_employment = min(embed_size, num_employmentCat // 2 + 2)
        self.embedding_dim_gender = min(embed_size, num_genderCat // 2 + 1)
        self.embedding_dim_famTypology = min(embed_size, num_famTypologyCat // 2 + 2)
        self.embedding_dim_numFamMemb = min(embed_size, num_numFamMembCat // 2 + 2)
        self.embedding_dim_ageClass = min(embed_size, num_ageClassCat // 2 + 1)
        self.embedding_dim_region = min(embed_size, num_regionCat // 2 + 1)
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

        # Dynamic LSTM layers initialization
        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.shared_rnns = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # Use LayerNorm instead of BatchNorm1d
        self.dropouts = nn.ModuleList()
        self.multiplier = 2 # by default, all RNNs options accepted as bidirectional

        for i in range(num_hidden_layers):
            hidden_size = hidden_units[i % len(hidden_units)]
            if self.rnn_type == 'LSTM':
                rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            elif self.rnn_type == 'RNN':
                rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            else:
                raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', 'RNN'.")
            self.shared_rnns.append(rnn_layer)
            self.layer_norms.append(nn.LayerNorm(hidden_size * self.multiplier))  # Use LayerNorm here
            self.dropouts.append(nn.Dropout(p=dropout_RNNs))
            input_size = hidden_size * self.multiplier

        # Add a global LayerNorm for the concatenated features
        self.global_layer_norm = nn.LayerNorm(normalized_shape=total_embedding_dim + num_continuous_features) #this one for embeddings

        # Activity output layer
        self.activity_layer_norm = nn.LayerNorm(input_size)  # Change to LayerNorm
        self.activity_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_activity)  # Assuming bidirectional LSTM

        # Location output layer with Dropout: Dropout is used to help prevent overfitting
        self.location_dropout = nn.Dropout(p=dropout_loc)
        self.location_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_location)  # Assuming bidirectional LSTM

        # withNOBODY output layer with Dropout: Dropout is used to help prevent overfitting
        self.withNOB_dropout = nn.Dropout(p=dropout_withNOB)
        self.withNOB_dense = nn.Linear(hidden_units[-1] * self.multiplier, output_dim_withNOB)  # Assuming bidirectional LSTM

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
        famTypology_embedded = self.famTypology_embedding(famTypology_input).reshape(-1, 24,
                                                                                     self.embedding_dim_famTypology)
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
        MobPhoneOwn_embedded = self.MobPhoneOwn_embedding(MobPhoneOwn_input).reshape(-1, 24,
                                                                                     self.embedding_dim_MobPhoneOwn)
        CarOwn_embedded = self.CarOwn_embedding(CarOwn_input).reshape(-1, 24, self.embedding_dim_CarOwn)
        # Embeddings:Added Demographics 3
        FamTypoSimple_embedded = self.FamTypoSimple_embedding(FamTypoSimple_input).reshape(-1, 24,
                                                                                           self.embedding_dim_FamTypoSimple)
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

        # Normalize concatenated features
        concatenated_features = self.global_layer_norm(concatenated_features)

        rnn_out = concatenated_features

        # Dynamic RNNs processing
        for rnn_layer, layer_norm, dropout in zip(self.shared_rnns, self.layer_norms, self.dropouts):
            rnn_out, _ = rnn_layer(rnn_out)
            rnn_out = layer_norm(rnn_out)  # Apply LayerNorm directly to rnn_out
            rnn_out = dropout(rnn_out) # Dropout applied here, controlled by dropout_RNNs parameter

        # BRANCHING OUT TO OCCUPANT_ACTIVITY, The model has distinct output layers which allows for specialized processing for each target.
        # Activity
        activity_output = self.activity_dense(self.activation_act(rnn_out.contiguous().reshape(-1, rnn_out.shape[2])))
        activity_output = activity_output.reshape(-1, 24, activity_output.shape[-1])  # Reshape to sequence form

        # Location with Dropout
        location_output = self.location_dropout(rnn_out)
        location_output = self.location_dense(self.activation_binary(location_output.contiguous().reshape(-1, location_output.shape[2])))
        location_output = location_output.reshape(-1, 24, location_output.shape[-1])  # Reshape to sequence form

        # WithNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(rnn_out)
        withNOB_output = self.withNOB_dense(self.activation_binary(withNOB_output.contiguous().reshape(-1, withNOB_output.shape[2])))
        withNOB_output = withNOB_output.reshape(-1, 24, withNOB_output.shape[-1])  # Reshape to sequence form

        return activity_output, location_output, withNOB_output
#PROCESS-------------------------------------------------------------------------------------------------------------
def classify_afterTuning_withNewData(model, X_new, df, label_encoder, mps_device, csv_filepath, arch_type, output_path=None,):
    import os, torch, pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    file_name = f"{filename_prefix}_model_predictions_{arch_type}_newData.csv"

    # Indices for categorical features (adjust these if needed)
    educationDegree_idx = 0
    employmentStatus_idx = 1
    gender_idx = 2
    famTypology_idx = 3
    numFamMembers_idx = 4
    ageClass_idx = 5
    region_idx = 6
    MartStat_idx = 7
    Kins_idx = 8
    OccProf_idx = 9
    FamTypo_idx = 10
    OccSeqNum_idx = 11
    Citizen_idx = 12
    InterOwn_idx = 13
    MobPhoneOwn_idx = 14
    CarOwn_idx = 15
    FamTypoSimple_idx = 16
    HomeOwn_idx = 17
    RoomCount_idx = 18
    EcoSector_idx = 19
    JobType_idx = 20
    OCCinHH_idx = 21
    season_idx = 22
    weekend_idx = 23
    num_categorical_features = 24  # Total number of categorical columns

    # Build a dataset using only features (no targets)
    dataset = TensorDataset(
        torch.tensor(X_new[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, numFamMembers_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, ageClass_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, region_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, MartStat_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, Kins_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, OccProf_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, FamTypo_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, OccSeqNum_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, Citizen_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, InterOwn_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, MobPhoneOwn_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, CarOwn_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, FamTypoSimple_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, HomeOwn_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, RoomCount_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, EcoSector_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, JobType_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, OCCinHH_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, season_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X_new[:, :, num_categorical_features:], dtype=torch.float)  # Continuous data
    )

    loader = DataLoader(dataset, batch_size=24, shuffle=False)

    all_activity_preds = []
    all_location_preds = []
    all_withNOB_preds = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            features = [d.to(mps_device) for d in data]
            activity_output, location_output, withNOB_output = model(*features)
            _, predicted_activity = torch.max(activity_output, 2)
            predicted_location = torch.round(torch.sigmoid(location_output)).int()
            predicted_withNOB = torch.round(torch.sigmoid(withNOB_output)).int()

            all_activity_preds.extend(predicted_activity.reshape(-1).cpu().numpy())
            all_location_preds.extend(predicted_location.reshape(-1).cpu().numpy())
            all_withNOB_preds.extend(predicted_withNOB.reshape(-1).cpu().numpy())

    # Convert predicted activity labels to original categories
    predicted_activity_categories = label_encoder.inverse_transform(all_activity_preds)


    # Add predictions as new columns in the original dataframe
    df['Classified_Occupant_Activity'] = predicted_activity_categories
    df['Classified_Location'] = all_location_preds
    df['Classified_withNOBODY'] = all_withNOB_preds

    # Check and drop specific columns if they exist
    for col in ['Occupant_Activity', 'location', 'withNOBODY']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"classify_afterTuning_{arch_type} is completed")
    return df
def AI_DLclassification(input_filepath, model_type='LSTM', output_path=None):
    import torch
    import json
    import os
    import pickle
    df = pd.read_csv(input_filepath)
    X_new = data_preprocess(df)

    encoder_path = r"dataset_CENTUS/inputs100HHID/occupant_activity_label_encoder.pkl"
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    #print("Label encoder loaded from", encoder_path)

    # MODELING---------------------------------------------------------------------------------------------------------------
    # Load best hyperparameters
    with open(f"dataset_CENTUS/inputs100HHID/tus_main_EqPad100HHID_RAWDATA_22_best_TuningParams_LSTM.json", 'r') as infile:
        best_params = json.load(infile)
    print(best_params)

    # Embedding dimensions and other parameters
    def load_num_features(filename):
        with open(filename, 'r') as f:
            num_features = json.load(f)
        print(f"num_features loaded from {filename}")
        return num_features

    num_features = load_num_features(r"dataset_CENTUS/inputs100HHID/num_features.json")
    dropout_loc = 0.25
    dropout_withNOB = 0.1
    dropout_embedding = 0
    dropout_RNNs = 0

    # Unpack hyperparameters
    num_hidden_layers = best_params['num_hidden_layers']
    hidden_units = [best_params[f'hidden_units_l{i}'] for i in range(num_hidden_layers)]
    embed_size = best_params['embedding_size']
    # The output dimension based on the one-hot encoded target
    output_dim_activity = 145
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary
    mps_device = torch.device("mps")

    model = RNNsModelTuning(
        **num_features,
        output_dim_activity=output_dim_activity,
        output_dim_location=output_dim_location,
        output_dim_withNOB=output_dim_withNOB,
        num_hidden_layers=num_hidden_layers,
        hidden_units=hidden_units,
        rnn_type=model_type,
        dropout_loc=dropout_loc,
        dropout_withNOB=dropout_withNOB,
        dropout_embedding=dropout_embedding,
        dropout_RNNs=dropout_RNNs,
        embed_size=embed_size,
    ).to(mps_device)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.load_state_dict(torch.load(
        "dataset_CENTUS/inputs100HHID/tus_main_EqPad100HHID_RAWDATA_22_best_model_LSTM.pth",
        map_location=device,
    ))

    classify_afterTuning_withNewData(model, X_new, df, label_encoder, mps_device, input_filepath, arch_type=model_type, output_path=output_path)

#VISUALIZATION-------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
def visualize_predictions_all_occupants(csv_path):
    """
    Loads the CSV and visualizes predictions for each family member (Occupant_ID_in_HH) over time using line plots.
    For each occupant, three subplots are created (one per classification column) with 'hourEnd_Activity'
    on the x-axis and the corresponding categorical prediction (as numeric codes) on the y-axis.
    """
    df = pd.read_csv(csv_path)

    # Define prediction columns
    classification_cols = [
        "Classified_Occupant_Activity",
        "Classified_Location",
        "Classified_withNOBODY"
    ]

    # Get unique occupant IDs
    occupant_ids = df["Occupant_ID_in_HH"].unique()
    n_occupants = len(occupant_ids)

    # Create a grid: rows for occupants, columns for each classification column
    fig, axes = plt.subplots(nrows=n_occupants, ncols=len(classification_cols),
                             figsize=(5 * len(classification_cols), 4 * n_occupants), sharex=True)

    # Ensure axes is 2D even if only one occupant
    if n_occupants == 1:
        axes = axes.reshape(1, -1)

    for i, occ in enumerate(occupant_ids):
        occ_df = df[df["Occupant_ID_in_HH"] == occ]
        # Sort by time to ensure a proper line plot
        occ_df = occ_df.sort_values("hourEnd_Activity")

        for j, col in enumerate(classification_cols):
            # Convert prediction column to categorical and extract numeric codes and categories
            cat_series = pd.Categorical(occ_df[col])
            codes = cat_series.codes
            categories = cat_series.categories

            ax = axes[i, j]
            # Use a line plot with markers
            ax.plot(occ_df["hourEnd_Activity"], codes, color='blue', marker='o', linestyle='-')
            ax.set_title(f"{col} (Occupant {occ})")
            ax.set_xlabel("Hour End Activity")
            ax.set_ylabel("Category")
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # pipeline-visual: https://docs.google.com/presentation/d/1l-rP260eG2kS4SEYLNSfmJEtv2A_DmmVPuwYJNKHF9o/edit#slide=id.g32308f188aa_0_124
    # INPUTS
    census_main_merged =  r'dataset_CENSUS_main/Census_main_merged.csv'

    # OUTPUTS ##########################################################################################################
    # https://docs.google.com/presentation/d/1l-rP260eG2kS4SEYLNSfmJEtv2A_DmmVPuwYJNKHF9o/edit#slide=id.g2d7e0a07f9f_0_406
    # 1st step
    selResfromCENSUS = r'dataset_CENTUS/selected_residential.csv'
    # 2nd step
    selResOCCfromCENSUS = "dataset_CENTUS/selected_residential_household.csv"  # New CSV to save selected household data
    # 3rd step
    alignedResOCCwithCENTUS = "dataset_CENTUS/aligned_residential_household.csv"  # New CSV to save aligned household data
    # 4th step
    EqPadHHID_ResOCCwithCENTUS= "dataset_CENTUS/EqPadHHID_residential_household.csv"  # New CSV to save processed household data
    # 5th step
    classfied_EqPadHHID_ResOCCwithCENTUS = "dataset_CENTUS/classified_EqPadHHID_residential_household.csv"  # New CSV to save processed household data

    # CLASSIFICATION ###################################################################################################
    # 1st & 2nd step: selecting residential data and assinging household
    select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None)
    #select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None, residential_id=15829804)

    # 3rd step: aligning the household data of residentials
    trained_model = r"dataset_CENTUS/inputs100HHID/multiOut_bestModel.pth"
    main_classify_new_data(input_path=selResOCCfromCENSUS, trained_model_AI_ML=trained_model, output_path=alignedResOCCwithCENTUS)

    # 4th step: converting to temporal data with equalize sequences and padding, 60-minute intervals
    temporalConversion(input_path=alignedResOCCwithCENTUS, output_path=EqPadHHID_ResOCCwithCENTUS)

    # 5th step: temporal data augmentation
    AI_DLclassification(input_filepath=EqPadHHID_ResOCCwithCENTUS, model_type='LSTM', output_path=classfied_EqPadHHID_ResOCCwithCENTUS)

    # 5th step-Visualization: temporal data augmentation
    visualize_predictions_all_occupants(csv_path=classfied_EqPadHHID_ResOCCwithCENTUS)

