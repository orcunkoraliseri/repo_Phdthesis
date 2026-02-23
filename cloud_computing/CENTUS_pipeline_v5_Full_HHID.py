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
def main_classify_new_data(input_path, trained_model_AI_ML, output_path=None, n_households=None):
    """
    Main function to load data, select a specific number of households,
    and run classification.

    Parameters:
        input_path (str): Path to the input CSV.
        trained_model_AI_ML (str): Path to the .pth model.
        output_path (str, optional): Where to save results.
        n_households (int, optional): Number of households to process.
                                      If None, processes the entire file.
    """

    # 1. Read the CSV file containing the full dataset.
    new_data = pd.read_csv(input_path)

    # 2. Select specific number of households (if requested)
    if n_households is not None:
        # Assuming 'Residential_ID' is the column grouping individuals into households
        if "Residential_ID" in new_data.columns:
            unique_ids = new_data["Residential_ID"].unique()

            if len(unique_ids) > n_households:
                # Select the first N household IDs
                selected_ids = unique_ids[:n_households]

                # Filter the dataframe to keep ONLY individuals in those households
                new_data = new_data[new_data["Residential_ID"].isin(selected_ids)].copy()
                print(f"Dataset filtered to {n_households} households ({len(new_data)} individuals).")
            else:
                print(f"Requested {n_households} households, but file only contains {len(unique_ids)}. Processing all.")
        else:
            print("Warning: 'Residential_ID' column not found. Cannot filter by household count. Processing all rows.")

    # Define the target output columns.
    output_columns = [
        'Family Typology',
        'Nuclear Family, Occupant Profile',
        'Nuclear Family, Typology',
        'Nuclear Family, Occupant Sequence Number'
    ]

    # 3. Call classify_new_data()
    # This function is already vectorized and will handle the filtered dataframe in one go.
    predictions = classify_new_data(trained_model_AI_ML, new_data, output_columns)

    # 4. Append the classified output columns to the dataframe.
    for col in output_columns:
        if col in predictions:
            new_data[col] = predictions[col]

    # 5. Print the predictions (Summary)
    print("\n--- Classification Summary ---")
    if len(new_data) > 20:
        print(new_data[output_columns].head(10).to_string(index=False))
        print(f"\n... and {len(new_data) - 10} more rows.")
    else:
        print(new_data[output_columns].to_string(index=False))

    # 6. Save the combined dataframe
    if output_path:
        new_data.to_csv(output_path, index=False)
        print(f"\nClassified data saved to {output_path}")

    return new_data

def classify_new_data(model_path, new_data, output_columns):
    """
    Classifies new data using the trained multi-output model.
    (This function remains largely unchanged as it is already vectorized)
    """

    # Transform new_data using the preprocessor.
    preprocessor = joblib.load(r'dataset_CENTUS/inputsFullData/preprocessor.pkl')
    new_data_processed = preprocessor.transform(new_data)

    # Define input features
    input_nominal_cols = preprocessor.input_nominal_columns_
    ordinal_cols = preprocessor.ordinal_raw_columns_ + preprocessor.household_aggregated_columns_

    # Convert to tensors
    x_nominal = torch.tensor(new_data_processed[input_nominal_cols].values, dtype=torch.long)
    x_ordinal = torch.tensor(new_data_processed[ordinal_cols].values, dtype=torch.float32)

    # Calculate embedding and output sizes
    embedding_sizes = calculate_embedding_sizes({
        col: preprocessor.nominal_encodings_[col] for col in input_nominal_cols
    })
    output_sizes = {col: len(preprocessor.nominal_encodings_[col]) for col in output_columns}

    # Define output dropout rates
    output_dropout_rate = {
        "Family Typology": 0.00,
        "Nuclear Family, Occupant Profile": 0.15,
        "Nuclear Family, Typology": 0.15,
        "Nuclear Family, Occupant Sequence Number": 0.21,
    }

    # Instantiate the model
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

    # Load weights and evaluate
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # Generate predictions
    with torch.no_grad():
        outputs = model([x_nominal[:, i] for i in range(x_nominal.shape[1])], x_ordinal)

    # Convert logits to labels
    predictions = {}
    for col in output_columns:
        predicted_indices = outputs[col].argmax(dim=1).cpu().numpy()
        reverse_mapping = {v: k for k, v in preprocessor.nominal_encodings_[col].items()}
        predictions[col] = [reverse_mapping[idx] for idx in predicted_indices]

    return predictions

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

    df_processed.columns = df_processed.columns.str.replace(' ', '_')  # Replace spaces with underscores in column names
    #print("from Temporal:", df_processed.columns)

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
                    "months_season",
                    "week_or_weekend"]

    import json, pandas as pd
    with open( r"dataset_CENTUS/inputsFullData/category_mapping.json") as f:
        cat_map = json.load(f)

    for col in one_hot_cols:  # same list you use in training
        df[col] = pd.Categorical(df[col],
                                 categories=cat_map[col]).codes

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

        # withNOBODY with Dropout
        withNOB_output = self.withNOB_dropout(rnn_out)
        withNOB_output = self.withNOB_dense(self.activation_binary(withNOB_output.contiguous().reshape(-1, withNOB_output.shape[2])))
        withNOB_output = withNOB_output.reshape(-1, 24, withNOB_output.shape[-1])  # Reshape to sequence form

        return activity_output, location_output, withNOB_output

# CLASSIFICATION PROCESS------------------------------------------------------------------------------------------------
import os
import pandas as pd
import tempfile
import shutil
def batch_classify_and_combine(input_filepath, combined_filepath):
    """
    1. Create a temp folder beside input_filepath.
    2. For each months_season (1–4) and week_or_weekend (1–3):
       • load input, overwrite the two columns
       • run AI_DLclassification saving into temp folder
    3. Read all temp outputs, concat them, save to combined_filepath.
    4. Delete the temp folder.
    """
    base_dir = os.path.dirname(input_filepath)
    #print(pd.read_csv(input_filepath).columns.tolist())
    temp_dir = tempfile.mkdtemp(prefix="tmp_classify_", dir=base_dir)

    results = []
    for month in range(1, 5):
        for week in range(1, 4):
            df = pd.read_csv(input_filepath)
            df['months_season']   = month
            df['week_or_weekend'] = week

            tmp_in = os.path.join(temp_dir, f"tmp_ms{month}_ww{week}.csv")
            df.to_csv(tmp_in, index=False)

            out_file = os.path.join(temp_dir, f"classified_ms{month}_ww{week}.csv")
            AI_DLclassification(
                input_filepath=tmp_in,
                model_type='LSTM',
                output_path=out_file
            )

            os.remove(tmp_in)
            df_out = pd.read_csv(out_file)
            df_out['months_season']   = month
            df_out['week_or_weekend'] = week
            results.append(df_out)

    combined = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(combined_filepath), exist_ok=True)
    combined.to_csv(combined_filepath, index=False)

    # clean up
    shutil.rmtree(temp_dir)
#PROCESS------------------------------------------------------------------------------------------------
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
def classify_afterTuning_withNewData(
    model,
    X_new,
    df,
    label_encoder,
    mps_device,
    csv_filepath,
    arch_type,
    output_path: None,
    mode: str = "probabilistic",
    temperature: float = 1.0,
    top_k: int = 10,
):
    """Classify new data with an already‑trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model (LSTM / Transformer).
    X_new : np.ndarray | torch.Tensor
        Shape (n_samples, seq_len, n_features).
    df : pandas.DataFrame
        Original dataframe; predictions will be appended as new cols.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encoder used for activity labels.
    mps_device : torch.device
        Compute device ("mps", "cuda", or "cpu").
    csv_filepath : str
        Path of the source CSV (only used for naming output file).
    arch_type : str
        Tag for architecture (e.g. "lstm", "transformer").
    output_path : str | None, default None
        If None, auto‑generate next to *csv_filepath*.
    mode : {"argmax", "probabilistic", "stochastic", "top_k"}, default "argmax"
        Prediction strategy.
    temperature : float, default 1.0
        Used when *mode* is "stochastic" or "top_k" (>0:  <1 sharpens, >1 flattens).
    top_k : int, default 5
        Keep only the highest‑*k* logits before sampling when *mode*="top_k". Ignored otherwise.
    """

    if output_path is None:
        filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
        output_path = f"{filename_prefix}_model_predictions_{arch_type}_{mode}.csv"

    # ---------------------------- build DataLoader ---------------------------
    num_categorical_features = 24  # first N features are int‑encoded categories
    dataset = TensorDataset(
        *[
            torch.tensor(X_new[:, :, i], dtype=torch.long)
            for i in range(num_categorical_features)
        ],
        torch.tensor(X_new[:, :, num_categorical_features:], dtype=torch.float),
    )
    loader = DataLoader(dataset, batch_size=24, shuffle=False)

    all_act, all_loc, all_nob = [], [], []

    model.eval()
    with torch.no_grad():
        for data in loader:
            features = [d.to(mps_device) for d in data]
            act_out, loc_out, nob_out = model(*features)

            # -----------------------------------------------------------------
            if mode == "argmax":
                act_pred = torch.argmax(act_out, dim=-1)
                loc_pred = torch.round(torch.sigmoid(loc_out)).long()
                nob_pred = torch.round(torch.sigmoid(nob_out)).long()

            elif mode == "probabilistic":
                act_probs = torch.softmax(act_out, dim=-1)
                loc_probs = torch.sigmoid(loc_out)
                nob_probs = torch.sigmoid(nob_out)

                act_pred = torch.distributions.Categorical(act_probs).sample()
                loc_pred = torch.distributions.Bernoulli(loc_probs).sample().long()
                nob_pred = torch.distributions.Bernoulli(nob_probs).sample().long()

            elif mode == "stochastic":
                if temperature <= 0:
                    raise ValueError("temperature must be > 0 for stochastic mode")
                act_probs = torch.softmax(act_out / temperature, dim=-1)
                loc_probs = torch.sigmoid(loc_out / temperature)
                nob_probs = torch.sigmoid(nob_out / temperature)

                act_pred = torch.multinomial(act_probs.view(-1, act_probs.size(-1)), 1).view(act_probs.size()[:-1])
                loc_pred = torch.distributions.Bernoulli(loc_probs).sample().long()
                nob_pred = torch.distributions.Bernoulli(nob_probs).sample().long()

            elif mode == "top_k":
                if top_k < 1:
                    raise ValueError("top_k must be >=1 for top_k mode")
                if temperature <= 0:
                    raise ValueError("temperature must be > 0 for top_k mode")

                # Apply temperature, then softmax
                act_logits = act_out / temperature
                act_probs_full = torch.softmax(act_logits, dim=-1)

                # Keep only top‑k probs per time‑step
                k = min(top_k, act_probs_full.size(-1))
                topk_probs, topk_idx = torch.topk(act_probs_full, k=k, dim=-1)
                # Renormalise so they sum to 1
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                # Sample from reduced distribution
                sampled_k = torch.distributions.Categorical(topk_probs).sample()
                # Map back to original class indices
                act_pred = torch.gather(topk_idx, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

                # Binary outputs: still Bernoulli sampling (no concept of top‑k)
                loc_probs = torch.sigmoid(loc_out / temperature)
                nob_probs = 1 - torch.sigmoid(nob_out / temperature)
                loc_pred = torch.distributions.Bernoulli(loc_probs).sample().long()
                nob_pred = torch.distributions.Bernoulli(nob_probs).sample().long()
            else:
                raise ValueError(
                    "Invalid mode! Choose 'argmax', 'probabilistic', 'stochastic', or 'top_k'."
                )

            # ------------------------------------------------ collect batch ----
            all_act.extend(act_pred.reshape(-1).cpu().numpy())
            all_loc.extend(loc_pred.reshape(-1).cpu().numpy())
            all_nob.extend(nob_pred.reshape(-1).cpu().numpy())

    # ------------------------------ post‑process --------------------------------
    df["Classified_Occupant_Activity"] = label_encoder.inverse_transform(all_act)
    df["Classified_Location"] = all_loc
    df["Classified_withNOBODY"] = all_nob

    for col in ("Occupant_Activity", "location", "withNOBODY"):
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path} using mode '{mode}'.")
    print(f"classify_afterTuning_{arch_type} completed.")

    return df
def AI_DLclassification(input_filepath, model_type='LSTM', output_path=None):
    import torch
    import json
    import os
    import pickle
    import pandas as pd
    df = pd.read_csv(input_filepath)
    #print("from AI_DLclassification", df.columns)
    X_new = data_preprocess(df)

    encoder_path = r"dataset_CENTUS/inputsFullData/occupant_activity_label_encoder.pkl"
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    #print("Label encoder loaded from", encoder_path)

    # MODELING---------------------------------------------------------------------------------------------------------------
    # Load best hyperparameters
    with open(f"dataset_CENTUS/inputsFullData/tus_main_EqPadHHID_RAWDATA_31_v4_best_TuningParams_LSTM.json", 'r') as infile:
        best_params = json.load(infile)
    print(best_params)

    # --- 2. Load num_features JSON (fallback to saved values) ---
    with open("dataset_CENTUS/inputsFullData/num_features.json", 'r') as f:
        num_features = json.load(f)
    print("Loaded num_features from JSON:", num_features)

    # Unpack hyperparameters
    mps_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # --- 4. Instantiate model with matching arch ---
    model = RNNsModelTuning(
        **num_features,
        output_dim_activity=best_params.get('n_classes_activity', 145),
        output_dim_location=best_params.get('n_classes_location', 1),
        output_dim_withNOB=best_params.get('n_classes_withNOB', 1),
        num_hidden_layers=best_params['num_hidden_layers'],
        hidden_units=[best_params[f'hidden_units_l{i}'] for i in range(best_params['num_hidden_layers'])],
        rnn_type='LSTM',
        embed_size=best_params['embedding_size'],
        dropout_loc=best_params.get('dropout_loc', 0.25),
        dropout_withNOB=best_params.get('dropout_withNOB', 0.1),
        dropout_embedding=best_params.get('dropout_embedding', 0.0),
        dropout_RNNs=best_params.get('dropout_RNNs', 0.0),
    ).to(mps_device)
    # --- 5. Load checkpoint safely ---
    checkpoint = torch.load(
        "dataset_CENTUS/inputsFullData/tus_main_EqPadHHID_RAWDATA_31_v4_best_model_LSTM.pth",
        map_location=mps_device
    )
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    df_original = pd.read_csv(input_filepath)
    classify_afterTuning_withNewData(model, X_new, df_original, label_encoder, mps_device, input_filepath, arch_type=model_type, output_path=output_path, mode='probabilistic')

#VISUALIZATION----------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import ast
# — before any plotting —
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman']
# if you need math text in Times as well:
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm']      = 'Times New Roman'
#-----------------------------------------------------------------------------------------------------------------------
def load_data(csv_path, demo_csv_path, header_dict=None):
    """
    Load main data, demo data, labels, and category mappings.
    Returns:
      df, demo_df, highlight_combos,
      season_map, week_map, labels_df, category_mappings
    """
    # main classification data
    df = pd.read_csv(csv_path)

    # demo data
    demo_df = pd.read_csv(demo_csv_path)
    if header_dict:
        demo_df.rename(columns=header_dict, inplace=True)

    # parse category mappings from fixed CSV path
    cat_csv = r"dataset_CENTUS/00CategoriesFORDemographics.csv"
    cat_df = pd.read_csv(cat_csv)
    category_mappings = {}
    for entry in cat_df['Category']:
        col, mapping_str = entry.split('=', 1)
        col = col.strip()
        category_mappings[col] = ast.literal_eval(mapping_str.strip())

    # mappings for season/week names
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    week_map   = {1: 'Weekdays', 2: 'Saturday', 3: 'Sunday'}

    # highlight combos based on demo
    highlight_combos = set(zip(demo_df['months_season'], demo_df['week_or_weekend']))

    # activity labels from fixed labels CSV
    base_dir = os.path.dirname(csv_path)
    labels_df = pd.read_csv(os.path.join(base_dir, '00Labels-Names-OccAct.csv'))

    # ensure location is categorical
    df['Classified_Location'] = pd.Categorical(df['Classified_Location'], categories=[0,1])
    df['Classified_withNOBODY'] = pd.Categorical(df['Classified_withNOBODY'], categories=[0,1])
    return df, demo_df, highlight_combos, season_map, week_map, labels_df, category_mappings
def plot_single_subplot(ax, subset, season_map, week_map, combo, highlight_combos):
    """Plot co‑presence and activity annotations on one Axes **without vertical
    connectors** – every hour is a horizontal segment.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for drawing.
    subset : pandas.DataFrame
        Rows for a single occupant, single (season, week) combo.
    season_map, week_map : dict
        Mappings from numeric code → name.
    combo : tuple(int, int)
        (months_season, week_or_weekend)
    highlight_combos : set[tuple(int, int)]
        Combos that should be highlighted (yellow background, gold border).
    """
    # -------------------------------------------------- styling (highlight)
    month, week = combo
    if combo in highlight_combos:
        for spine in ax.spines.values():
            spine.set_edgecolor("gold"); spine.set_linewidth(2)
        ax.set_facecolor((1, 1, 0, 0.05))
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor("dimgray"); spine.set_linewidth(1)
        ax.set_facecolor("white")

    # -------------------------------------------------- prep arrays
    subset = subset.sort_values("hourEnd_Activity")
    hours  = subset["hourEnd_Activity"].to_numpy()
    codes  = subset["Classified_Location"].cat.codes.to_numpy() * 0.5  # 0 → outside, 0.5 → inside
    colors = subset["Classified_withNOBODY"].map({1: "coral", 0: "dodgerblue"}).to_numpy()
    acts   = subset["Classified_Occupant_Activity"].to_numpy()

    # Typical hour‑to‑hour spacing (used for last segment)
    import numpy as np
    delta = np.median(np.diff(hours)) if hours.size > 1 else 1

    # -------------------------------------------------- draw horizontal segments
    # Draw one segment for *every* hour. For hours[ i ], the segment spans
    #   [hours[i], hours[i+1]]  except the last which spans [hours[-1], hours[-1] + delta].
    for i in range(len(hours)):
        x_start = hours[i]
        x_end   = hours[i + 1] if i < len(hours) - 1 else hours[i] + delta
        ax.hlines(
            y=codes[i],
            xmin=x_start,
            xmax=x_end,
            colors=colors[i],
            linewidth=4,
        )

    # -------------------------------------------------- annotate activities
    for x, y, act, code in zip(hours, codes, acts, subset["Classified_Location"].cat.codes):
        y_off, va = (y - 0.03, "top") if code == 0 else (y + 0.02, "bottom")
        ax.text(
            x +0.5,
            y_off,
            str(act),
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
            fontstyle="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )

    # -------------------------------------------------- cosmetics
    ax.set_title(f"Season: {season_map[month]}, {week_map[week]}", fontsize=18, loc="left")
    ax.set_xticks(sorted(hours))
    ax.tick_params(axis='x', labelsize=14)
    ax.set_yticks([0, 0.5])
    ax.set_yticklabels(["outside", "inside"], fontsize=18, rotation=90, va="center")
    ax.set_ylim(-0.15, 0.65)
    ax.set_xmargin(0)
    ax.grid(axis="x", linestyle="--", linewidth=0.25, color="gray")
def add_legends(fig, df, demo_df, labels_df, category_mappings):
    """Add legends: co-presence, augmentation, activity, demographics."""
    # separate legends for co-presence and augmentation
    # co-presence legend
    patches_cp = [
        mpatches.Patch(color='dodgerblue', label='Alone'),
        mpatches.Patch(color='coral',      label='Accompanied')
    ]
    leg_cp = fig.legend(handles=patches_cp,
                        loc='upper left',
                        bbox_to_anchor=(0.025,0.99),
                        ncol=1,
                        title='Co-presence')
    fig.add_artist(leg_cp)

    # augmentation legend
    patches_aug = [
        mpatches.Patch(color='gold',  label='Existing TUS schedule'),
        mpatches.Patch(color='black', label='Augmented schedule(s)')
    ]
    leg_aug = fig.legend(handles=patches_aug,
                         loc='upper left',
                         bbox_to_anchor=(0.095,0.99),
                         ncol=1,
                         title='Augmentation')
    fig.add_artist(leg_aug)

    # occupant activity legend
    acts    = sorted(df['Classified_Occupant_Activity'].unique())
    act_map = labels_df.set_index('Category Label')['Category Name'].to_dict()
    patches_act = [
        mpatches.Patch(facecolor='white', edgecolor='black',
                       label=f"{act}: {act_map.get(act,'')}" )
        for act in acts
    ]
    leg_act = fig.legend(handles=patches_act,
                         loc='upper center',
                         #bbox_to_anchor=(0.595,0.99),
                         bbox_to_anchor=(0.59, 0.99),
                         ncol=6,
                         title='Occupant Activity',
                         fontsize=15,
                         title_fontsize=15)
    fig.add_artist(leg_act)

    # demographics per occupant
    ignore = {'hourStart_Activity', 'hourEnd_Activity',
              'months_season', 'week_or_weekend'}
    demo_cols = [c for c in demo_df.columns if c not in ignore]
    patches_demo = []
    for col in demo_cols:
        mapping = category_mappings.get(col, {})
        for v in sorted(demo_df[col].dropna().unique()):
            label = mapping.get(v, str(v))
            patches_demo.append(
                mpatches.Patch(facecolor='white', edgecolor='black',
                               label=f"{col}: {label}")
            )
    fig.legend(handles=patches_demo,
               loc='lower center',
               bbox_to_anchor=(0.5,0),
               ncol=6,
               title='Demographics',
               fontsize=15,
               title_fontsize=15)
def visualize_predictions_by_combo_st9(csv_path,demo_csv_path,header_dict=None,occupant_ids=None):
    # load data
    df, demo_df, highlight_combos, season_map, week_map, labels_df, category_mappings = \
        load_data(csv_path, demo_csv_path, header_dict)

    # determine occupants
    all_occ = sorted(df['Occupant_ID_in_HH'].unique())
    if occupant_ids is None:
        occupant_ids = all_occ
    else:
        occupant_ids = [occ for occ in occupant_ids if occ in all_occ]

    # season/week combos
    combos = [(m, w)
              for m in sorted(df['months_season'].unique())
              for w in sorted(df['week_or_weekend'].unique())]

    for occ in occupant_ids:
        # subset for this occupant
        df_occ = df[df['Occupant_ID_in_HH'] == occ]
        demo_df_occ = demo_df[demo_df['Occupant_ID'] == occ]
        # compute occupant-specific highlight combos
        highlight_combos_occ = set(zip(demo_df_occ['months_season'], demo_df_occ['week_or_weekend']))

        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(24,15), sharex=True, sharey=True)
        fig.subplots_adjust(top=0.90, bottom=0.12)
        axes = axes.flatten()

        for ax, combo in zip(axes, combos):
            subset = df_occ[(df_occ['months_season']==combo[0]) & (df_occ['week_or_weekend']==combo[1])]
            plot_single_subplot(ax, subset, season_map, week_map, combo, highlight_combos_occ)

        #fig.suptitle(f"Occupant ID: {occ}", fontsize=16)
        add_legends(fig, df_occ, demo_df_occ, labels_df, category_mappings)
        plt.tight_layout(rect=[0,0.125,1,0.85])
        plt.show()
#-----------------------------------------------------------------------------------------------------------------------
def visualize_predictions_all_occupants(csv_path):
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
    output_dir = r"dataset_CENTUS/temporalDatasets"
    header_dict = {
        "Family_Typology_Simple": "SimpleFamilyType",
        "Employment status": "Employment",
        "Job Type": "JobType",
        "Economic Sector, Profession": "JobSector",
        "Family Typology": "FamilyType",
        "Full_Part_time": "Full/PartTime",
        "Permanent/fixed": "EmploymentType",
        "Education Degree": "Education",
        "Age Classes": "AgeClass",
        "Kinship Relationship": "Affiliation",
        "Mobile Phone Ownership": "PhoneOwnership",
        "Nuclear Family, Typology": "NuclearFamilyType",
        "Nuclear Family, Occupant Profile": "OccupantProfileFamily",
        "Nuclear Family, Occupant Sequence Number": "OccupantOrder",
        "Occupant_ID_in_HH": "Occupant_ID",
        "Internet Access": "InternetAccess",
        "Number Family Members": "#FamilyMembers",
        "Marital Status": "MaritalStatus",
        "Room Count": "RoomCount",
        "Car Ownership": "CarOwnership",
        "Home Ownership": "HomeOwnership",
        "House Area": "HouseArea",
    }

    # OUTPUTS ##########################################################################################################
    # https://docs.google.com/presentation/d/1l-rP260eG2kS4SEYLNSfmJEtv2A_DmmVPuwYJNKHF9o/edit#slide=id.g2d7e0a07f9f_0_406
    # 1st step
    selResfromCENSUS = r'dataset_CENTUS/selected_residential.csv'
    # 2nd step
    selResOCCfromCENSUS = "dataset_CENTUS/selected_residential_household.csv"  # New CSV to save selected household data
    # 3rd step
    alignedResOCCwithCENTUS = "dataset_CENTUS/aligned_residential_household.csv"  # New CSV to save aligned household data
    #alignedResOCCwithCENTUS = "dataset_CENTUS/test_filtered_data.csv"  # New CSV to save aligned household data
    EqPadHHID_ResOCCwithCENTUS= "dataset_CENTUS/EqPadHHID_residential_household.csv"  # New CSV to save processed household data
    # 5th step
    classfied_EqPadHHID_ResOCCwithCENTUS = "dataset_CENTUS/classified_EqPadHHID_residential_household.csv"  # New CSV to save processed household data

    # CLASSIFICATION ###################################################################################################
    # 1st & 2nd step: selecting residential data and assinging household
    #select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None)
    #select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None, residential_id=23833085) # 1-person household, selected male
    #select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None, residential_id=7280490) # 1-person household, selected female
    #select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None, residential_id=17577100) # 4-person households, married couple
    #select_RES_HH(input=census_main_merged, outRES=selResfromCENSUS, outRESOCC=selResOCCfromCENSUS, num_members=None, residential_id=23821161) # 3-person households, married couple

    # 3rd step: aligning the household data of residentials
    trained_model = r"dataset_CENTUS/inputsFullData/multiOut_bestModel.pth"
    main_classify_new_data(
        input_path=census_main_merged, # selResOCCfromCENSUS
        trained_model_AI_ML=trained_model,
        output_path=alignedResOCCwithCENTUS,
        n_households=1000  # <--- NEW INPUT
    )

    # 4th step: converting to temporal data with equalize sequences and padding, 60-minute intervals
    temporalConversion(input_path=alignedResOCCwithCENTUS, output_path=EqPadHHID_ResOCCwithCENTUS)

    # ONLY CLASSIFICATION-----------------------------------------------------------------------------------------------
    # 5th step-A: temporal data classification
    #AI_DLclassification(input_filepath=EqPadHHID_ResOCCwithCENTUS, model_type='LSTM', output_path=classfied_EqPadHHID_ResOCCwithCENTUS)
    # 5th step-Visualization: temporal data classification
    #visualize_predictions_all_occupants(csv_path=classfied_EqPadHHID_ResOCCwithCENTUS)

    # ONLY AUGMENTATION-------------------------------------------------------------------------------------------------
    # 5th step-B: temporal data augmentation
    batch_classify_and_combine(EqPadHHID_ResOCCwithCENTUS, combined_filepath=classfied_EqPadHHID_ResOCCwithCENTUS)
    #visualize_predictions_by_combo_st9(csv_path=classfied_EqPadHHID_ResOCCwithCENTUS, demo_csv_path=EqPadHHID_ResOCCwithCENTUS, header_dict=header_dict)