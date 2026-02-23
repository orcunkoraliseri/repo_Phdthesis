
#PREPROCESS-------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle
# Sub-functions for modular processing
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
    #data['Mobile_Phone_to_Total_Devices'] = data.groupby('Household_ID')['Mobile Phone Ownership'].transform('sum') / (
    #    data.groupby('Household_ID')['Mobile Phone Ownership'].transform('sum') + data.groupby('Household_ID')['Landline Ownership'].transform('sum'))
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
                                       #'Landline Ownership',
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
#MODELING---------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
# Helper function to calculate embedding sizes
def calculate_embedding_sizes(nominal_encodings):
    embedding_sizes = []
    for col, mapping in nominal_encodings.items():
        num_categories = len(mapping)
        embedding_dim = min(50, num_categories // 2)
        embedding_sizes.append((num_categories, embedding_dim))
    return embedding_sizes
# Updated Neural Network for Multi-Output Classification
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
#TRAINING---------------------------------------------------------------------------------------------------------------
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
def train_multi_output_model(
    model, train_loader, val_loader, criterions, optimizer,
    num_epochs=10, early_stop_threshold=0.95, save_path="multiOut_bestModel.pth", output_columns=None):
    if output_columns is None:
        output_columns = list(criterions.keys())  # Default to criterions keys if not provided

    history = {
        'train_loss': {output: [] for output in criterions.keys()},
        'val_loss': {output: [] for output in criterions.keys()},
        'train_acc': {output: [] for output in criterions.keys()},
        'val_acc': {output: [] for output in criterions.keys()},
        'train_f1': {output: [] for output in criterions.keys()},
        'val_f1': {output: [] for output in criterions.keys()}
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_per_output = {output: 0 for output in criterions.keys()}
        total_train_samples = 0
        train_correct = {output: 0 for output in criterions.keys()}
        train_true = {output: [] for output in criterions.keys()}
        train_pred = {output: [] for output in criterions.keys()}

        for x_nominal, x_ordinal, y in train_loader:
            optimizer.zero_grad()
            outputs = model([x_nominal[:, i] for i in range(x_nominal.shape[1])], x_ordinal)
            total_loss = 0
            for output_name, criterion in criterions.items():
                loss = criterion(outputs[output_name], y[output_name].long())
                total_loss += loss
                train_loss_per_output[output_name] += loss.item() * x_nominal.size(0)
                predictions = outputs[output_name].argmax(dim=1)
                train_correct[output_name] += (predictions == y[output_name]).sum().item()
                train_true[output_name].extend(y[output_name].cpu().numpy())
                train_pred[output_name].extend(predictions.cpu().numpy())

            total_train_samples += x_nominal.size(0)
            total_loss.backward()
            optimizer.step()

        for output_name in criterions.keys():
            train_loss_per_output[output_name] /= total_train_samples
        train_accuracy = {output: train_correct[output] / total_train_samples for output in criterions.keys()}
        train_f1 = {output: f1_score(train_true[output], train_pred[output], average='weighted') for output in criterions.keys()}

        # Validation phase
        model.eval()
        val_loss_per_output = {output: 0 for output in criterions.keys()}
        total_val_samples = 0
        val_correct = {output: 0 for output in criterions.keys()}
        val_true = {output: [] for output in criterions.keys()}
        val_pred = {output: [] for output in criterions.keys()}

        with torch.no_grad():
            for x_nominal, x_ordinal, y in val_loader:
                outputs = model([x_nominal[:, i] for i in range(x_nominal.shape[1])], x_ordinal)
                for output_name, criterion in criterions.items():
                    loss = criterion(outputs[output_name], y[output_name].long())
                    val_loss_per_output[output_name] += loss.item() * x_nominal.size(0)
                    predictions = outputs[output_name].argmax(dim=1)
                    val_correct[output_name] += (predictions == y[output_name]).sum().item()
                    val_true[output_name].extend(y[output_name].cpu().numpy())
                    val_pred[output_name].extend(predictions.cpu().numpy())
                total_val_samples += x_nominal.size(0)

        for output_name in criterions.keys():
            val_loss_per_output[output_name] /= total_val_samples
        val_accuracy = {output: val_correct[output] / total_val_samples for output in criterions.keys()}
        val_f1 = {output: f1_score(val_true[output], val_pred[output], average='weighted') for output in criterions.keys()}

        # Record history
        for output_name in criterions.keys():
            history['train_loss'][output_name].append(train_loss_per_output[output_name])
            history['val_loss'][output_name].append(val_loss_per_output[output_name])
            history['train_acc'][output_name].append(train_accuracy[output_name])
            history['val_acc'][output_name].append(val_accuracy[output_name])
            history['train_f1'][output_name].append(train_f1[output_name])
            history['val_f1'][output_name].append(val_f1[output_name])

        print(f"Epoch {epoch+1}/{num_epochs}")
        for output_name in criterions.keys():
            print(f" - {output_name} | Train Loss: {train_loss_per_output[output_name]:.4f}, Train Acc: {train_accuracy[output_name]:.4f}, Train F1: {train_f1[output_name]:.4f}, "
                  f"Val Loss: {val_loss_per_output[output_name]:.4f}, Val Acc: {val_accuracy[output_name]:.4f}, Val F1: {val_f1[output_name]:.4f}")

        # Early stopping criteria
        if all(acc >= early_stop_threshold for acc in train_accuracy.values()) and all(acc >= early_stop_threshold for acc in val_accuracy.values()):
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            break

    # Plot training history
    plot_training_history(history, output_columns)
#MAIN-------------------------------------------------------------------------------------------------------------------
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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
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

import dill
import joblib
import pickle
from sklearn.model_selection import train_test_split
import torch
def ML(df, output_columns, base_nominal_columns,
       learning_rate=0.0004, num_epochs=100, batch_size=8, ordinal_hidden_size=24, fc_hidden_sizes=(24,),
       dropout_rate=0.0, use_batch_norm=True):
    # Define file paths for the preprocessed data and the preprocessor.
    preprocessed_data_path = "preprocessed_data_3rdSTEP_AI_ML.pkl"
    preprocessor_path = "preprocessor_3rdSTEP_AI_ML.pkl" #this is crucial for after training classfication process

    # Check if both files exist. If so, load them; otherwise, run preprocessing.
    if os.path.exists(preprocessed_data_path) and os.path.exists(preprocessor_path):
        print("Loading preprocessed data and preprocessor from disk...")
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = dill.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = dill.load(f)
        # Extract preprocessed splits (assumed to be saved as a dictionary).
        train_processed = preprocessed_data["train"]
        val_processed = preprocessed_data["val"]
        test_processed = preprocessed_data["test"]
    else:
        print("Running preprocessing and saving to disk...")
        # Split raw data into training, validation, and test sets.
        train_raw, temp_raw = train_test_split(df, test_size=0.5, random_state=42)
        val_raw, test_raw = train_test_split(temp_raw, test_size=0.25, random_state=42)

        # Instantiate and fit the custom preprocessor on training data.
        preprocessor = CustomPreprocessor(output_columns=output_columns, base_nominal_columns=base_nominal_columns)
        train_processed = preprocessor.fit_transform(train_raw)
        # Transform validation and test sets.
        val_processed = preprocessor.transform(val_raw)
        test_processed = preprocessor.transform(test_raw)

        # Save the fitted preprocessor using dill.
        try:
            joblib.dump(preprocessor, preprocessor_path, protocol=2)
            print("Preprocessor saved as '{}' using joblib.".format(preprocessor_path))
        except pickle.PicklingError as e:
            print("Joblib failed to pickle the preprocessor. Falling back to dill.")
            with open(preprocessor_path, 'wb') as f:
                dill.dump(preprocessor, f, protocol=2)
            print("Preprocessor saved as '{}' using dill.".format(preprocessor_path))

        # Save the preprocessed data as a dictionary.
        preprocessed_data = {
            "train": train_processed,
            "val": val_processed,
            "test": test_processed
        }
        with open(preprocessed_data_path, 'wb') as f:
            dill.dump(preprocessed_data, f)
        print("Preprocessed data saved as '{}'.".format(preprocessed_data_path))

    # Use input_columns from the preprocessor (which excludes outputs).
    input_columns = preprocessor.input_columns_
    print("Input Columns:", input_columns)

    # Split features and targets.
    X_train, y_train = train_processed[input_columns], train_processed[output_columns]
    X_val, y_val = val_processed[input_columns], val_processed[output_columns]
    X_test, y_test = test_processed[input_columns], test_processed[output_columns]

    # Calculate embedding sizes using nominal encodings.
    embedding_sizes = calculate_embedding_sizes({
        col: preprocessor.nominal_encodings_[col] for col in preprocessor.input_nominal_columns_
    })

    # Define ordinal columns (used in the model).
    ordinalCols = preprocessor.ordinal_raw_columns_ + preprocessor.household_aggregated_columns_

    # Create datasets using the input features.
    train_dataset = CustomDataset(X_train, y_train, preprocessor.input_nominal_columns_, ordinalCols, output_columns)
    val_dataset = CustomDataset(X_val, y_val, preprocessor.input_nominal_columns_, ordinalCols, output_columns)
    test_dataset = CustomDataset(X_test, y_test, preprocessor.input_nominal_columns_, ordinalCols, output_columns)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine output sizes.
    output_sizes = {col: len(y_train[col].unique()) for col in output_columns}

    output_dropout_rate = {
        "Family Typology": 0.0025,  # 0.0025, default value
        "Nuclear Family, Occupant Profile": 0.05,
        "Nuclear Family, Typology": 0.05,
        "Nuclear Family, Occupant Sequence Number": 0.05,
    }

    # Initialize the model.
    model = MultiOutputNeuralNetwork(
        embedding_sizes=embedding_sizes,
        num_ordinal_features=len(ordinalCols),
        output_sizes=output_sizes,
        ordinal_hidden_size=ordinal_hidden_size,
        fc_hidden_sizes=fc_hidden_sizes,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        output_dropout_rate=output_dropout_rate,
    )
    print(model)

    # Define optimizer and loss functions.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterions = {col: nn.CrossEntropyLoss(reduction='mean') for col in output_columns}

    # Train the model.
    train_multi_output_model(model, train_loader, val_loader, criterions, optimizer, num_epochs)

    return model, train_loader, val_loader, test_loader

#CLASSIFICATION---------------------------------------------------------------------------------------------------------
import torch
import torch
def classify_new_data(model_path, new_data, output_columns, base_nominal_columns):
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
    # Instantiate the custom preprocessor with the same settings used in training.
    #preprocessor = CustomPreprocessor(output_columns=output_columns, base_nominal_columns=base_nominal_columns)

    # Transform new_data using the preprocessor.
    # (Assumes that the preprocessor has been fitted on training data and that its state
    #  is available; otherwise, you should load the fitted preprocessor.)
    preprocessor = joblib.load('preprocessor_3rdSTEP_AI_ML.pkl')
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
    model.load_state_dict(torch.load(model_path))
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
from sklearn.metrics import accuracy_score
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
"""
def test_classify_new_data():
    import pandas as pd
    # Read the CSV file containing the full dataset.
    data = pd.read_csv(r'dataset_TUS_individual/TUS_indiv_03_final_moreCol.csv')

    # Select a sample from the data (for example, the first 10 rows).
    #new_data = data.head(10)
    import random
    num_rows = 30
    # Ensure that there are enough rows in the dataset.
    if len(data) < num_rows:
        raise ValueError("The dataset does not contain enough rows.")
    # Choose a random starting index such that there are 'num_rows' consecutive rows.
    start_index = random.randint(0, len(data) - num_rows)
    # Select the consecutive rows.
    new_data = data.iloc[start_index: start_index + num_rows]

    # You can continue with your test using new_data
    print(f"Selected rows from {start_index} to {start_index + num_rows - 1}")

    # Define the target output columns.
    output_columns = [
        'Family Typology',
        'Nuclear Family, Occupant Profile',
        'Nuclear Family, Typology',
        'Nuclear Family, Occupant Sequence Number'
    ]

    # Define the base nominal columns used during training.
    base_nominal_columns = [
        'Region', 'Employment status', 'Gender', 'Marital Status', 'Kinship Relationship',
        'Citizenship', "Job Type", "Economic Sector, Profession", "Home Ownership",
        "Family_Typology_Simple", 'Internet Access', 'Car Ownership', 
        #"Landline Ownership",
        'Mobile Phone Ownership'
    ]

    # Call classify_new_data() using the saved model weights (adjust the path as needed).
    predictions = classify_new_data("multiOut_bestModel.pth", new_data, output_columns, base_nominal_columns)

    # Print the predictions.
    print("Predictions:")
    for col, preds in predictions.items():
        print(f"{col}: {preds}")

    # And assuming new_data contains the ground truth for the output columns,
    # compute the accuracy scores:
    accuracy_dict = compute_accuracy(predictions, new_data, output_columns)

    # Print accuracy for each target column:
    print("Accuracy:")
    for col, acc in accuracy_dict.items():
        print(f"{col}: {acc * 100:.2f}%")
"""
def main_classify_new_data():
    """
    Main function for classifying new data and evaluating performance.

    This function:
      - Reads the CSV dataset.
      - Randomly selects `num_rows` consecutive rows.
      - Defines the target output columns and base nominal columns.
      - Calls classify_new_data() with the saved model weights.
      - Prints the predictions.
      - Computes and prints the accuracy for each output column.

    Parameters:
      num_rows (int): Number of consecutive rows to randomly select from the dataset.
    """
    import pandas as pd
    import random

    # Read the CSV file containing the full dataset.
    data = pd.read_csv(r'dataset_TUS_individual/TUS_indiv_03_final_moreCol.csv')

    # currently, this belongs to TUS, but it will change with CENSUS dataset
    num_rows = 30
    # Ensure that there are enough rows in the dataset.
    if len(data) < num_rows:
        raise ValueError("The dataset does not contain enough rows.")

    # Choose a random starting index such that there are `num_rows` consecutive rows.
    start_index = random.randint(0, len(data) - num_rows)
    # Select the consecutive rows.
    new_data = data.iloc[start_index: start_index + num_rows]
    print(f"Selected rows from {start_index} to {start_index + num_rows - 1}")

    # Define the target output columns.
    output_columns = [
        'Family Typology',
        'Nuclear Family, Occupant Profile',
        'Nuclear Family, Typology',
        'Nuclear Family, Occupant Sequence Number'
    ]

    # Define the base nominal columns used during training.
    base_nominal_columns = [
        'Region', 'Employment status', 'Gender', 'Marital Status', 'Kinship Relationship',
        'Citizenship', "Job Type", "Economic Sector, Profession", "Home Ownership",
        "Family_Typology_Simple", 'Internet Access', 'Car Ownership',
        #"Landline Ownership",
        'Mobile Phone Ownership'
    ]

    # Call classify_new_data() using the saved model weights.
    predictions = classify_new_data("multiOut_bestModel.pth", new_data, output_columns, base_nominal_columns)

    # Print the predictions.
    print("Predictions:")
    for col, preds in predictions.items():
        print(f"{col}: {preds}")

    # Compute accuracy (assuming new_data contains the ground truth for the output columns).
    accuracy_dict = compute_accuracy(predictions, new_data, output_columns)

    # Print accuracy for each target column.
    print("Accuracy:")
    for col, acc in accuracy_dict.items():
        print(f"{col}: {acc * 100:.2f}%")

#PLOTTING --------------------------------------------------------------------------------------------------------------
def plot_training_history(history, output_columns):
    epochs = range(1, len(list(history['train_loss'].values())[0]) + 1)

    # Plot Loss
    plt.figure(figsize=(7, 10))
    plt.suptitle("Loss Over Epochs", fontsize=16)
    for i, output_name in enumerate(output_columns):
        plt.subplot(len(output_columns), 1, i + 1)
        plt.plot(epochs, history['train_loss'][output_name], label='Train Loss')
        plt.plot(epochs, history['val_loss'][output_name], label='Validation Loss')
        plt.title(f"{output_name} Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(7, 10))
    plt.suptitle("Accuracy Over Epochs", fontsize=16)
    for i, output_name in enumerate(output_columns):
        plt.subplot(len(output_columns), 1, i + 1)
        plt.plot(epochs, history['train_acc'][output_name], label='Train Accuracy')
        plt.plot(epochs, history['val_acc'][output_name], label='Validation Accuracy')
        plt.title(f"{output_name} Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot F1-Score
    plt.figure(figsize=(7, 10))
    plt.suptitle("F1-Score Over Epochs", fontsize=16)
    for i, output_name in enumerate(output_columns):
        plt.subplot(len(output_columns), 1, i + 1)
        plt.plot(epochs, history['train_f1'][output_name], label='Train F1 Score')
        plt.plot(epochs, history['val_f1'][output_name], label='Validation F1 Score')
        plt.title(f"{output_name} F1-Score")
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score')
        plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#PERFORMANCE ANALYSIS --------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score
import os
import shutil
import matplotlib.pyplot as plt
def compute_multi_output_feature_importance(model, dataset, nominal_columns, ordinal_columns, metric=accuracy_score,
                                            save_folder="feature_importance_plots"):
    """
    Computes permutation feature importance for the multi-output model using the dataset
    and saves plots in a specified folder.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Folder '{save_folder}' created.")
    else:
        # Check if the folder is empty and clear it
        if os.listdir(save_folder):
            print(f"Folder '{save_folder}' is not empty. Clearing contents.")
            shutil.rmtree(save_folder)
            os.makedirs(save_folder)

    model.eval()  # Ensure the model is in evaluation mode
    nominal_data, ordinal_data, y = zip(*[dataset[i] for i in range(len(dataset))])
    nominal_data = torch.stack(nominal_data)
    ordinal_data = torch.stack(ordinal_data)

    # Convert y to a dictionary for multi-output
    y_dict = {output_name: torch.tensor([y[i][output_name] for i in range(len(y))]) for output_name in
              model.output_layers.keys()}

    feature_importances_per_output = {}

    # Iterate over each output head
    for output_name in model.output_layers.keys():
        print(f"\nCalculating feature importance for '{output_name}'")

        # Original model predictions for the specific output
        with torch.no_grad():
            original_preds = model(
                [nominal_data[:, i] for i in range(nominal_data.shape[1])], ordinal_data
            )[output_name].argmax(dim=1).cpu().numpy()
        original_score = metric(y_dict[output_name].numpy(), original_preds)

        feature_importances = {}

        # Evaluate importance for nominal and ordinal features
        all_features = {
            "nominal": (range(nominal_data.shape[1]), nominal_columns),
            "ordinal": (range(ordinal_data.shape[1]), ordinal_columns),
        }

        for feature_type, (indices, column_names) in all_features.items():
            for idx, column_name in zip(indices, column_names):
                # Shuffle the feature column
                if feature_type == "nominal":
                    shuffled_nominal_data = nominal_data.clone()
                    shuffled_nominal_data[:, idx] = shuffled_nominal_data[:, idx][torch.randperm(nominal_data.size(0))]
                    shuffled_preds = model(
                        [shuffled_nominal_data[:, i] for i in range(shuffled_nominal_data.shape[1])], ordinal_data
                    )[output_name].argmax(dim=1).cpu().numpy()
                elif feature_type == "ordinal":
                    shuffled_ordinal_data = ordinal_data.clone()
                    shuffled_ordinal_data[:, idx] = shuffled_ordinal_data[:, idx][torch.randperm(ordinal_data.size(0))]
                    shuffled_preds = model(
                        [nominal_data[:, i] for i in range(nominal_data.shape[1])], shuffled_ordinal_data
                    )[output_name].argmax(dim=1).cpu().numpy()

                # Calculate drop in performance
                shuffled_score = metric(y_dict[output_name].numpy(), shuffled_preds)
                feature_importances[column_name] = original_score - shuffled_score

        # Store feature importances for this output
        feature_importances_per_output[output_name] = feature_importances

        # Sort and plot
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_importances)

        plt.figure(figsize=(12, 8))
        plt.barh(features, scores)
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.xlabel("Drop in Performance", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.title(f"Feature Importances for '{output_name}'", fontsize=16)
        plt.yticks(fontsize=8)
        plt.gca().invert_yaxis()  # Invert so the most important features are at the top
        plt.tight_layout()

        # Save plot to folder
        plot_filename = os.path.join(save_folder, f"{output_name}_feature_importance.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to avoid memory issues
        print(f"Saved plot for '{output_name}' to '{plot_filename}'")

    print(f"All plots saved to '{save_folder}'.")
    return feature_importances_per_output
if __name__ == '__main__':
    import pandas as pd
    from preProcessing_Func import analysis_func as dfppaf
    tus_subset_final_moreCol = r'dataset_TUS_individual/TUS_indiv_03_final_moreCol.csv'
    data = pd.read_csv(tus_subset_final_moreCol)
    #print(data.columns)
    #outPerformance:https://docs.google.com/spreadsheets/d/1L2-GIfeopAzJjyQxa5Qw3gZDzSMihs1kyyyR91WvdVQ/edit?gid=0#gid=0
    output_columns = ['Family Typology',  'Nuclear Family, Occupant Profile', 'Nuclear Family, Typology',
                      'Nuclear Family, Occupant Sequence Number']

    base_nominal_columns = [ 'Region', 'Employment status', 'Gender', 'Marital Status', 'Kinship Relationship',
                             'Citizenship', "Job Type", "Economic Sector, Profession", "Home Ownership",
                             "Family_Typology_Simple", 'Internet Access', 'Car Ownership',
                             #"Landline Ownership",
                             'Mobile Phone Ownership',]

    # TRAINING #########################################################################################################
    model, train_loader, val_loader, test_loader = ML(data, output_columns, base_nominal_columns)
    # CLASSIFICATION ###################################################################################################
    trained_model = "multiOut_bestModel.pth"
    #main_classify_new_data()
    # ANALYSIS #########################################################################################################
    ID_DROP = ["Household_ID"]
    non_visual = False
    dfppaf.analysis(input_path=tus_subset_final_moreCol, unique=non_visual, uniqueIDcolstoDrop=ID_DROP)