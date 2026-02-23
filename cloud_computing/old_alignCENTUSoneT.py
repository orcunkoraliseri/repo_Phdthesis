
#PREPROCESS-------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import pickle
def preprocess_and_save(df, outCol, save_path="preprocessed_data_singleT.pkl"):
    if os.path.exists(save_path):
        print("Loading preprocessed data from disk...")
        with open(save_path, 'rb') as file:
            preprocessed_data = pickle.load(file)
    else:
        print("Running preprocessing and saving to disk...")
        preprocessed_data = preprocess_data(df, outCol)
        with open(save_path, 'wb') as file:
            pickle.dump(preprocessed_data, file)
    return preprocessed_data
def preprocess_data(data, output_column):
    # FEATURE ENGINEERING
    # Drop Household_ID temporarily (it will be used for aggregation)
    household_data = data[['Household_ID', 'Occupant_ID_in_HH', 'Age Classes', 'Education Degree', "Room Count",
                           "Full_Part_time",'Mobile Phone Ownership','Marital Status',]].copy()

    # Create household-level aggregated features
    household_aggregated = household_data.groupby('Household_ID').agg({
        'Age Classes': ['mean', 'std'],
        'Education Degree': ['mean', 'std'],
        "Full_Part_time": ['mean', 'std'],
        "Mobile Phone Ownership": ['mean', 'std'],
        "Marital Status": ['mean', 'std'],
    })

    # Rename columns for clarity
    household_aggregated.columns = [
        'Avg_Age_Classes', 'Std_Age_Classes',
        'Avg_Education_Degree', 'Std_Education_Degree',
        'Avg_Full_Part_time', 'Std_Full_Part_time',
        'Avg_Mobile_Phone', 'Std_Mobile_Phone',
        'Avg_Marital_Status', 'Std_Marital_Status',
    ]
    household_aggregated.reset_index(inplace=True)

    # Replace NaN values in standard deviation columns with 0
    household_aggregated.fillna(0, inplace=True)

    # Merge aggregated features back into the dataset
    data = data.merge(household_aggregated, on='Household_ID', how='left')

    # ADVANCED FEATURE ENGINEERING: Interaction Features
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

    # ADVANCED FEATURE ENGINEERING: Normalized ratios
    data['Age_Diversity'] = data['Std_Age_Classes'] / data['Number Family Members']
    data['Room_Per_Member'] = data['Room Count'] / data['Number Family Members']
    data['Education_Per_Member'] = data['Avg_Education_Degree'] / data['Number Family Members']
    data['Occupancy_Density'] = data['Number Family Members'] / data['Room Count']
    data['Income_Per_Member'] = data.groupby('Household_ID')['Main Income Source'].transform('mean') / data['Number Family Members']
    data['Kinship_Diversity'] = data.groupby('Household_ID')['Kinship Relationship'].transform('nunique') / data['Number Family Members']
    #data['Dependency_Ratio'] = data.groupby('Household_ID').apply(calculate_dependency_ratio).reset_index(drop=True)
    data['PartTime_FullTime_Ratio'] = data.groupby('Household_ID')['Full_Part_time'].transform(lambda x: (x == 1).sum() / (x == 2).sum() if (x == 2).sum() != 0 else 0)
    #data['Employment_Intensity'] = data.groupby('Household_ID')['Full_Part_time'].transform(lambda x: (x == 2).sum() / ((x == 1).sum() + (x == 2).sum()))
    data['Employment_Intensity'] = data.groupby('Household_ID')['Full_Part_time'].transform(lambda x: (x == 2).sum() / ((x == 1).sum() + (x == 2).sum()) if ((x == 1).sum() + (x == 2).sum()) != 0 else 0)
    data['FullTime_Per_Member'] = data.groupby('Household_ID')['Full_Part_time'].transform(lambda x: (x == 2).sum() / len(x))

    #ADVANCED FEATURE ENGINEERING: Household Composition Features
    data['Generational_Diversity'] = data.groupby('Household_ID')['Age Classes'].transform('nunique')
    data['Gender_Diversity'] = data.groupby('Household_ID')['Gender'].transform(lambda x: (x == 1).sum() / (x == 2).sum() if (x == 2).sum() != 0 else float('inf'))
    data['Role_Concentration'] = data.groupby('Household_ID')['Kinship Relationship'].transform(lambda x: x.value_counts(normalize=True).iloc[0])
    data['Marital_Diversity'] = data.groupby('Household_ID')['Marital Status'].transform('nunique')

    # Drop Household_ID after aggregation
    data = data.drop(columns=['Household_ID'])

    # PREPROCESSING
    ordinal_raw_columns = ['Occupant_ID_in_HH', 'Number Family Members', 'Age Classes', 'Education Degree', "Room Count",]

    # Define household-aggregated columns
    household_aggregated_columns =  [
        'Avg_Age_Classes', 'Std_Age_Classes',
        'Avg_Education_Degree', 'Std_Education_Degree',
        'Avg_Full_Part_time', 'Std_Full_Part_time',
        'Avg_Mobile_Phone', 'Std_Mobile_Phone',
        'Avg_Marital_Status', 'Std_Marital_Status',
    ]

    # Combine raw and aggregated columns
    nominal_columns = [
        output_column,  # Dynamically include the output column
        'Region', 'Employment status', 'Gender', 'Marital Status', 'Kinship Relationship', 'Citizenship', "Job Type",
        "Economic Sector, Profession", "Home Ownership", "Family_Typology_Simple", 'Internet Access',  'Car Ownership',
        "Landline Ownership", 'Mobile Phone Ownership', 'Region_Employment', 'Kinship_Age', 'Job_Economic_Sector', 'Region_Family_Typology',
        'Employment_HH_Size', 'Education_Job_Type', "Gender_Kinship", "Citizenship_Employment", "Age_Region",
        "Room_Ownership", 'Age_Diversity', 'Room_Per_Member', 'Education_Per_Member', 'Occupancy_Density',
        'Income_Per_Member', 'Kinship_Diversity', "Generational_Diversity", "Gender_Diversity", "Role_Concentration",
        "Marital_Diversity", "FullPart_Gender", 'FullPart_AgeClass','FullPart_Region', 'PartTime_FullTime_Ratio',
        'Employment_Intensity', 'FullTime_Per_Member',
    ]

    # Ordinal Encoding for raw ordinal columns
    ordinal_encoder = OrdinalEncoder()
    data[ordinal_raw_columns] = ordinal_encoder.fit_transform(data[ordinal_raw_columns])

    # Apply scaling to all ordinal columns
    scaler = RobustScaler()
    data[ordinal_raw_columns + household_aggregated_columns] = scaler.fit_transform(data[ordinal_raw_columns + household_aggregated_columns])

    # Create consistent mappings for nominal columns
    # Integer Encoding for nominal columns (for embedding)
    nominal_encodings = {}
    for col in nominal_columns:
        unique_values = sorted(data[col].unique())  # Ensure consistent ordering
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        data[col] = data[col].map(mapping)  # Apply the mapping
        nominal_encodings[col] = mapping

    #print("Columns after preprocessing:", data.columns)
    return data, nominal_encodings, nominal_columns, ordinal_raw_columns, household_aggregated_columns
def split_data(data, input_columns, output_columns, test_size=0.2, validation_size=0.1, random_state=42):
    # Split into inputs and outputs
    X = data[input_columns]
    y = data[output_columns]

    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Further split the training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test)
    }

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
class NeuralNetwork(nn.Module):
    def __init__(
        self,
        embedding_sizes,
        num_ordinal_features,
        output_size,
        ordinal_hidden_size=64,
        fc_hidden_sizes=(128, 64),
        dropout_rate=0.3,
        use_batch_norm=True,
    ):
        super(NeuralNetwork, self).__init__()

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

        # Output layer for single-task learning
        self.output_layer = nn.Linear(fc_hidden_sizes[-1], output_size)

    def forward(self, x_nominal, x_ordinal):
        # Pass nominal inputs through embeddings
        embedded = [embedding(x) for embedding, x in zip(self.embeddings, x_nominal)]
        embedded = torch.cat(embedded, dim=1)

        # Pass ordinal inputs through dense layer
        ordinal_out = self.ordinal_layer(x_ordinal)

        # Combine embedded and ordinal features
        combined = torch.cat([embedded, ordinal_out], dim=1)
        fc_out = self.fc(combined)

        # Compute output
        output = self.output_layer(fc_out)
        return output

#TRAINING---------------------------------------------------------------------------------------------------------------
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader, Dataset
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, early_stop_threshold=0.96, save_path="singleOut_bestModel.pth"):
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        total_train_samples = 0
        train_correct = 0
        train_true = []
        train_pred = []

        for x_nominal, x_ordinal, y in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model([x_nominal[:, i] for i in range(x_nominal.shape[1])], x_ordinal)

            # Compute loss
            loss = criterion(output, y.long())
            train_loss += loss.item() * x_nominal.size(0)
            total_train_samples += x_nominal.size(0)

            # Compute predictions
            predictions = output.argmax(dim=1)
            train_correct += (predictions == y).sum().item()
            train_true.extend(y.cpu().numpy())
            train_pred.extend(predictions.cpu().numpy())

            # Backward pass
            loss.backward()
            optimizer.step()

        # Normalize training loss and calculate accuracy
        train_loss /= total_train_samples
        train_accuracy = train_correct / total_train_samples
        train_f1 = f1_score(train_true, train_pred, average='weighted')

        # Validation phase
        model.eval()
        val_loss = 0
        total_val_samples = 0
        val_correct = 0
        val_true = []
        val_pred = []

        with torch.no_grad():
            for x_nominal, x_ordinal, y in val_loader:
                output = model([x_nominal[:, i] for i in range(x_nominal.shape[1])], x_ordinal)
                loss = criterion(output, y.long())
                val_loss += loss.item() * x_nominal.size(0)
                total_val_samples += x_nominal.size(0)

                # Compute predictions
                predictions = output.argmax(dim=1)
                val_correct += (predictions == y).sum().item()
                val_true.extend(y.cpu().numpy())
                val_pred.extend(predictions.cpu().numpy())

        # Normalize validation loss and calculate accuracy
        val_loss /= total_val_samples
        val_accuracy = val_correct / total_val_samples
        val_f1 = f1_score(val_true, val_pred, average='weighted')

        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping criteria
        best_train_accuracy = max(best_train_accuracy, train_accuracy)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        if train_accuracy >= early_stop_threshold and val_accuracy >= early_stop_threshold:
            print(f"Early stopping triggered at epoch {epoch+1}. "
                  f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}.")
            # Save the trained model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            break

#MAIN-------------------------------------------------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, X, y, nominal_columns, ordinal_columns, output_columns):
        self.nominal_data = X[nominal_columns].values
        self.ordinal_data = X[ordinal_columns].values
        self.targets = y[output_columns].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x_nominal = torch.tensor(self.nominal_data[idx], dtype=torch.long)  # Long for embeddings
        x_ordinal = torch.tensor(self.ordinal_data[idx], dtype=torch.float32)
        # Flatten the target to make it 1D
        y = torch.tensor(self.targets[idx], dtype=torch.long).squeeze()  # Use `.squeeze()` to remove extra dimensions
        return x_nominal, x_ordinal, y
def ML(df, outCol,
        learning_rate=0.001, num_epochs=20, batch_size=64, ordinal_hidden_size=64, fc_hidden_sizes=(128, 64),
       dropout_rate=0.3, use_batch_norm=True, save_path="preprocessed_data.pkl"):
    # Load preprocessed data or preprocess if not available
    preprocessed_data, nominal_encodings, nominalCols, ordinalRawCols, aggregated_columns = preprocess_and_save(df, outCol, save_path)

    if output_column in nominalCols:
        nominalCols.remove(output_column)

    # Ensure household aggregated columns are included in input_columns
    input_columns = nominalCols + ordinalRawCols \
                    + aggregated_columns
    print("input_columns:", input_columns)

    # Split the data
    splits = split_data(preprocessed_data, input_columns, [output_column], test_size=0.2, validation_size=0.1,)

    # Access splits
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    # Calculate dynamic embedding sizes
    embedding_sizes = calculate_embedding_sizes(
        {col: nominal_encodings[col] for col in nominalCols})

    # Define nominal and ordinal columns
    ordinalCols = ordinalRawCols \
                  + aggregated_columns

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, nominalCols, ordinalCols, [outCol])
    val_dataset = CustomDataset(X_val, y_val, nominalCols, ordinalCols, [outCol])
    test_dataset = CustomDataset(X_test, y_test, nominalCols, ordinalCols, [outCol])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    output_size = len(y_train[output_column].unique())

    # Initialize the model with hyperparameters
    model = NeuralNetwork(
        embedding_sizes=embedding_sizes,
        num_ordinal_features=len(ordinalCols),
        output_size=output_size,
        ordinal_hidden_size=ordinal_hidden_size,
        fc_hidden_sizes=fc_hidden_sizes,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,)
    print(model)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    #criterion = nn.BCEWithLogitsLoss(reduction='mean')  # Binary

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Call the function with appropriate arguments
    #compute_feature_importance(model, test_dataset, nominalCols, ordinalCols + aggregated_columns)
    return model, train_loader, val_loader, test_loader

#PERFORMANCE ANALYSIS --------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score
def compute_feature_importance(model, dataset, nominal_columns, ordinal_columns, metric=accuracy_score):
    """
    Computes permutation feature importance for the model using the dataset.
    """
    model.eval()  # Ensure the model is in evaluation mode
    nominal_data, ordinal_data, y = zip(*[dataset[i] for i in range(len(dataset))])
    nominal_data = torch.stack(nominal_data)
    ordinal_data = torch.stack(ordinal_data)
    y = torch.tensor(y)

    # Original model predictions
    with torch.no_grad():
        original_preds = model(
            [nominal_data[:, i] for i in range(nominal_data.shape[1])], ordinal_data
        ).argmax(dim=1).cpu().numpy()
    original_score = metric(y.numpy(), original_preds)

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
                ).argmax(dim=1).cpu().numpy()
            elif feature_type == "ordinal":
                shuffled_ordinal_data = ordinal_data.clone()
                shuffled_ordinal_data[:, idx] = shuffled_ordinal_data[:, idx][torch.randperm(ordinal_data.size(0))]
                shuffled_preds = model(
                    [nominal_data[:, i] for i in range(nominal_data.shape[1])], shuffled_ordinal_data
                ).argmax(dim=1).cpu().numpy()

            # Calculate drop in performance
            shuffled_score = metric(y.numpy(), shuffled_preds)
            feature_importances[column_name] = original_score - shuffled_score

    # Print sorted feature importances
    for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance}")

    # Visualization
    import matplotlib.pyplot as plt

    # Sort and plot
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_importances)

    plt.figure(figsize=(12, 8))  # Extended chart width and height
    plt.barh(features, scores)

    # Apply Times New Roman font to the plot
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.xlabel("Drop in Performance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.title("Feature Importances", fontsize=16)

    # Adjust y-axis labels for readability
    plt.yticks(fontsize=9)
    plt.gca().invert_yaxis()  # Invert so the most important features are at the top
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    import pandas as pd
    from preProcessing_Func import analysis_func as dfppaf
    tus_subset_final_moreCol = r'dataset_TUS_individual/TUS_indiv_03_final_moreCol.csv'
    data = pd.read_csv(tus_subset_final_moreCol)

    output_column = 'Family Typology'

    # output performances:  https://docs.google.com/spreadsheets/d/1L2-GIfeopAzJjyQxa5Qw3gZDzSMihs1kyyyR91WvdVQ/edit?gid=0#gid=0

    """
    output_columns = [
    'Family Typology', 
    'Main Income Source', 
    'School Enrollment', 
    'Nuclear Family, Occupant Profile', 
    'Nuclear Family, Typology', 
    'Nuclear Family, Occupant Sequence Number', 
    """
    model, train_loader, val_loader, test_loader = ML(
        df=data, outCol=output_column,
        learning_rate=0.025, num_epochs=20, batch_size=64,
        ordinal_hidden_size=24,  fc_hidden_sizes=(24,),
        dropout_rate=0.0,  use_batch_norm=True)

    # ANALYSIS #########################################################################################################
    ID_DROP = ["Household_ID"]
    non_visual = False
    dfppaf.analysis(input_path=tus_subset_final_moreCol, unique=non_visual, uniqueIDcolstoDrop=ID_DROP)
