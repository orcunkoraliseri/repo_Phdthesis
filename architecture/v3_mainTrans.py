# TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER
# EXTRA_________________________________________________________________________________________________________________
def record_trial_parameters(trial, model, hyperparameters, output_csv='Transformer_tuning_params.csv'):
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

def save_trial_details(study, filename_prefix="trial_results"):
    import os
    import csv
    # Create directory to store the outputs
    folder_name = "TuningcsvOutputs"
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

    # Define fieldnames for the CSV
    fieldnames = ['iter', 'target', 'activity_accuracy', 'location_accuracy', 'withNOB_accuracy', 'last_epoch'] + list(
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

# TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER
# BASE MODEL____________________________________________________________________________________________________________
def modelBaseTransformerTraining(df, epochNum=50, batch_size=48, learning_rate=0.001):
    from v3_modeling import TransformerModel
    from v3_preprocessing import data_preprocess
    from v3_training import train_and_evaluate_modelBase
    from v3_plotting import plot_history

    X_train, y_activity_train, y_location_train, y_withNOB_train, X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
    X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder = data_preprocess(df)

    # MODEL FEATURES ---------------------------------------------------------------------------------------------------
    import torch
    # EMBEDDINGS
    # most effective features according to impact analysis
    num_educationCat = 3
    num_employmentCat = 6
    num_genderCat = 2
    num_famTypologyCat = 4
    num_numFamMembCat = 6
    # manually selected feature for ordeting the occupants in the HH
    num_OCCinHHCat = 6
    # non-temporal TUS daily features
    num_seasonCat = 4
    num_unique_weekCat = 3
    # 4 continuous features, sin and cos of 'hourStart_Activity', 'hourEnd_Activity'
    num_continuous_features = 4
    # embed size
    embed_size = 250

    # MODEL PARAMETERS
    num_hidden_layers = 3
    dropout_loc = 0.1
    dropout_withNOB = 0.1
    dropout_embedding = 0.1
    dropout_TFTs = 0.1

    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary

    # MODEL-------------------------------------------------------------------------------------------------------------
    import warnings
    # Suppress warnings during model initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TransformerModel(num_educationCat=num_educationCat,
                                 num_employmentCat=num_employmentCat,
                                 num_genderCat=num_genderCat,
                                 num_famTypologyCat=num_famTypologyCat,
                                 num_numFamMembCat=num_numFamMembCat,
                                 num_OCCinHHCat=num_OCCinHHCat,
                                 num_seasonCat=num_seasonCat,
                                 num_unique_weekCat=num_unique_weekCat,
                                 num_continuous_features=num_continuous_features,
                                 output_dim_activity=output_dim_activity,
                                 output_dim_location=output_dim_location,
                                 output_dim_withNOB=output_dim_withNOB,
                                 num_hidden_layers=num_hidden_layers,
                                 dropout_loc = dropout_loc,
                                 dropout_withNOB = dropout_withNOB,
                                 dropout_embedding=dropout_embedding,
                                 dropout_transformer=dropout_TFTs,
                                 embed_size=embed_size,
                                 nhead=4)

    #model.apply(init_weights)

    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")
    model.to(mps_device)  # Move model to MPS device

    # TEST: MODEL PROPERTIES -------------------------------------------------------------------------------------------
    #print(model)

    # TRAIN & EVALUATE--------------------------------------------------------------------------------------------------
    # Assuming 'model' is the compiled model from the provided architecture
    # Call the train and evaluate function
    history, trained_model = train_and_evaluate_modelBase(model,
                                                          X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                          X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                          device=mps_device,
                                                          epochs=epochNum, batch_size=batch_size, learning_rate=learning_rate,
                                                          checkpoint_path='best_model_base_Transformer.pth')

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

def modelBaseTransformerTrainingNoEmbed(df, epochNum=50, batch_size=48, learning_rate=0.001, nhead=4, numHiddenLayers=3,
                                        dropout_loc = 0.1, dropout_withNOB = 0.1, dropout_TFTs = 0.1):

    from v3_modeling import TransformerModelNoEmbed
    from v3_training import train_and_evaluate_modelBaseNoEmbed
    from v3_preprocessing import data_preprocessNoEmbedding
    from v3_plotting import plot_history

    df = df[['Household_ID',
             'months_season', 'week_or_weekend',
             'Occupant_ID_in_HH',
             'hourStart_Activity', 'hourEnd_Activity',
             'Occupant_Activity', 'location', 'withNOBODY']]

    X_train, y_activity_train, y_location_train, y_withNOB_train, X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
    X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder = data_preprocessNoEmbedding(df)

    # MODEL FEATURES ---------------------------------------------------------------------------------------------------
    import torch
    num_continuous_features = 4

    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary

    # MODEL-------------------------------------------------------------------------------------------------------------
    # Suppress warnings during model initialization
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TransformerModelNoEmbed(
                     num_continuous_features,
                     output_dim_activity, output_dim_location, output_dim_withNOB,
                     num_hidden_layers=numHiddenLayers,
                     dropout_loc = dropout_loc, dropout_withNOB = dropout_withNOB, dropout_transformer=dropout_TFTs,
                     nhead=nhead)

    #model.apply(init_weights)

    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")
    model.to(mps_device)  # Move model to MPS device

    # TEST: MODEL PROPERTIES -------------------------------------------------------------------------------------------
    #print(model)

    # TRAIN & EVALUATE--------------------------------------------------------------------------------------------------
    # Assuming 'model' is the compiled model from the provided architecture
    # Call the train and evaluate function
    history, trained_model = train_and_evaluate_modelBaseNoEmbed(model,
                                                          X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                          X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                          device=mps_device,
                                                          epochs=epochNum, batch_size=batch_size, learning_rate=learning_rate,
                                                          checkpoint_path='best_model_NoEmbed_Transformer.pth')

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

# TUNING:NO CROSS VALIDATION __________________________________________________________________________________________
def objectiveTransformer(trial, epochs, df): #https://docs.google.com/presentation/d/1l-rP260eG2kS4SEYLNSfmJEtv2A_DmmVPuwYJNKHF9o/edit#slide=id.g2e06efbb215_2_0
    from v3_modeling import TransformerModelTuning, weights_init_he_normal, weights_init_lecun_normal, weights_init_glorot_uniform
    from v3_training import train_and_evaluate_model_tuning
    from v3_preprocessing import data_preprocess
    import torch

    X_train, y_activity_train, y_location_train, y_withNOB_train, X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
    X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder = data_preprocess(df)

    # EMBEDDINGS
    # most effective features according to impact analysis
    num_educationCat = 3
    num_employmentCat = 6
    num_genderCat = 2
    num_famTypologyCat = 4
    num_numFamMembCat = 6
    num_OCCinHHCat = 6
    # non-temporal TUS daily features
    num_seasonCat = 4
    num_unique_weekCat = 3
    # 4 continuous features, sin and cos of 'hourStart_Activity', 'hourEnd_Activity'
    num_continuous_features = 4

    # CONSTANT VALUES
    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    #print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary
    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")

    # MODEL HYPERPARAMETERS
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [3, 6, 9])
    dropout_loc = trial.suggest_categorical('dropout_loc', [0.1, 0.5, 1.0])
    dropout_withNOB = trial.suggest_categorical('dropout_withNOB', [0.1, 0.5, 1.0])
    dropout_embedding = trial.suggest_categorical('dropout_embedding', [0.1, 0.5, 1.0])
    dropout_TFTs = trial.suggest_categorical('dropout_TFTs', [0.1, 0.5, 1.0])
    embed_size = trial.suggest_categorical('embedding_size', [50, 100, 200,300])
    nhead = trial.suggest_categorical('nhead', [2, 4])

    # TRAINING HYPERPARAMETERS
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    batch_size = trial.suggest_categorical('batch_size', [48, 96])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001,0.0005,0.0001])

    # TUNING HYPERPARAMETERS
    epochs = epochs
    import warnings

    # Suppress warnings during model initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TransformerModelTuning(num_educationCat=num_educationCat,
                                       num_employmentCat=num_employmentCat,
                                       num_genderCat=num_genderCat,
                                       num_famTypologyCat=num_famTypologyCat,
                                       num_numFamMembCat=num_numFamMembCat,
                                       num_OCCinHHCat=num_OCCinHHCat,
                                       num_seasonCat=num_seasonCat,
                                       num_unique_weekCat=num_unique_weekCat,
                                       num_continuous_features=num_continuous_features,
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
                                       ).to(mps_device)

    # Train the model (you might want to put your training code into a function)
    history, trained_model =  train_and_evaluate_model_tuning(model,
                                                             X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                             X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                             optimizer_name,
                                                             epochs, batch_size, learning_rate,
                                                             device=mps_device,
                                                             checkpoint_path='best_modelTransformer.pth',
                                                             verbose=False ,use_tensorboard=False, load_checkpoint=False,
                                                              use_early_stopping=True)

    # Extract the last epoch's validation losses
    val_loss_activity = history['valid_activity_loss'][-1]
    val_loss_location = history['valid_location_loss'][-1]
    val_loss_withNOB = history['valid_withNOB_loss'][-1]

    # Suggest weights for each loss component as discrete values
    weight_activity = trial.suggest_categorical('weight_activity', [0.25, 0.5, 0.75, 1.0])
    weight_location = trial.suggest_categorical('weight_location', [0.25, 0.5, 0.75, 1.0])
    weight_withNOB = trial.suggest_categorical('weight_withNOB', [0.25, 0.5, 0.75, 1.0])

    # Normalize weights so that they sum to 1
    total_weight = weight_activity + weight_location + weight_withNOB
    weight_activity /= total_weight
    weight_location /= total_weight
    weight_withNOB /= total_weight

    # Return the weighted sum of the validation losses
    weighted_loss = (weight_activity * val_loss_activity +
                     weight_location * val_loss_location +
                     weight_withNOB * val_loss_withNOB)

    return weighted_loss

def main_tuningTransformer(csv_filepath, n_trials, epochs):
    import os
    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the Optuna study
    import optuna
    import json
    import optuna.visualization as vis
    # Define a wrapper function to pass the epochs parameter
    def objective_wrapper(trial):
        return objectiveTransformer(trial, epochs=epochs)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_wrapper, n_trials=n_trials)

    # Save best trial's hyperparameters to a JSON file
    best_trial_params = study.best_trial.params
    with open(f"{filename_prefix}_best_TuningParams_Transformer.json", 'w') as outfile:
        json.dump(best_trial_params, outfile)

    # Optional: Generate and save visualization plot
    parallel_coordinate_plot = vis.plot_parallel_coordinate(study)
    parallel_coordinate_plot.write_html(f"{filename_prefix}_Transformer.html")

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    print('Value:', study.best_trial.value)
    print('Params:', best_trial_params)

# TUNING:CROSS VALIDATION __________________________________________________________________________________________
def objectiveTransformer_kFold(trial, epochs, df,num_split):  # https://docs.google.com/presentation/d/1l-rP260eG2kS4SEYLNSfmJEtv2A_DmmVPuwYJNKHF9o/edit#slide=id.g2e06efbb215_2_0
    from sklearn.preprocessing import LabelEncoder
    from v3_modeling import TransformerModelTuning
    from v3_training import train_and_evaluate_model_tuning
    from v3_preprocessing import data_preprocess_k_fold_split
    import torch
    import matplotlib.pyplot as plt
    import os

    # PLOTTING: Create folder to store the plots
    trial_plots_ACC_folder = "Tuning_ACCURACY_Trial_History_Plots"
    if not os.path.exists(trial_plots_ACC_folder):
        os.makedirs(trial_plots_ACC_folder)

    # PLOTTING: Create folder to store the plots
    trial_plots_LOSS_folder = "Tuning_LOSS_Trial_History_Plots"
    if not os.path.exists(trial_plots_LOSS_folder):
        os.makedirs(trial_plots_LOSS_folder)

    # EMBEDDINGS
    # most effective features according to impact analysis
    num_educationCat = df['Education Degree'].nunique()
    num_employmentCat = df['Employment status'].nunique()
    num_genderCat = df['Gender'].nunique()
    num_famTypologyCat = df['Family Typology'].nunique()
    num_numFamMembCat = df['Number Family Members'].nunique()
    num_OCCinHHCat = df['Occupant_ID_in_HH'].nunique()
    # non-temporal TUS daily features
    num_seasonCat = 4
    num_unique_weekCat = 3
    # 4 continuous features, sin and cos of 'hourStart_Activity', 'hourEnd_Activity'
    num_continuous_features = 4

    # CONSTANT VALUES
    mps_device = torch.device("mps")
    optimizer_name = "adam"
    dropout_loc = 0.25
    dropout_withNOB = 0.25
    dropout_embedding = 0
    dropout_TFTs = 0
    weight_activity = 1
    weight_location = 1
    weight_withNOB = 1

    # MODEL HYPERPARAMETERS
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [3, 5, 7])
    embed_size = trial.suggest_categorical('embedding_size', [50, 100, 200, 300])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])

    # TRAINING HYPERPARAMETERS
    batch_size = trial.suggest_categorical('batch_size', [48, 96, 128])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0005, 0.0001])

    # TUNING HYPERPARAMETERS
    folds, df_preprocessed = data_preprocess_k_fold_split(df, n_splits=num_split)
    total_val_loss_activity = 0.0
    total_val_loss_location = 0.0
    total_val_loss_withNOB = 0.0
    total_activity_accuracy = 0.0
    total_location_accuracy = 0.0
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

        X_train = train_data.values.reshape(-1, 48, train_data.shape[1])
        X_valid = val_data.values.reshape(-1, 48, val_data.shape[1])

        y_activity_train = y_activity_train.reshape(-1, 48)
        y_activity_valid = y_activity_valid.reshape(-1, 48)
        y_location_train = y_location_train.reshape(-1, 48)
        y_location_valid = y_location_valid.reshape(-1, 48)
        y_withNOB_train = y_withNOB_train.reshape(-1, 48)
        y_withNOB_valid = y_withNOB_valid.reshape(-1, 48)

        output_dim_activity = len(set(y_activity_train.flatten()))
        output_dim_location = 1
        output_dim_withNOB = 1

        import warnings
        # Suppress warnings during model initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = TransformerModelTuning(num_educationCat=num_educationCat,
                                           num_employmentCat=num_employmentCat,
                                           num_genderCat=num_genderCat,
                                           num_famTypologyCat=num_famTypologyCat,
                                           num_numFamMembCat=num_numFamMembCat,
                                           num_OCCinHHCat=num_OCCinHHCat,
                                           num_seasonCat=num_seasonCat,
                                           num_unique_weekCat=num_unique_weekCat,
                                           num_continuous_features=num_continuous_features,
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
                                           ).to(mps_device)

        history, trained_model = train_and_evaluate_model_tuning(
            model, X_train, y_activity_train, y_location_train, y_withNOB_train,
            X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
            optimizer_name=optimizer_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=mps_device, w_act=weight_activity, w_loc=weight_location, w_NOB=weight_withNOB,
            checkpoint_path=f'best_modelLSTM_fold{fold_idx}.pth',
            verbose=False, use_tensorboard=False, load_checkpoint=False,
            use_early_stopping=True,
            use_scheduler=True,
        )

        # PLOTTING: Collect history data for plotting
        training_loss_history.extend(history['train_activity_loss'])
        validation_loss_history.extend(history['valid_activity_loss'])
        training_accuracy_history.extend(history['train_activity_accuracy'])
        validation_accuracy_history.extend(history['valid_activity_accuracy'])

        total_val_loss_activity += history['valid_activity_loss'][-1]
        total_val_loss_location += history['valid_location_loss'][-1]
        total_val_loss_withNOB += history['valid_withNOB_loss'][-1]
        total_activity_accuracy += history['valid_activity_accuracy'][-1]
        total_location_accuracy += history['valid_location_accuracy'][-1]
        total_withNOB_accuracy += history['valid_withNOB_accuracy'][-1]

        # After training the model
        last_epoch = len(
            history['valid_activity_loss'])  # Assuming the length of the loss list corresponds to the number of epochs

    # Collect hyperparameters in a dictionary
    hyperparameters = {
        'num_hidden_layers': num_hidden_layers,
        'nhead': nhead,
        'embed_size': embed_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    # Call the function to record parameters and hyperparameters
    record_trial_parameters(trial, model, hyperparameters)

    avg_val_loss_activity = total_val_loss_activity / len(folds)
    avg_val_loss_location = total_val_loss_location / len(folds)
    avg_val_loss_withNOB = total_val_loss_withNOB / len(folds)

    avg_activity_accuracy = total_activity_accuracy / len(folds)
    avg_location_accuracy = total_location_accuracy / len(folds)
    avg_withNOB_accuracy = total_withNOB_accuracy / len(folds)

    # PLOTTING
    # Plot the trial's training and validation loss and accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss_history, label="Training Loss")
    plt.plot(validation_loss_history, label="Validation Loss")
    plt.title(f"Training and Validation Loss (Trial {trial.number})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{trial_plots_LOSS_folder}/loss_trial_{trial.number}.png")
    plt.close()
    # PLOTTING
    plt.figure(figsize=(10, 6))
    plt.plot(training_accuracy_history, label="Training Accuracy")
    plt.plot(validation_accuracy_history, label="Validation Accuracy")
    plt.title(f"Training and Validation Accuracy (Trial {trial.number})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{trial_plots_ACC_folder}/accuracy_trial_{trial.number}.png")
    plt.close()

    total_weight = weight_activity + weight_location + weight_withNOB
    weight_activity /= total_weight
    weight_location /= total_weight
    weight_withNOB /= total_weight

    weighted_loss = (
            weight_activity * avg_val_loss_activity +
            weight_location * avg_val_loss_location +
            weight_withNOB * avg_val_loss_withNOB
    )

    return weighted_loss, avg_activity_accuracy, avg_location_accuracy, avg_withNOB_accuracy, last_epoch

def main_tuningTransformer_kFold(csv_filepath, n_trials, epochs, num_split, df):
    import os
    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the Optuna study
    import optuna
    import json
    import torch
    from v3_plotting import vis_KDE_HyperCombin, vis_Bubble_HyperCombin, vis_Scatter_HyperCombin, vis_Hexbin_HyperCombin, vis_PCP
    # Define a wrapper function to pass the epochs parameter
    def objective_wrapper(trial):
        weighted_loss, avg_activity_accuracy, avg_location_accuracy, avg_withNOB_accuracy, last_epoch = objectiveTransformer_kFold(trial,
                                                                                                                epochs=epochs,
                                                                                                                df=df,
                                                                                                                num_split=num_split)

        # Optionally store accuracy values in the trial's user attributes
        trial.set_user_attr('activity_accuracy', avg_activity_accuracy)
        trial.set_user_attr('location_accuracy', avg_location_accuracy)
        trial.set_user_attr('withNOB_accuracy', avg_withNOB_accuracy)
        trial.set_user_attr('last_epoch', last_epoch)

        # Clear the CUDA cache after each trial to free up memory
        torch.cuda.empty_cache()

        return weighted_loss
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_wrapper, n_trials=n_trials)

    # Save best trial's hyperparameters to a JSON file
    best_trial_params = study.best_trial.params
    with open(f"{filename_prefix}_best_TuningParams_Transformer.json", 'w') as outfile:
        json.dump(best_trial_params, outfile)

    save_trial_details(study, "TResults")
    vis_Scatter_HyperCombin(study, 'ScatterPlots')
    vis_Hexbin_HyperCombin(study, 'HexbinPlots')
    vis_KDE_HyperCombin(study, 'KDEPlots')
    vis_Bubble_HyperCombin(study, 'bubblePlots')
    vis_PCP(study, target_name='Target (e.g., weighted_loss)')

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    print('Value:', study.best_trial.value)
    print('Params:', best_trial_params)

# TRAINING: AFTER TUNING -----------------------------------------------------------------------------------------------
def trainEvaluate_afterTuning_Transformer(df, csv_filepath, epochBase=500,):
    import torch
    import json
    import os
    from v3_modeling import TransformerModelTuning
    from v3_training import train_and_evaluate_model_tuning
    from v3_evaluate import evaluate_and_save_afterTuning, evalACT_classify_afterTuning
    from v3_preprocessing import data_preprocess
    from v3_plotting import plot_history

    X_train, y_activity_train, y_location_train, y_withNOB_train, X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
    X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder = data_preprocess(df)

    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    #print("filename_prefix:", filename_prefix)

    # Load best hyperparameters
    with open(f"{filename_prefix}_best_TuningParams_Transformer.json", 'r') as infile:
        best_params = json.load(infile)
    print(best_params)

    # EMBEDDINGS
    # most effective features according to impact analysis
    num_educationCat = df['Education Degree'].nunique()
    num_employmentCat = df['Employment status'].nunique()
    num_genderCat = df['Gender'].nunique()
    num_famTypologyCat = df['Family Typology'].nunique()
    num_numFamMembCat = df['Number Family Members'].nunique()
    # manually selected feature for ordeting the occupants in the HH
    num_OCCinHHCat = df['Occupant_ID_in_HH'].nunique()
    # non-temporal TUS daily features
    num_seasonCat = 4
    num_unique_weekCat = 3
    # 4 continuous features, sin and cos of 'hourStart_Activity', 'hourEnd_Activity'
    num_continuous_features = 4

    optimizer_name = "adam"
    dropout_loc = 0.25
    dropout_withNOB = 0.25
    dropout_embedding = 0
    dropout_TFTs = 0
    w_act = 1
    w_loc = 1
    w_NOB = 1

    # Unpack hyperparameters
    num_hidden_layers = best_params['num_hidden_layers']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    embed_size = best_params['embedding_size']
    #optimizer_name = best_params['optimizer']
    #dropout_loc = best_params['dropout_loc']
    #dropout_withNOB = best_params['dropout_withNOB']
    #dropout_embedding = best_params['dropout_embedding']
    #dropout_TFTs = best_params['dropout_TFTs']
    nhead = best_params['nhead']

    #w_act = best_params['weight_activity']
    #w_loc = best_params['weight_location']
    #w_NOB = best_params['weight_withNOB']

    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary

    # Suppress warnings during model initialization
    import warnings
    # Suppress warnings during model initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TransformerModelTuning(num_educationCat=num_educationCat,
                                       num_employmentCat=num_employmentCat,
                                       num_genderCat=num_genderCat,
                                       num_famTypologyCat=num_famTypologyCat,
                                       num_numFamMembCat=num_numFamMembCat,
                                       num_OCCinHHCat=num_OCCinHHCat,
                                       num_seasonCat=num_seasonCat,
                                       num_unique_weekCat=num_unique_weekCat,
                                       num_continuous_features=num_continuous_features,
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
                                       )

    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")
    #mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(mps_device)  # Move model to MPS device

    #print(model)

    # Train the model
    history, trained_model =  train_and_evaluate_model_tuning(model,
                                                             X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                             X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                             optimizer_name=optimizer_name,
                                                             epochs=epochBase,
                                                             batch_size=batch_size,
                                                             learning_rate=learning_rate,
                                                             device=mps_device, w_act=w_act, w_loc=w_loc, w_NOB=w_NOB,
                                                             checkpoint_path=f"{filename_prefix}_best_modelTransformer.pth",
                                                             verbose=True, use_tensorboard=True, load_checkpoint=False,
                                                             record_memory_usage=False, memory_log_path='memory_usage.txt',
                                                              use_early_stopping=False,
                                                              use_scheduler=True,)


    # Optionally: Evaluate the trained model further and save
    # Example: Save the model if it's better than previous models
    torch.save(trained_model.state_dict(), f"{filename_prefix}_best_modelTransformer.pth")
    print("training_afterTuning()_Transformer is completed")

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

    # how to open tensorboard recordings
    # tensorboard --logdir=runs/End_to_End_full_embedding # from terminal

    model.load_state_dict(torch.load(f"{filename_prefix}_best_modelTransformer.pth"))
    evaluate_and_save_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder=label_encoder, device=mps_device, csv_filepath= csv_filepath, arch_type="Transformer")

    # EVALUATION for model performance for each activity categories ----------------------------------------------------
    # Assume `model`, `X_test`, `y_test`, and `label_encoder` are already defined and the model is trained
    evalACT_classify_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder, device=mps_device, save_csv=True, csv_filepath= csv_filepath, arch_type="Transformer")