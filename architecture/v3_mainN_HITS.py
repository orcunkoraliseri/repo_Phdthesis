# TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER
# EXTRA_________________________________________________________________________________________________________________
def record_trial_parameters(trial, model, hyperparameters, output_csv='NHiTS_tuning_params.csv'):
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

# N-HITS: BASEMODEL NO EMBEDDING ---------------------------------------------------------------------------------------
def BaseTrainNoEmbed(df, h_unit, n_epoch=50, b_size=48, l_rate=0.001, n_block=3, n_layer=3):

    from v3_preprocessing import data_preprocessNoEmbedding
    from v3_plotting import plot_history

    df = df[['Household_ID',
             'months_season', 'week_or_weekend',
             'Occupant_ID_in_HH',
             'hourStart_Activity', 'hourEnd_Activity',
             'Occupant_Activity', 'location', 'withNOBODY']]

    X_train, y_activity_train, y_location_train, y_withNOB_train, \
        X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
        X_test, y_activity_test, y_location_test, y_withNOB_test, \
        label_encoder = data_preprocessNoEmbedding(df)

    from v3_modeling import NHiTSModelNoEmbed
    from v3_training import train_and_evaluate_modelBaseNoEmbed
    # MODEL FEATURES ---------------------------------------------------------------------------------------------------
    import torch
    # 4 continuous features, sin and cos of 'hourStart_Activity', 'hourEnd_Activity'
    num_continuous_features = 4
    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary

    # MODEL-------------------------------------------------------------------------------------------------------------
    model = NHiTSModelNoEmbed(
        input_size=num_continuous_features,
        output_dim_activity=output_dim_activity,
        output_dim_location=output_dim_location,
        output_dim_withNOB=output_dim_withNOB,
        hidden_units=h_unit,
        num_blocks=n_block,
        num_layers=n_layer,
        )

    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")
    #mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                                                          epochs=n_epoch, batch_size=b_size, learning_rate=l_rate,
                                                          checkpoint_path=f'best_model_N_HiTS_NoEmbed.pth')

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

# N-HITS: BASEMODEL ----------------------------------------------------------------------------------------------------
def BaseTrain(df, h_unit=3, n_epoch=50, b_size=48, l_rate=0.001, n_block=3, n_layer=3, embed_size= 250, dr_embed = 0.5,
              drop_act=0.25, drop_loc=0.25, drop_NOB=0.25, w_act=1, w_loc=1, w_NOB=1,):
    """
    example:
    v3_mainN_HITS.BaseTrain(df=df, h_unit=512, n_epoch = 200, b_size = 24, l_rate = 0.01, n_block = 4, n_layer = 4,
                            embed_size = 500, dr_embed = 0.00, drop_act = 0.0, drop_loc = 0.25, drop_NOB = 0.25, w_act=1, w_loc=1, w_NOB=1,)
    """
    from v3_preprocessing import data_preprocess
    from v3_plotting import plot_history

    X_train, y_activity_train, y_location_train, y_withNOB_train, \
        X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
        X_test, y_activity_test, y_location_test, y_withNOB_test, \
        label_encoder = data_preprocess(df)

    from v3_modeling import NHiTSModel
    from v3_training import train_and_evaluate_modelBase
    # MODEL FEATURES ---------------------------------------------------------------------------------------------------
    import torch
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
    # embed size

    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary

    # MODEL-------------------------------------------------------------------------------------------------------------
    model = NHiTSModel(
        num_educationCat, num_employmentCat, num_genderCat, num_famTypologyCat,
        num_numFamMembCat,
        num_OCCinHHCat,
        num_seasonCat, num_unique_weekCat,
        num_continuous_features,
        output_dim_activity=output_dim_activity,
        output_dim_location=output_dim_location,
        output_dim_withNOB=output_dim_withNOB,
        hidden_units=h_unit,
        num_blocks=n_block,
        num_layers=n_layer,
        embed_size=embed_size,
        dropout_embedding=dr_embed,
        drop_act=drop_act,
        drop_loc=drop_loc,
        drop_NOB=drop_NOB,)

    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")
    #mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(mps_device)  # Move model to MPS device

    # TEST: MODEL PROPERTIES -------------------------------------------------------------------------------------------
    #print(model)

    # TRAIN & EVALUATE--------------------------------------------------------------------------------------------------
    # Assuming 'model' is the compiled model from the provided architecture
    # Call the train and evaluate function
    history, trained_model = train_and_evaluate_modelBase(model,
                                                          X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                          X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                          w_act=w_act, w_loc=w_loc, w_NOB=w_NOB,
                                                          device=mps_device,
                                                          epochs=n_epoch, batch_size=b_size, learning_rate=l_rate,
                                                          checkpoint_path=f'best_model_N_HiTS.pth')

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

# N-HITS: TUNING WITH CROSS VALIDATION ---------------------------------------------------------------------------------
def objecN_HITS_kFold(trial, epochs, df,num_split):  # https://docs.google.com/presentation/d/1l-rP260eG2kS4SEYLNSfmJEtv2A_DmmVPuwYJNKHF9o/edit#slide=id.g2e06efbb215_2_0
    from sklearn.preprocessing import LabelEncoder
    from v3_modeling import NHiTSModelTuning
    from v3_training import train_and_evaluate_model_tuning
    from v3_preprocessing import data_preprocess_k_fold_split
    import torch

    # EMBEDDINGS
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

    # CONSTANT VALUES
    mps_device = torch.device("mps")
    #mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_name = "adam"
    dropout_loc = 0.25
    dropout_withNOB = 0.25
    dropout_embedding = 0
    weight_activity = 1
    weight_location = 1
    weight_withNOB = 1

    # MODEL HYPERPARAMETERS
    hidden_units = trial.suggest_categorical('hidden_units', [256, 512, 1024])
    num_blocks = trial.suggest_categorical('num_blocks', [3, 5, 7])
    num_layers = trial.suggest_categorical('num_layers', [7, 9, 11])
    embed_size = trial.suggest_categorical('embedding_size', [50, 100, 200, 300, 400])
    # TRAINING HYPERPARAMETERS
    batch_size = trial.suggest_categorical('batch_size', [24, 48,])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0005, 0.0001])

    # TUNING HYPERPARAMETERS
    folds, df_preprocessed = data_preprocess_k_fold_split(df, n_splits=num_split)
    total_val_loss_activity = 0.0
    total_val_loss_location = 0.0
    total_val_loss_withNOB = 0.0

    total_activity_accuracy = 0.0
    total_location_accuracy = 0.0
    total_withNOB_accuracy = 0.0

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
            model = NHiTSModelTuning(num_educationCat=num_educationCat, num_employmentCat=num_employmentCat,
                                     num_genderCat=num_genderCat, num_famTypologyCat=num_famTypologyCat,
                                     num_numFamMembCat=num_numFamMembCat,num_OCCinHHCat=num_OCCinHHCat,
                                     num_seasonCat=num_seasonCat, num_unique_weekCat=num_unique_weekCat,
                                     num_continuous_features=num_continuous_features,
                                     output_dim_activity=output_dim_activity, output_dim_location=output_dim_location, output_dim_withNOB=output_dim_withNOB,
                                     dropout_embedding=dropout_embedding,
                                     drop_loc=dropout_loc, drop_NOB=dropout_withNOB,
                                     hidden_units=hidden_units, num_blocks=num_blocks, num_layers=num_layers, embed_size=embed_size).to(mps_device)

        history, trained_model = train_and_evaluate_model_tuning(
            model, X_train, y_activity_train, y_location_train, y_withNOB_train,
            X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
            optimizer_name=optimizer_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=mps_device, w_act=weight_activity, w_loc=weight_location, w_NOB=weight_withNOB,
            checkpoint_path=f'best_model_N_HITS_fold{fold_idx}.pth',
            verbose=False, use_tensorboard=False, load_checkpoint=False,
            use_early_stopping=True,
        )

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
        'hidden_units': hidden_units,
        'num_blocks': num_blocks,
        'num_layers': num_layers,
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

    total_weight = weight_activity + weight_location + weight_withNOB
    weight_activity /= total_weight
    weight_location /= total_weight
    weight_withNOB /= total_weight

    weighted_loss = (weight_activity * avg_val_loss_activity + weight_location * avg_val_loss_location + weight_withNOB * avg_val_loss_withNOB)

    return weighted_loss, avg_activity_accuracy, avg_location_accuracy, avg_withNOB_accuracy, last_epoch

def main_tuning_N_HITS_kFold(csv_filepath, n_trials, epochs, num_split, df):
    import os
    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the Optuna study
    import optuna
    import json
    import optuna.visualization as vis
    from v3_plotting import visualize_advanced_parallel_coordinates, visualize_hyperparameters_and_combinations

    # Define a wrapper function to pass the epochs parameter
    def objective_wrapper(trial):
        weighted_loss, avg_activity_accuracy, avg_location_accuracy, avg_withNOB_accuracy, last_epoch = objecN_HITS_kFold(trial,
                                                                                                                epochs=epochs,
                                                                                                                df=df,
                                                                                                                num_split=num_split)

        # Optionally store accuracy values in the trial's user attributes
        trial.set_user_attr('activity_accuracy', avg_activity_accuracy)
        trial.set_user_attr('location_accuracy', avg_location_accuracy)
        trial.set_user_attr('withNOB_accuracy', avg_withNOB_accuracy)
        trial.set_user_attr('last_epoch', last_epoch)

        return weighted_loss
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_wrapper, n_trials=n_trials)

    # Save best trial's hyperparameters to a JSON file
    best_trial_params = study.best_trial.params
    with open(f"{filename_prefix}_best_TuningParams_N_HITS_kFold.json", 'w') as outfile:
        json.dump(best_trial_params, outfile)

    # Optional: Generate and save visualization plot
    parallel_coordinate_plot = vis.plot_parallel_coordinate(study)
    parallel_coordinate_plot.write_html(f"{filename_prefix}_N_HITS_kFold.html")

    # Save details of all trials to a .txt file
    with open(f"{filename_prefix}_all_trials.txt", 'w') as f:
        f.write('All trial results:\n')
        for trial in study.trials:
            f.write(f"Iter: {trial.number}, Target: {trial.value:.4f}, Params: {trial.params}, ")
            f.write(f"Activity Accuracy: {trial.user_attrs.get('activity_accuracy', 'N/A'):.4f}, ")
            f.write(f"Location Accuracy: {trial.user_attrs.get('location_accuracy', 'N/A'):.4f}, ")
            f.write(f"WithNOB Accuracy: {trial.user_attrs.get('withNOB_accuracy', 'N/A'):.4f},")
            f.write(f"last epoch: {trial.user_attrs.get('last_epoch', 'N/A'):},")

    # Collect all possible keys for fieldnames
    all_params_keys = set()
    for trial in study.trials:
        all_params_keys.update(trial.params.keys())

    # Define fieldnames
    fieldnames = ['iter', 'target', 'activity_accuracy', 'location_accuracy', 'withNOB_accuracy', 'last_epoch'] + list(
        all_params_keys)

    # Save details of all trials to a .csv file
    import csv
    with open(f"{filename_prefix}_all_trials.csv", 'w', newline='') as csvfile:
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

    # Plot each hyperparameter against the target
    import matplotlib.pyplot as plt
    for param in all_params_keys:
        param_values = [trial.params.get(param) for trial in study.trials]
        target_values = [trial.value for trial in study.trials]

        plt.figure()
        plt.scatter(param_values, target_values)
        plt.xlabel(param)
        plt.ylabel('Target (e.g., weighted_loss)')
        plt.title(f'{param} vs Target')
        plt.savefig(f'{filename_prefix}_{param}_vs_target.png')
        plt.close()

    visualize_hyperparameters_and_combinations(study, 'hyperparam_plots')
    visualize_advanced_parallel_coordinates(study, target_name='Target (e.g., weighted_loss)')

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    print('Value:', study.best_trial.value)
    print('Params:', best_trial_params)

# TRAINING: AFTER TUNING -----------------------------------------------------------------------------------------------
def trainEvaluate_afterTuning_N_HITS(df, csv_filepath, epochBase=500,):
    import torch
    import json
    import os
    from v3_modeling import NHiTSModelTuning
    from v3_training import train_and_evaluate_model_tuning
    from v3_evaluate import evaluate_and_save_afterTuning, evalACT_classify_afterTuning
    from v3_preprocessing import data_preprocess
    from v3_plotting import plot_history

    X_train, y_activity_train, y_location_train, y_withNOB_train, X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, \
    X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder = data_preprocess(df)

    # Extract the filename without extension
    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    print("filename_prefix:", filename_prefix)

    # Load best hyperparameters
    with open(f"{filename_prefix}_best_TuningParams_N_HITS_kFold.json", 'r') as infile:
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

    # CONSTANT VALUES
    optimizer_name = "adam"
    dropout_loc = 0.25
    dropout_withNOB = 0.25
    dropout_embedding = 0
    weight_activity = 1
    weight_location = 1
    weight_withNOB = 1

    # Unpack Model hyperparameters
    hidden_units = best_params['hidden_units']
    num_blocks = best_params['num_blocks']
    num_layers = best_params['num_layers']
    embed_size = best_params['embedding_size']

    # The output dimension based on the one-hot encoded target
    output_dim_activity = len(set(y_activity_train.flatten()))
    print("output_dim_activity:", output_dim_activity)
    output_dim_location = 1  # it is binary
    output_dim_withNOB = 1  # it is binary

    # Suppress warnings during model initialization
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NHiTSModelTuning(num_educationCat=num_educationCat, num_employmentCat=num_employmentCat,
                                 num_genderCat=num_genderCat, num_famTypologyCat=num_famTypologyCat,
                                 num_numFamMembCat=num_numFamMembCat, num_OCCinHHCat=num_OCCinHHCat,
                                 num_seasonCat=num_seasonCat, num_unique_weekCat=num_unique_weekCat,
                                 num_continuous_features=num_continuous_features,
                                 output_dim_activity=output_dim_activity, output_dim_location=output_dim_location,
                                 output_dim_withNOB=output_dim_withNOB,
                                 dropout_embedding=dropout_embedding,
                                 drop_loc=dropout_loc, drop_NOB=dropout_withNOB,
                                 hidden_units=hidden_units, num_blocks=num_blocks, num_layers=num_layers,
                                 embed_size=embed_size)

    # input shape: (season_input, weekend_input, continuous_input), ((48,), (48,), (48, num_continuous_features))
    mps_device = torch.device("mps")
    #mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(mps_device)  # Move model to MPS device
    #print(model)

    # Unpack Tuning hyperparameters
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']

    # Train the model
    history, trained_model =  train_and_evaluate_model_tuning(model,
                                                             X_train, y_activity_train, y_location_train, y_withNOB_train,
                                                             X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                                             optimizer_name,
                                                             epochBase, batch_size, learning_rate,
                                                             device=mps_device,
                                                             w_act=weight_activity, w_loc=weight_location, w_NOB=weight_withNOB,
                                                             checkpoint_path=f"{filename_prefix}_best_modelN_HITS.pth",
                                                             verbose=True, use_tensorboard=True, load_checkpoint=False,
                                                             use_early_stopping=False)

    # TRAIN & EVALUATE: VISUALIZE
    plot_history(history)

    # Optionally: Evaluate the trained model further and save
    # Example: Save the model if it's better than previous models
    #torch.save(trained_model.state_dict(), f"{filename_prefix}_finalN_HITS.pth")
    print("training_afterTuning()_N_HITS is completed")

    # how to open tensorboard recordings
    # tensorboard --logdir=runs/End_to_End_full_embedding # from terminal

    model.load_state_dict(torch.load(f"{filename_prefix}_best_modelN_HITS.pth"))
    evaluate_and_save_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder=label_encoder, device=mps_device, csv_filepath= csv_filepath, arch_type="N_HITS")

    # EVALUATION for model performance for each activity categories ----------------------------------------------------
    # Assume `model`, `X_test`, `y_test`, and `label_encoder` are already defined and the model is trained
    evalACT_classify_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder, device=mps_device, save_csv=True, csv_filepath= csv_filepath, arch_type="N_HITS")

