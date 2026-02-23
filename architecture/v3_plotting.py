#VISUALIZE--------------------------------------------------------------------------------------------------------------
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

    # Plot Location Loss
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

def plot_history_LessEmbed(history):
    plt.figure(figsize=(18, 12))

    # Activity Accuracy
    train_activity_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_activity_accuracy']]
    valid_activity_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_activity_accuracy']]

    # Activity Loss
    train_activity_loss = history['train_activity_loss']
    valid_activity_loss = history['valid_activity_loss']

    # Plot Activity Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(train_activity_accuracy, label='Train Activity Accuracy')
    plt.plot(valid_activity_accuracy, label='Validation Activity Accuracy')
    plt.title('Activity (Categorical) Accuracy')
    plt.ylabel('Accuracy')
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

    plt.tight_layout()
    plt.show()

def plot_history_NoEmbedSimpler(history):
    plt.figure(figsize=(18, 12))

    # Activity Accuracy
    train_activity_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_activity_accuracy']]
    valid_activity_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_activity_accuracy']]

    # Location Accuracy
    train_location_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_location_accuracy']]
    valid_location_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_location_accuracy']]

    # Activity Loss
    train_activity_loss = history['train_activity_loss']
    valid_activity_loss = history['valid_activity_loss']

    # Location Loss
    train_location_loss = history['train_location_loss']
    valid_location_loss = history['valid_location_loss']

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

    plt.tight_layout()
    plt.show()

def plot_history_NoEmbedSimplest(history):
    plt.figure(figsize=(18, 12))

    # Location Accuracy
    train_location_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['train_location_accuracy']]
    valid_location_accuracy = [t.cpu().numpy() if torch.is_tensor(t) else t for t in history['valid_location_accuracy']]

    # Location Loss
    train_location_loss = history['train_loss']
    valid_location_loss = history['valid_loss']

    # Plot Location Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(train_location_accuracy, label='Train Location Accuracy')
    plt.plot(valid_location_accuracy, label='Validation Location Accuracy')
    plt.title('Location (Binary) Accuracy')
    plt.ylabel(' ')
    plt.xlabel(' ')
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

    plt.tight_layout()
    plt.show()


#VISUALIZE - TUNING--------------------------------------------------------------------------------------------------------------

def plot_hyperparameters_vs_target(num_hidden_layers_list, hidden_units_list, embed_size_list, batch_size_list, learning_rate_list, target_values_list):
    import matplotlib.pyplot as plt
    # Plot for num_hidden_layers
    plt.figure()
    plt.scatter(num_hidden_layers_list, target_values_list)
    plt.xlabel('num_hidden_layers')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('num_hidden_layers vs Target')
    plt.show()

    # Plot for hidden_units (consider flattening nested lists for multiple layers)
    for i in range(len(hidden_units_list[0])):
        hidden_units_i = [units[i] for units in hidden_units_list]
        plt.figure()
        plt.scatter(hidden_units_i, target_values_list)
        plt.xlabel(f'hidden_units_layer_{i+1}')
        plt.ylabel('Target (e.g., weighted_loss)')
        plt.title(f'hidden_units_layer_{i+1} vs Target')
        plt.show()

    # Plot for embed_size
    plt.figure()
    plt.scatter(embed_size_list, target_values_list)
    plt.xlabel('embed_size')
    plt.ylabel('Target (e.g., weighted_loss)')
    plt.title('embed_size vs Target')
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
    fig.show()
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
        sns.kdeplot(x=param_values, y=target_values, fill=True, cmap='Blues', thresh=0.1)
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
        plt.scatter(unique_points[:, 0], unique_points[:, 1], s=counts * 50, alpha=0.6, edgecolor='black')
        plt.colorbar(label='Target (e.g., weighted_loss)')
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Bubble Plot: {param1} & {param2} vs Target')
        plt.savefig(f'{folder_name}/{filename_prefix}_{param1}_{param2}_bubble_vs_target.png')
        plt.close()

    print(f"Bubble plots saved in folder: {folder_name} with prefix: {filename_prefix}")