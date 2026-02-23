# IMPACT ANALYSIS--------------------------------------------------------------------------------------------------------
def impact_analysis(X_train, y_activity_train, y_location_train, y_with_train, target_column='Occupant_Activity'):
    import xgboost as xgb
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Select only the features of interest
    features = ['Region', 'Number Family Members', 'Family Typology', 'Age Classes',
                'Employment status', 'Education Degree', 'Gender', 'Home Type']

    # Extract the columns of interest from X_train
    X_train_selected = X_train[:, :, [2, 3, 4, 5, 6, 7, 8, 9]]  # Adjust indices according to the order of features in X_train

    # Flatten the data for processing
    X_train_flat = X_train_selected.reshape(-1, X_train_selected.shape[2])

    if target_column == 'Occupant_Activity':
        # Flatten the target data
        y_activity_flat = y_activity_train.reshape(-1)

        # Train the model for the activity categories
        model = xgb.XGBClassifier()
        model.fit(X_train_flat, y_activity_flat)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_flat)

        # Identify the top categories that capture 95% of all values in y_activity_train
        y_activity_series = pd.Series(y_activity_flat)
        value_counts = y_activity_series.value_counts(normalize=True)
        cumulative_sum = value_counts.cumsum()
        top_categories = cumulative_sum[cumulative_sum <= 0.95].index
        print("top_categories:", top_categories)

        # Filter the data to include only the top categories
        mask = y_activity_series.isin(top_categories)
        X_train_filtered = X_train_flat[mask]

        # Filter SHAP values for the top categories
        shap_values_filtered = [shap_values[i][mask] for i in range(len(shap_values))]

        # Plot SHAP values for the filtered activity categories
        plt.figure(figsize=(36, 30))  # Adjust the figure size as needed
        shap.summary_plot(shap_values_filtered, X_train_filtered, feature_names=features, plot_type="bar", show=False)
        # Remove the legend manually
        plt.gca().get_legend().remove()
        plt.gcf().subplots_adjust(left=0.35)  # Adjust the left margin to fit the feature names
        plt.show()

    else:
        if target_column == 'location':
            y_target_train = y_location_train
        elif target_column == 'withNOBODY':
            y_target_train = y_with_train
        else:
            raise ValueError(f"Unsupported target column: {target_column}")

        # Flatten the target data
        y_target_flat = y_target_train.reshape(-1)

        # Train the model for the specified target column
        model = xgb.XGBClassifier()
        model.fit(X_train_flat, y_target_flat)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_flat)

        # Plot SHAP values for the specified target column
        shap.summary_plot(shap_values, X_train_flat, feature_names=features)
