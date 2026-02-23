
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
import torch

# AFTER TUNING EVALUATE AND SAVE ---------------------------------------------------------------------------------------
def evaluate_and_save_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder, device, csv_filepath, arch_type):

    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the new filename with the model predictions suffix
    file_name = f"{filename_prefix}_model_predictions_{arch_type}.csv"

    # Replace these indices with the correct indices for according to exisiting data
    educationDegree_idx = 0
    employmentStatus_idx = 1
    gender_idx = 2
    famTypology_idx = 3
    numFamMembers_idx = 4
    OCCinHH_idx = 5
    season_idx = 6
    weekend_idx = 7
    num_categorical_features = 8  # Update this to the total number of categorical features

    test_dataset = TensorDataset(
        torch.tensor(X_test[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, numFamMembers_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OCCinHH_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, season_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, num_categorical_features:], dtype=torch.float),  # Continuous data
        torch.tensor(y_activity_test, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location_test, dtype=torch.float),  # Location labels as floats (binary classification)
        torch.tensor(y_withNOB_test, dtype=torch.float),  # withNOBODY labels as floats (binary classification)
    )

    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, drop_last=False)

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
            activity_output, location_output, withNOB_output = model(*features)
            _, predicted_activity = torch.max(activity_output, 2)
            predicted_location = torch.round(torch.sigmoid(location_output)).int()
            predicted_withNOB = torch.round(torch.sigmoid(withNOB_output)).int()

            # Append to lists
            all_activity_preds.extend(predicted_activity.view(-1).cpu().numpy())
            all_location_preds.extend(predicted_location.view(-1).cpu().numpy())
            all_withNOB_preds.extend(predicted_withNOB.view(-1).cpu().numpy())

            all_activity_actuals.extend(activity_target.view(-1).cpu().numpy())
            all_location_actuals.extend(location_target.view(-1).cpu().numpy())
            all_withNOB_actuals.extend(withNOB_target.view(-1).cpu().numpy())

    # Inverse transform to get categories from label encoded activity predictions
    all_actuals_categories = label_encoder.inverse_transform(all_activity_actuals)
    all_activity_preds_categories = label_encoder.inverse_transform(all_activity_preds)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Actual Activity': all_actuals_categories,
        'Predicted Activity Category': all_activity_preds_categories,
        'Actual Location': all_location_actuals,
        'Predicted Location': all_location_preds,
        'Actual withNOB': all_withNOB_actuals,
        'Predicted withNOB': all_withNOB_preds,
    })
    results_df.to_csv(file_name, index=False)
    print(f'Results saved to {file_name}')
    print(f"evaluate_and_save_afterTuning_{arch_type} is completed")
    return results_df  # Optional: return the results dataframe

# EVALUATION for model performance for each activity categories ----------------------------------------------------
def evalACT_classify_afterTuning(model, X_test, y_activity_test, y_location_test, y_withNOB_test, label_encoder, device, csv_filepath, arch_type,save_csv=False,):
    from sklearn.metrics import classification_report, confusion_matrix

    filename_prefix = os.path.splitext(os.path.basename(csv_filepath))[0]
    # Create the new filename with the model predictions suffix
    file_name = f"{filename_prefix}_classificationReport_{arch_type}.csv"

    # Replace these indices with the correct indices for according to exisiting data
    educationDegree_idx = 0
    employmentStatus_idx = 1
    gender_idx = 2
    famTypology_idx = 3
    numFamMembers_idx = 4
    OCCinHH_idx = 5
    season_idx = 6
    weekend_idx = 7
    num_categorical_features = 8  # Update this to the total number of categorical features

    test_dataset = TensorDataset(
        torch.tensor(X_test[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, numFamMembers_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, OCCinHH_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, season_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X_test[:, :, num_categorical_features:], dtype=torch.float),  # Continuous data
        torch.tensor(y_activity_test, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location_test, dtype=torch.float),  # Location labels as floats (binary classification)
        torch.tensor(y_withNOB_test, dtype=torch.float),  # withNOBODY labels as floats (binary classification)
    )

    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, drop_last=False)

    all_activity_preds = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # Unpack the data
            *features, activity_target, location_target, withNOB_target = [d.to(device) for d in data]
            # Make predictions
            activity_output, location_output, withNOB_output = model(*features)
            _, predicted_activity = torch.max(activity_output, 2)
            all_activity_preds.extend(predicted_activity.view(-1).cpu().numpy())

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

    #print("Classification Report:\n", class_report_df)
    #print("Confusion Matrix:\n", conf_matrix)

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
