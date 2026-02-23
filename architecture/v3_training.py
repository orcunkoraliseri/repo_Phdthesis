# TRAIN & EVALUATE: CHECKING -------------------------------------------------------------------------------------------

def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            print(f'Gradient norm for {name}: {grad_norm}')
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f'Gradient for {name}: mean={param.grad.mean().item()}, std={param.grad.std().item()}')

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

# BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE BASE
# TRAIN & EVALUATE------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def create_dataloader(X, y_activity, y_location, y_withNOB, batch_size, shuffle=False, drop_last=False):
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

    dataset = TensorDataset(
        torch.tensor(X[:, :, educationDegree_idx], dtype=torch.long),
        torch.tensor(X[:, :, employmentStatus_idx], dtype=torch.long),
        torch.tensor(X[:, :, gender_idx], dtype=torch.long),
        torch.tensor(X[:, :, famTypology_idx], dtype=torch.long),
        torch.tensor(X[:, :, numFamMembers_idx], dtype=torch.long),
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
    correct_location_predictions = 0
    correct_withNOB_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0
    total_withNOB_predictions = 0

    for batch_idx, data in enumerate(data_loader):
        try:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            education_data, employment_data, gender_data, famTypology_data, \
            numFamMemb_data, OCCinHH_data, season_data, \
            weekend_data, continuous_data, activity_target, location_target, withNOB_target = \
            data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), \
            data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device), \
            data[8].to(device), data[9].to(device), data[10].to(device), data[11].to(device)

            opt.zero_grad()
            # Model's forward pass
            activity_output, location_output, withNOB_output = model(education_data, employment_data, gender_data, famTypology_data, \
            numFamMemb_data, OCCinHH_data, season_data, weekend_data, continuous_data,)

            # Compute the loss for both outputs
            loss_act = cr_act(activity_output.view(-1, outDimAct), activity_target.view(-1)) # criterion_activity
            loss_loc = cr_loc(location_output.view(-1), location_target.view(-1).float()) # criterion_location
            loss_NOB = cr_NOB(withNOB_output.view(-1), withNOB_target.view(-1).float()) # criterion_withNOBODY

            # Normalize the weights
            total_w = w_act + w_loc + w_NOB # calculation of total weight
            w_act /= total_w # w_act is weight_activity
            w_loc /= total_w # w_loc is weight_location
            w_NOB /= total_w # w_NOB is weight_withNOBODY

            total_loss = (w_act * loss_act + w_loc * loss_loc + w_NOB * loss_NOB)
            total_loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

            # Calculate "withNOBODY" accuracy
            predicted_withNOB_classes = torch.sigmoid(withNOB_output.view(-1)) > 0.5
            correct_withNOB_predictions += (predicted_withNOB_classes == withNOB_target.view(-1).float()).float().sum()
            total_withNOB_predictions += withNOB_target.numel()

        except Exception as e:
            # Handle errors during the training loop
            print(f"An error occurred during training at batch {batch_idx}: {e}")
            continue  # Skip this batch and continue with the next

    # Avoid division by zero
    if total_activity_predictions == 0 or total_location_predictions == 0 or total_withNOB_predictions==0:
        raise ValueError("No predictions made for activity or location - check your data and model outputs.")

    # Check gradients
    #check_gradients(model)

    # Average loss and accuracy
    train_loss_avg = train_loss / len(data_loader.dataset)
    train_activity_loss = train_activity_loss / len(data_loader.dataset)
    train_location_loss = train_location_loss / len(data_loader.dataset)
    train_withNOB_loss = train_withNOB_loss / len(data_loader.dataset)

    train_activity_accuracy = correct_activity_predictions / total_activity_predictions
    train_location_accuracy = correct_location_predictions / total_location_predictions
    train_withNOB_accuracy = correct_withNOB_predictions / total_withNOB_predictions

    return train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy

def validate_model(model, data_loader, cr_act, cr_loc, cr_NOB, device, outDimAct, w_act, w_loc, w_NOB):
    model.eval()
    valid_loss = 0.0
    valid_activity_loss = 0.0
    valid_location_loss = 0.0
    valid_withNOB_loss = 0.0

    correct_activity_predictions = 0
    correct_location_predictions = 0
    correct_withNOB_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0
    total_withNOB_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            education_data, employment_data, gender_data, famTypology_data, \
            numFamMemb_data,  OCCinHH_data, season_data, \
            weekend_data, continuous_data, activity_target, location_target, withNOB_target = \
            data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), \
            data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device), \
            data[8].to(device), data[9].to(device), data[10].to(device), data[11].to(device)

            activity_output, location_output, withNOB_output = model(education_data, employment_data, gender_data, famTypology_data, \
            numFamMemb_data,  OCCinHH_data, season_data, weekend_data, continuous_data,)

            loss_activity = cr_act(activity_output.view(-1, outDimAct), activity_target.view(-1))   # criterion_activity, output_dim_activity
            loss_location = cr_loc(location_output.view(-1), location_target.view(-1).float())      # criterion_location
            loss_withNOB = cr_NOB(withNOB_output.view(-1), withNOB_target.view(-1).float())         # criterion_withNOB

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
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

            # Calculate "withNOBODY" accuracy
            predicted_withNOB_classes = torch.sigmoid(withNOB_output.view(-1)) > 0.5
            correct_withNOB_predictions += (predicted_withNOB_classes == withNOB_target.view(-1).float()).float().sum()
            total_withNOB_predictions += withNOB_target.numel()

    valid_loss_avg = valid_loss / len(data_loader.dataset)
    valid_activity_loss = valid_activity_loss / len(data_loader.dataset)
    valid_location_loss = valid_location_loss / len(data_loader.dataset)
    valid_withNOB_loss = valid_withNOB_loss / len(data_loader.dataset)

    valid_activity_accuracy = correct_activity_predictions / total_activity_predictions
    valid_location_accuracy = correct_location_predictions / total_location_predictions
    valid_withNOB_accuracy = correct_withNOB_predictions / total_withNOB_predictions

    return valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy

def train_and_evaluate_modelBase(model,
                                 X_train, y_activity_train, y_location_train, y_withNOB_train,
                                 X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                 device, w_act=1, w_loc=1, w_NOB=1,
                                 epochs=100, batch_size=48, learning_rate=0.01,
                                 checkpoint_path='best_model_base_training.pth'):

    output_dim_activity = len(set(y_activity_train.flatten()))

    # Create data loaders
    train_loader = create_dataloader(X_train, y_activity_train, y_location_train, y_withNOB_train,  batch_size, shuffle=True,
                                     drop_last=False)
    valid_loader = create_dataloader(X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, batch_size, shuffle=False,
                                     drop_last=False)

    # Define loss functions for each task
    criterion_activity = torch.nn.CrossEntropyLoss()
    criterion_location = torch.nn.BCEWithLogitsLoss()
    criterion_withNOB = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #default LR: 0.001

    history = {'train_loss': [], 'valid_loss': [],
               'train_activity_loss': [], 'train_location_loss': [],'train_withNOB_loss': [],
               'valid_activity_loss': [], 'valid_location_loss': [], 'valid_withNOB_loss': [],
               'train_activity_accuracy': [],'train_location_accuracy': [], 'train_withNOB_accuracy': [],
               'valid_activity_accuracy': [], 'valid_location_accuracy': [],  'valid_withNOB_accuracy': []}
    best_valid_accuracy = 0.0

    for epoch in range(epochs):
        train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, \
            train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy  = train_model(model, train_loader,
            criterion_activity, criterion_location, criterion_withNOB,
            optimizer, device, output_dim_activity, w_act=w_act, w_loc=w_loc, w_NOB=w_NOB)
        valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, \
            valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy = validate_model(
            model, valid_loader, criterion_activity, criterion_location, criterion_withNOB,
            device, output_dim_activity, w_act=w_act, w_loc=w_loc, w_NOB=w_NOB)

        history['train_loss'].append(train_loss_avg)
        history['train_activity_loss'].append(train_activity_loss)
        history['train_location_loss'].append(train_location_loss)
        history['train_withNOB_loss'].append(train_withNOB_loss)

        history['valid_loss'].append(valid_loss_avg)
        history['valid_activity_loss'].append(valid_activity_loss)
        history['valid_location_loss'].append(valid_location_loss)
        history['valid_withNOB_loss'].append(valid_withNOB_loss)

        history['train_activity_accuracy'].append(train_activity_accuracy.item())
        history['valid_activity_accuracy'].append(valid_activity_accuracy.item())

        history['train_location_accuracy'].append(train_location_accuracy.item())
        history['valid_location_accuracy'].append(valid_location_accuracy.item())

        history['train_withNOB_accuracy'].append(train_withNOB_accuracy.item())
        history['valid_withNOB_accuracy'].append(valid_withNOB_accuracy.item())

        print(f'Epoch {epoch}: '
              f'Val_Act_Acc: {valid_activity_accuracy:.4f}, Val_Loc_Acc_: {valid_location_accuracy:.4f}, Val_withNOB_Acc: {valid_withNOB_accuracy:.4f}, '
              f'Train_Act_Acc: {train_activity_accuracy:.4f}, Train_Loc_Acc: {train_location_accuracy:.4f}, Train_withNOB_Acc: {train_withNOB_accuracy:.4f}, '
              f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},' 
              f'Train Act Loss: {train_activity_loss:.4f}, Train Loc Loss: {train_location_loss:.4f}, Train withNOB Loss: {train_withNOB_loss:.4f},' 
              f'Val Act Loss: {valid_activity_loss:.4f}, Val Loc Loss: {valid_location_loss:.4f},  Val withNOB Loss: {valid_withNOB_loss:.4f},')

        # Check if the model's validation accuracy for activity is improved
        if valid_activity_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_activity_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        """ Close during tuning"""
        # Check if the accuracy threshold is met and stop training if it is: PREVENT MEMORIZATION
        accuracy_threshold = 0.95
        if best_valid_accuracy >= accuracy_threshold:
            print(f'Early stopping triggered at epoch {epoch} due to reaching accuracy threshold of {accuracy_threshold:.2f}')
            break

    return history, model

# TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING TUNING
# TUNING: TRAIN & EVALUATE----------------------------------------------------------------------------------------------
def train_and_evaluate_model_tuning(model,
                                 X_train, y_activity_train, y_location_train, y_withNOB_train,
                                 X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                 optimizer_name,
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
    train_loader = create_dataloader(X_train, y_activity_train, y_location_train, y_withNOB_train,  batch_size, shuffle=True,
                                     drop_last=False)
    valid_loader = create_dataloader(X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, batch_size, shuffle=False,
                                     drop_last=False)

    # Define loss functions for each task
    cr_act = torch.nn.CrossEntropyLoss()
    cr_loc = torch.nn.BCEWithLogitsLoss()
    cr_NOB = torch.nn.BCEWithLogitsLoss()

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Optional learning rate scheduler
    if use_scheduler:
        import torch.optim.lr_scheduler as lr_scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
        history = {'train_loss': [], 'valid_loss': [],
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
    min_delta = 0.001 # Minimum change to qualify as an improvement

    for epoch in range(start_epoch, epochs):
        if record_memory_usage:
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 ** 2)
            memory_usages.append(memory_usage_mb)
            memory_log_file.write(f'Epoch {epoch} - Memory Usage: {memory_usage_mb} MB\n')

        train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, \
            train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy  = train_model(model, train_loader,
            cr_act, cr_loc, cr_NOB,
            optimizer, device, outDimAct, w_act=w_act, w_loc=w_loc, w_NOB=w_NOB)
        valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, \
            valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy = validate_model(
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

        history['train_activity_accuracy'].append(train_activity_accuracy.item())
        history['valid_activity_accuracy'].append(valid_activity_accuracy.item())

        history['train_location_accuracy'].append(train_location_accuracy.item())
        history['valid_location_accuracy'].append(valid_location_accuracy.item())

        history['train_withNOB_accuracy'].append(train_withNOB_accuracy.item())
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
                'Accuracy/train_activity': train_activity_accuracy.item(),
                'Accuracy/valid_activity': valid_activity_accuracy.item(),
                'Accuracy/train_location': train_location_accuracy.item(),
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
                      f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},'
                      f'Train Act Loss: {train_activity_loss:.4f}, Train Loc Loss: {train_location_loss:.4f}, Train withNOB Loss: {train_withNOB_loss:.4f},'
                      f'Val Act Loss: {valid_activity_loss:.4f}, Val Loc Loss: {valid_location_loss:.4f},  Val withNOB Loss: {valid_withNOB_loss:.4f},')

        # Conditionally print output
        if verbose:
            print(f'Epoch {epoch}: '
                  f'Val_Act_Acc: {valid_activity_accuracy:.4f}, Val_Loc_Acc_: {valid_location_accuracy:.4f}, Val_withNOB_Acc: {valid_withNOB_accuracy:.4f}, '
                  f'Train_Act_Acc: {train_activity_accuracy:.4f}, Train_Loc_Acc: {train_location_accuracy:.4f}, Train_withNOB_Acc: {train_withNOB_accuracy:.4f}, '
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

            # Additional early stopping condition: Accuracy threshold
            accuracy_threshold = 0.95
            if best_valid_accuracy >= accuracy_threshold:
                print(f'Early stopping triggered at epoch {epoch} due to reaching accuracy threshold of {accuracy_threshold:.2f}')
                break

    if use_tensorboard:
        # Close the TensorBoard writer
        writer.close()

    if record_memory_usage:
        average_memory_usage = sum(memory_usages) / len(memory_usages)
        memory_log_file.write(f'Average Memory Usage: {average_memory_usage} MB\n')
        print("memory_usage is recorded in the .txt file")
        memory_log_file.close()

    return history, model

# NO_EMBEDDING: TRAIN & EVALUATE----------------------------------------------------------------------------------------
def create_dataloaderNoEmbed(X, y_activity, y_location, y_withNOB, batch_size, shuffle=False, drop_last=False):
    # Assuming X is now entirely continuous features
    #print(X.shape)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float),  # Continuous data
        torch.tensor(y_activity, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location, dtype=torch.float),  # Location labels as floats (binary classification)
        torch.tensor(y_withNOB, dtype=torch.float),  # withNOBODY labels as floats (binary classification)
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_modelNoEmbed(model, data_loader, criterion_activity, criterion_location, criterion_withNOB, optimizer, device, output_dim_activity):
    model.train()

    train_loss = 0.0
    train_activity_loss = 0.0
    train_location_loss = 0.0
    train_withNOB_loss = 0.0

    correct_activity_predictions = 0
    correct_location_predictions = 0
    correct_withNOB_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0
    total_withNOB_predictions = 0

    for batch_idx, data in enumerate(data_loader):
        try:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            continuous_data, activity_target, location_target, withNOB_target = \
            data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

            optimizer.zero_grad()
            # Model's forward pass
            activity_output, location_output, withNOB_output = model(continuous_data)

            # Compute the loss for both outputs
            loss_activity = criterion_activity(activity_output.view(-1, output_dim_activity), activity_target.view(-1))
            loss_location = criterion_location(location_output.view(-1), location_target.view(-1).float())
            loss_withNOB = criterion_withNOB(withNOB_output.view(-1), withNOB_target.view(-1).float())

            total_loss = loss_activity + loss_location + loss_withNOB
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += total_loss.item()
            train_activity_loss += loss_activity.item()
            train_location_loss += loss_location.item()
            train_withNOB_loss += loss_withNOB.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

            # Calculate "withNOBODY" accuracy
            predicted_withNOB_classes = torch.sigmoid(withNOB_output.view(-1)) > 0.5
            correct_withNOB_predictions += (predicted_withNOB_classes == withNOB_target.view(-1).float()).float().sum()
            total_withNOB_predictions += withNOB_target.numel()

        except Exception as e:
            # Handle errors during the training loop
            print(f"An error occurred during training at batch {batch_idx}: {e}")
            continue  # Skip this batch and continue with the next

    # Avoid division by zero
    if total_activity_predictions == 0 or total_location_predictions == 0 or total_withNOB_predictions==0:
        raise ValueError("No predictions made for activity or location - check your data and model outputs.")

    #Check gradients
    #check_gradients(model)

    # Average loss and accuracy
    train_loss_avg = train_loss / len(data_loader.dataset)
    train_activity_loss = train_activity_loss / len(data_loader.dataset)
    train_location_loss = train_location_loss / len(data_loader.dataset)
    train_withNOB_loss = train_withNOB_loss / len(data_loader.dataset)

    train_activity_accuracy = correct_activity_predictions / total_activity_predictions
    train_location_accuracy = correct_location_predictions / total_location_predictions
    train_withNOB_accuracy = correct_withNOB_predictions / total_withNOB_predictions

    return train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy

def validate_modelNoEmbed(model, data_loader, criterion_activity, criterion_location, criterion_withNOB, device, output_dim_activity):
    model.eval()
    valid_loss = 0.0
    valid_activity_loss = 0.0
    valid_location_loss = 0.0
    valid_withNOB_loss = 0.0

    correct_activity_predictions = 0
    correct_location_predictions = 0
    correct_withNOB_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0
    total_withNOB_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            continuous_data, activity_target, location_target, withNOB_target = \
            data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

            activity_output, location_output, withNOB_output = model(continuous_data)

            loss_activity = criterion_activity(activity_output.view(-1, output_dim_activity), activity_target.view(-1))
            loss_location = criterion_location(location_output.view(-1), location_target.view(-1).float())
            loss_withNOB = criterion_withNOB(withNOB_output.view(-1), withNOB_target.view(-1).float())

            valid_loss += loss_activity.item() + loss_location.item() + loss_withNOB.item()
            valid_activity_loss += loss_activity.item()
            valid_location_loss += loss_location.item()
            valid_withNOB_loss += loss_withNOB.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

            # Calculate "withNOBODY" accuracy
            predicted_withNOB_classes = torch.sigmoid(withNOB_output.view(-1)) > 0.5
            correct_withNOB_predictions += (predicted_withNOB_classes == withNOB_target.view(-1).float()).float().sum()
            total_withNOB_predictions += withNOB_target.numel()

    valid_loss_avg = valid_loss / len(data_loader.dataset)
    valid_activity_loss = valid_activity_loss / len(data_loader.dataset)
    valid_location_loss = valid_location_loss / len(data_loader.dataset)
    valid_withNOB_loss = valid_withNOB_loss / len(data_loader.dataset)

    valid_activity_accuracy = correct_activity_predictions / total_activity_predictions
    valid_location_accuracy = correct_location_predictions / total_location_predictions
    valid_withNOB_accuracy = correct_withNOB_predictions / total_withNOB_predictions

    return valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy

def train_and_evaluate_modelBaseNoEmbed(model,
                                 X_train, y_activity_train, y_location_train, y_withNOB_train,
                                 X_valid, y_activity_valid, y_location_valid, y_withNOB_valid,
                                 device,
                                 epochs=100, batch_size=48, learning_rate=0.01,
                                 checkpoint_path='best_modelNoEmbed.pth'):

    output_dim_activity = len(set(y_activity_train.flatten()))

    # Create data loaders
    train_loader = create_dataloaderNoEmbed(X_train, y_activity_train, y_location_train, y_withNOB_train,  batch_size, shuffle=True,
                                     drop_last=False)
    valid_loader = create_dataloaderNoEmbed(X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, batch_size, shuffle=False,
                                     drop_last=False)


    # Define loss functions for each task
    criterion_activity = torch.nn.CrossEntropyLoss()
    criterion_location = torch.nn.BCEWithLogitsLoss()
    criterion_withNOB = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #default LR: 0.001

    history = {'train_loss': [], 'valid_loss': [],
               'train_activity_loss': [], 'train_location_loss': [],'train_withNOB_loss': [],
               'valid_activity_loss': [], 'valid_location_loss': [], 'valid_withNOB_loss': [],
               'train_activity_accuracy': [],'train_location_accuracy': [], 'train_withNOB_accuracy': [],
               'valid_activity_accuracy': [], 'valid_location_accuracy': [],  'valid_withNOB_accuracy': []}
    best_valid_accuracy = 0.0

    for epoch in range(epochs):
        train_loss_avg, train_activity_loss, train_location_loss, train_withNOB_loss, \
            train_activity_accuracy, train_location_accuracy, train_withNOB_accuracy  = train_modelNoEmbed(model, train_loader,
            criterion_activity, criterion_location, criterion_withNOB,
            optimizer, device, output_dim_activity)
        valid_loss_avg, valid_activity_loss, valid_location_loss, valid_withNOB_loss, valid_activity_accuracy, valid_location_accuracy, valid_withNOB_accuracy = validate_modelNoEmbed(
            model, valid_loader, criterion_activity, criterion_location, criterion_withNOB, device, output_dim_activity)

        history['train_loss'].append(train_loss_avg)
        history['train_activity_loss'].append(train_activity_loss)
        history['train_location_loss'].append(train_location_loss)
        history['train_withNOB_loss'].append(train_withNOB_loss)

        history['valid_loss'].append(valid_loss_avg)
        history['valid_activity_loss'].append(valid_activity_loss)
        history['valid_location_loss'].append(valid_location_loss)
        history['valid_withNOB_loss'].append(valid_withNOB_loss)

        history['train_activity_accuracy'].append(train_activity_accuracy.item())
        history['valid_activity_accuracy'].append(valid_activity_accuracy.item())

        history['train_location_accuracy'].append(train_location_accuracy.item())
        history['valid_location_accuracy'].append(valid_location_accuracy.item())

        history['train_withNOB_accuracy'].append(train_withNOB_accuracy.item())
        history['valid_withNOB_accuracy'].append(valid_withNOB_accuracy.item())

        print(f'Epoch {epoch}: '
              f'Val_Act_Acc: {valid_activity_accuracy:.4f}, Val_Loc_Acc_: {valid_location_accuracy:.4f}, Val_withNOB_Acc: {valid_withNOB_accuracy:.4f}, '
              f'Train_Act_Acc: {train_activity_accuracy:.4f}, Train_Loc_Acc: {train_location_accuracy:.4f}, Train_withNOB_Acc: {train_withNOB_accuracy:.4f}, '
              f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},' 
              f'Train Act Loss: {train_activity_loss:.4f}, Train Loc Loss: {train_location_loss:.4f}, Train withNOB Loss: {train_withNOB_loss:.4f},' 
              f'Val Act Loss: {valid_activity_loss:.4f}, Val Loc Loss: {valid_location_loss:.4f},  Val withNOB Loss: {valid_withNOB_loss:.4f},')

        # Check if the model's validation accuracy for activity is improved
        if valid_activity_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_activity_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return history, model



# NO_EMBED_SIMPLER: TRAIN & EVALUATE------------------------------------------------------------------------------------
def loader_NoEmbed_Simpler(X, y_activity, y_location, batch_size, shuffle=False, drop_last=False):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float),  # Continuous data
        torch.tensor(y_activity, dtype=torch.long),  # Activity labels as long integers
        torch.tensor(y_location, dtype=torch.float),  # Location labels as floats (binary classification)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_NoEmbed_Simpler(model, data_loader, criterion_activity, criterion_location, optimizer, device, output_dim_activity):
    model.train()

    train_loss = 0.0
    train_activity_loss = 0.0
    train_location_loss = 0.0

    correct_activity_predictions = 0
    correct_location_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0

    for batch_idx, data in enumerate(data_loader):
        try:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            continuous_data, activity_target, location_target= \
            data[0].to(device), data[1].to(device), data[2].to(device)

            optimizer.zero_grad()
            # Model's forward pass
            activity_output, location_output= model(continuous_data)

            # Compute the loss for both outputs
            loss_activity = criterion_activity(activity_output.view(-1, output_dim_activity), activity_target.view(-1))
            loss_location = criterion_location(location_output.view(-1), location_target.view(-1).float())

            total_loss = loss_activity + loss_location
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_activity_loss += loss_activity.item()
            train_location_loss += loss_location.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

        except Exception as e:
            # Handle errors during the training loop
            print(f"An error occurred during training at batch {batch_idx}: {e}")
            continue  # Skip this batch and continue with the next

    #Check gradients
    #check_gradients(model)

    # Average loss and accuracy
    train_loss_avg = train_loss / len(data_loader.dataset)
    train_activity_loss = train_activity_loss / len(data_loader.dataset)
    train_location_loss = train_location_loss / len(data_loader.dataset)

    train_activity_accuracy = correct_activity_predictions / total_activity_predictions
    train_location_accuracy = correct_location_predictions / total_location_predictions

    return train_loss_avg, train_activity_loss, train_location_loss, train_activity_accuracy, train_location_accuracy,

def validate_NoEmbed_Simpler(model, data_loader, criterion_activity, criterion_location, device, output_dim_activity):
    model.eval()
    valid_loss = 0.0
    valid_activity_loss = 0.0
    valid_location_loss = 0.0

    correct_activity_predictions = 0
    correct_location_predictions = 0

    total_activity_predictions = 0
    total_location_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            continuous_data, activity_target, location_target= \
            data[0].to(device), data[1].to(device), data[2].to(device)

            activity_output, location_output = model(continuous_data)

            loss_activity = criterion_activity(activity_output.view(-1, output_dim_activity), activity_target.view(-1))
            loss_location = criterion_location(location_output.view(-1), location_target.view(-1).float())

            valid_loss += loss_activity.item() + loss_location.item()
            valid_activity_loss += loss_activity.item()
            valid_location_loss += loss_location.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

    valid_loss_avg = valid_loss / len(data_loader.dataset)
    valid_activity_loss = valid_activity_loss / len(data_loader.dataset)
    valid_location_loss = valid_location_loss / len(data_loader.dataset)

    valid_activity_accuracy = correct_activity_predictions / total_activity_predictions
    valid_location_accuracy = correct_location_predictions / total_location_predictions

    return valid_loss_avg, valid_activity_loss, valid_location_loss, valid_activity_accuracy, valid_location_accuracy

def trainEvalNoEmbed_Simpler(model,
                                 X_train, y_activity_train, y_location_train,
                                 X_valid, y_activity_valid, y_location_valid,
                                 device,
                                 epochs=100, batch_size=48, learning_rate=0.01,
                                 checkpoint_path='best_NoEmbed_Simpler.pth'):

    output_dim_activity = len(set(y_activity_train.flatten()))

    # Create data loaders
    train_loader = loader_NoEmbed_Simpler(X_train, y_activity_train, y_location_train, batch_size, shuffle=True,
                                     drop_last=True)
    valid_loader = loader_NoEmbed_Simpler(X_valid, y_activity_valid, y_location_valid, batch_size, shuffle=False,
                                     drop_last=True)

    # Define loss functions for each task
    criterion_activity = torch.nn.CrossEntropyLoss()
    criterion_location = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #default LR: 0.001

    history = {'train_loss': [], 'valid_loss': [],
               'train_activity_loss': [], 'train_location_loss': [],
               'valid_activity_loss': [], 'valid_location_loss': [],
               'train_activity_accuracy': [],'train_location_accuracy': [],
               'valid_activity_accuracy': [], 'valid_location_accuracy': [],}
    best_valid_accuracy = 0.0

    for epoch in range(epochs):
        train_loss_avg, train_activity_loss, train_location_loss,train_activity_accuracy, train_location_accuracy = train_NoEmbed_Simpler(
            model, train_loader, criterion_activity, criterion_location, optimizer, device, output_dim_activity)
        valid_loss_avg, valid_activity_loss, valid_location_loss, valid_activity_accuracy, valid_location_accuracy = validate_NoEmbed_Simpler(
            model, valid_loader, criterion_activity, criterion_location, device, output_dim_activity)

        history['train_loss'].append(train_loss_avg)
        history['train_activity_loss'].append(train_activity_loss)
        history['train_location_loss'].append(train_location_loss)

        history['valid_loss'].append(valid_loss_avg)
        history['valid_activity_loss'].append(valid_activity_loss)
        history['valid_location_loss'].append(valid_location_loss)

        history['train_activity_accuracy'].append(train_activity_accuracy.item())
        history['valid_activity_accuracy'].append(valid_activity_accuracy.item())

        history['train_location_accuracy'].append(train_location_accuracy.item())
        history['valid_location_accuracy'].append(valid_location_accuracy.item())

        print(f'Epoch {epoch}: '
              f'Val_Act_Acc: {valid_activity_accuracy:.4f}, Val_Loc_Acc_: {valid_location_accuracy:.4f},  '
              f'Train_Act_Acc: {train_activity_accuracy:.4f}, Train_Loc_Acc: {train_location_accuracy:.4f}, '
              f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},' 
              f'Train Act Loss: {train_activity_loss:.4f}, Train Loc Loss: {train_location_loss:.4f},' 
              f'Val Act Loss: {valid_activity_loss:.4f}, Val Loc Loss: {valid_location_loss:.4f},')

        # Check if the model's validation accuracy for activity is improved
        if valid_activity_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_activity_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return history, model



# NO_EMBED_SIMPLEST: TRAIN & EVALUATE-----------------------------------------------------------------------------------
def loader_NoEmbed_Simplest(X, y_location, batch_size, shuffle=False, drop_last=False):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float),  # Continuous data
        torch.tensor(y_location, dtype=torch.float),  # Location labels as floats (binary classification)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_NoEmbed_Simplest(model, data_loader, criterion_location, optimizer, device):
    model.train()

    train_location_loss = 0.0
    correct_location_predictions = 0
    total_location_predictions = 0

    for batch_idx, data in enumerate(data_loader):
        try:
            # Unpack all the data from the dataloader
            continuous_data, location_target= data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # Model's forward pass
            location_output= model(continuous_data)

            # Compute the loss for both outputs
            loss_location = criterion_location(location_output.view(-1), location_target.view(-1).float())
            loss_location.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_location_loss += loss_location.item()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()


        except Exception as e:
            # Handle errors during the training loop
            print(f"An error occurred during training at batch {batch_idx}: {e}")
            continue  # Skip this batch and continue with the next

    # In your training loop
    #monitor_gradients(model) # only print "mean"
    #check_gradients(model) # print "mean" & "std"

    # Average loss and accuracy
    train_location_loss = train_location_loss / len(data_loader.dataset)
    train_location_accuracy = correct_location_predictions / total_location_predictions

    return train_location_loss, train_location_accuracy,

def validate_NoEmbed_Simplest(model, data_loader, criterion_location, device):
    model.eval()
    valid_location_loss = 0.0
    correct_location_predictions = 0
    total_location_predictions = 0
    with torch.no_grad():
        for data in data_loader:
            # Unpack all the data from the dataloader
            continuous_data, location_target= \
            data[0].to(device), data[1].to(device)

            location_output = model(continuous_data)

            loss_location = criterion_location(location_output.view(-1), location_target.view(-1).float())

            valid_location_loss += loss_location.item()

            # Calculate location accuracy
            predicted_location_classes = torch.sigmoid(location_output.view(-1)) > 0.5
            correct_location_predictions += (predicted_location_classes == location_target.view(-1).float()).float().sum()
            total_location_predictions += location_target.numel()

    valid_location_loss_avg = valid_location_loss / len(data_loader.dataset)
    valid_location_accuracy = correct_location_predictions / total_location_predictions

    return valid_location_loss_avg, valid_location_accuracy

def trainEvalNoEmbed_Simplest(model,
                                 X_train, y_location_train,
                                 X_valid, y_location_valid,
                                 device,
                                 epochs=100, batch_size=48, learning_rate=0.01,
                                 checkpoint_path='best_NoEmbed_Simplest.pth'):

    # Create data loaders
    train_loader = loader_NoEmbed_Simplest(X_train, y_location_train, batch_size, shuffle=True, drop_last=True)
    valid_loader = loader_NoEmbed_Simplest(X_valid, y_location_valid, batch_size, shuffle=False, drop_last=True)

    # Define loss functions for each task
    criterion_location = torch.nn.BCEWithLogitsLoss()
    #criterion_location = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #default LR: 0.001

    history = {'train_loss': [], 'valid_loss': [],
               'train_location_loss': [], 'valid_location_loss': [],
               'train_location_accuracy': [], 'valid_location_accuracy': [],}
    best_valid_accuracy = 0.0

    for epoch in range(epochs):
        train_location_loss, train_location_accuracy = train_NoEmbed_Simplest(model, train_loader, criterion_location, optimizer, device)
        valid_location_loss_avg, valid_location_accuracy = validate_NoEmbed_Simplest(model, valid_loader, criterion_location, device,)

        history['train_loss'].append(train_location_loss)
        history['valid_loss'].append(valid_location_loss_avg)
        history['train_location_accuracy'].append(train_location_accuracy.item())
        history['valid_location_accuracy'].append(valid_location_accuracy.item())

        print(f'Epoch {epoch}: '
              f'Val_Loc_Acc_: {valid_location_accuracy:.4f},'
              f'Train_Loc_Acc: {train_location_accuracy:.4f},'
              f'Train Loss: {train_location_loss:.4f},'
              f'Valid Loss: {valid_location_loss_avg:.4f}')

        # Check if the model's validation accuracy for activity is improved
        if valid_location_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_location_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return history, model
# LESS_EMBED_SIMPLEST: TRAIN & EVALUATE-----------------------------------------------------------------------------------
def loader_lessEmbed(X, y_activity, batch_size, shuffle=False, drop_last=False):
    # Replace these indices with the correct indices for according to exisiting data
    season_idx = 0
    weekend_idx = 1
    num_categorical_features = 2 # Update this to the total number of categorical features

    dataset = TensorDataset(
        torch.tensor(X[:, :, season_idx], dtype=torch.long),
        torch.tensor(X[:, :, weekend_idx], dtype=torch.long),
        torch.tensor(X[:, :, num_categorical_features:], dtype=torch.float),  # Continuous data
        torch.tensor(y_activity, dtype=torch.long),  # Activity labels as long integers
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_model_lessEmbed(model, data_loader, criterion_activity, optimizer, device, output_dim_activity):
    model.train()

    train_loss = 0.0
    train_activity_loss = 0.0

    correct_activity_predictions = 0

    total_activity_predictions = 0

    for batch_idx, data in enumerate(data_loader):
        try:
            # Unpack all the data from the dataloader
            # !!! Adjust the number of unpacked variables based on how many there is
            season_data, weekend_data, continuous_data, activity_target= data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),

            optimizer.zero_grad()
            # Model's forward pass
            activity_output = model(season_data, weekend_data, continuous_data,)

            # Compute the loss for both outputs
            loss_activity = criterion_activity(activity_output.view(-1, output_dim_activity), activity_target.view(-1))

            total_loss = loss_activity
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_activity_loss += loss_activity.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

        except Exception as e:
            # Handle errors during the training loop
            print(f"An error occurred during training at batch {batch_idx}: {e}")
            continue  # Skip this batch and continue with the next

    # Avoid division by zero
    if total_activity_predictions == 0:
        raise ValueError("No predictions made for activity or location - check your data and model outputs.")

    # Check gradients
    #check_gradients(model)

    # Average loss and accuracy
    train_loss_avg = train_loss / len(data_loader.dataset)
    train_activity_loss = train_activity_loss / len(data_loader.dataset)

    train_activity_accuracy = correct_activity_predictions / total_activity_predictions

    return train_loss_avg, train_activity_loss, train_activity_accuracy

def validate_model_lessEmbed(model, data_loader, criterion_activity,device, output_dim_activity):
    model.eval()
    valid_loss = 0.0
    valid_activity_loss = 0.0

    correct_activity_predictions = 0

    total_activity_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            season_data, weekend_data, continuous_data, activity_target= data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),
            activity_output= model(season_data, weekend_data, continuous_data,)
            loss_activity = criterion_activity(activity_output.view(-1, output_dim_activity), activity_target.view(-1))
            valid_loss += loss_activity.item()
            valid_activity_loss += loss_activity.item()

            # Calculate activity accuracy
            _, predicted_activity_classes = torch.max(activity_output, dim=2)
            correct_activity_predictions += (predicted_activity_classes == activity_target).float().sum()
            total_activity_predictions += activity_target.numel()

    valid_loss_avg = valid_loss / len(data_loader.dataset)
    valid_activity_loss = valid_activity_loss / len(data_loader.dataset)

    valid_activity_accuracy = correct_activity_predictions / total_activity_predictions

    return valid_loss_avg, valid_activity_loss, valid_activity_accuracy,

def train_and_evaluate_lessEmbed(model,
                                 X_train, y_activity_train,
                                 X_valid, y_activity_valid,
                                 device,
                                 epochs=100, batch_size=48, learning_rate=0.01,
                                 checkpoint_path='best_model_LessEmbed_train.pth'):

    output_dim_activity = len(set(y_activity_train.flatten()))

    # Create data loaders
    train_loader = loader_lessEmbed(X_train, y_activity_train, batch_size, shuffle=True, drop_last=True)
    valid_loader = loader_lessEmbed(X_valid, y_activity_valid, batch_size, shuffle=False, drop_last=True)

    # Define loss functions for each task
    criterion_activity = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #default LR: 0.001

    history = {'train_loss': [], 'valid_loss': [],
               'train_activity_loss': [],
               'valid_activity_loss': [],
               'train_activity_accuracy': [],
               'valid_activity_accuracy': [], }
    best_valid_accuracy = 0.0

    for epoch in range(epochs):
        train_loss_avg, train_activity_loss, train_activity_accuracy, = train_model_lessEmbed(model, train_loader, criterion_activity,
                                                                                              optimizer, device, output_dim_activity)
        valid_loss_avg, valid_activity_loss, valid_activity_accuracy = validate_model_lessEmbed(model, valid_loader, criterion_activity,
                                                                                                device, output_dim_activity)

        history['train_loss'].append(train_loss_avg)
        history['train_activity_loss'].append(train_activity_loss)

        history['valid_loss'].append(valid_loss_avg)
        history['valid_activity_loss'].append(valid_activity_loss)

        history['train_activity_accuracy'].append(train_activity_accuracy.item())
        history['valid_activity_accuracy'].append(valid_activity_accuracy.item())

        print(f'Epoch {epoch}: '
              f'Val_Act_Acc: {valid_activity_accuracy:.4f}, '
              f'Train_Act_Acc: {train_activity_accuracy:.4f}, '
              f'Train Loss: {train_loss_avg:.4f},  Valid Loss: {valid_loss_avg:.4f},' 
              f'Train Act Loss: {train_activity_loss:.4f}, ' 
              f'Val Act Loss: {valid_activity_loss:.4f}, ')

        # Check if the model's validation accuracy for activity is improved
        if valid_activity_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_activity_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        """ Close during tuning"""
        # Check if the accuracy threshold is met and stop training if it is: PREVENT MEMORIZATION
        accuracy_threshold = 0.95
        if best_valid_accuracy >= accuracy_threshold:
            print(f'Early stopping triggered at epoch {epoch} due to reaching accuracy threshold of {accuracy_threshold:.2f}')
            break

    return history, model

