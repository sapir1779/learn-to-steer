import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import re
import shutil
import torch

from train_relation_classifier.loss_utils import get_classification_function


class EpochStats:
    def __init__(self, src, epoch, num_epochs, num_batches):
        self.src = src
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.classification_fn = get_classification_function()
        
        self.epoch_loss = 0
        self.correct_predictions = 0
        self.total_samples = 0
        self.timestep_correct = defaultdict(int)
        self.timestep_total = defaultdict(int)
        self.all_preds = []
        self.all_labels = []
    
    def update_batch(self, outputs, labels, timestep, loss):
        # Loss:
        self.epoch_loss += loss.item()
        
        # Count correct predictions:
        predicted_classes = self.classification_fn(outputs)
        labels = labels.cpu().numpy()
        self.all_preds.extend(predicted_classes)
        self.all_labels.extend(labels)
        self.correct_predictions += (predicted_classes == labels).sum()
        self.total_samples += labels.shape[0]
        
        # Per timestep:
        timestep_values = timestep.squeeze(dim=1).to(torch.int64).cpu().numpy().astype(int).tolist()
        for i, ts in enumerate(timestep_values):
            self.timestep_correct[ts] += (predicted_classes[i] == labels[i])
            self.timestep_total[ts] += 1
            
    def print_classification_report(self, relation_names, output_dir):
        # Classification report
        report_str = classification_report(self.all_labels, self.all_preds, 
                                       target_names=relation_names, zero_division=0.0)
        report_dict = classification_report(self.all_labels, self.all_preds, 
                                       target_names=relation_names, zero_division=0.0, output_dict=True)
        print(report_str)

        # Confusion Matrix
        cm = confusion_matrix(self.all_labels, self.all_preds)
        print(f"Confusion Matrix:\n{cm}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=relation_names, 
                    yticklabels=relation_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix: {self.src}')
        accuracy_fig_path = os.path.join(output_dir, f"{self.src}_{self.epoch}__confusion_matrix_per_class.png")
        plt.savefig(accuracy_fig_path)
        plt.close()
        
        return report_dict
    
    def process_epoch(self, relation_names, output_dir, model=None):
        # Compute loss and accuracy
        self.epoch_loss /= self.num_batches
        accuracy = (self.correct_predictions / self.total_samples) * 100
        print(
            f"Epoch [{self.epoch+1}/{self.num_epochs}], "
            f"{self.src} Loss: {self.epoch_loss:.2f}, "
            f"{self.src} Accuracy: {accuracy:.2f}%"
        )

        # Compute accuracy per timestep
        timestep_accuracies = {ts: (self.timestep_correct[ts] / self.timestep_total[ts]) * 100 
                                    for ts in self.timestep_correct}
        for ts in sorted(timestep_accuracies):
            acc = timestep_accuracies[ts]
            print(f"Timestep {ts}: {self.src} Accuracy = {acc:.2f}%")
        
        # Classification report and confusion matrix
        per_relation_report_dict = self.print_classification_report(relation_names, output_dir)
        
        print("----------------------------------------------------------------------------------")
        
        # Save metrics such as loss, accuracy, accuracy per timestep, and precision+recall per relation name
        metrics_data = {
            "loss": self.epoch_loss,
            "accuracy": accuracy,
            "timestep_accuracies": timestep_accuracies,
            "per_relation_report_dict": per_relation_report_dict,
        }
        with open(os.path.join(output_dir, f"{self.src}_{self.epoch}__metrics.json"), "w") as f:
            json.dump(metrics_data, f, indent=4)
            
        return self.epoch_loss, accuracy


def plot_and_save_metrics(train_losses, val_losses, test_losses, train_accuracies, 
                          val_accuracies, test_accuracies, output_dir):
    """
    Plots and saves the training/validation loss and accuracy over epochs.
    """

    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    if len(test_losses) > 0:
        plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    if len(test_accuracies) > 0:
        plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

    # Save numerical results in JSON format
    metrics_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracies": test_accuracies,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)
  

def copy_latest_early_stopping_start_model(output_dir):
    """ Copies the latest epoch in which the model first started the early stopping process. """
    pattern = re.compile(r"model__epoch_(\d+)__early_stopping_start\.pth")
    best_epoch = -1
    best_file = None

    # Walk through files in the directory
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_file = filename

    if best_file:
        src_path = os.path.join(output_dir, best_file)
        dst_path = os.path.join(output_dir, "ES_final_model.pth")
        shutil.copy(src_path, dst_path)
        print(f"Copied {best_file} to ES_final_model.pth")
    else:
        print("No matching early stopping model found.")


class EarlyStopping:
    def __init__(self, patience=1.0, min_delta=0.0):
        """
        Initializes the EarlyStopping instance.

        Parameters:
            patience (int): The number of epochs with no improvement after which training will be stopped.
            min_delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def should_stop(self, validation_loss):
        """
        Determines whether training should be stopped early.

        Parameters:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        
        started_no_improvement = False
        if validation_loss < (self.min_validation_loss - self.min_delta):
            # Reset counter if there is a significant improvement
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            # Increment counter if no improvement
            print(f"early stopping counter: {self.counter}")
            started_no_improvement = self.counter == 0
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True
            
        return self.early_stop, started_no_improvement
