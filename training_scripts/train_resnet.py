import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.models as models

FEATURES_DIR= "./data_preprocessed/log_mel_spectograms"
MODEL_CHECKPOINT_PATH = "bestwts/best_resnet_drone_model.pth"
BATCH_SIZE = 32
EPOCHS = 30           # Bunted up to 30 because Early Stopping will catch it safely
LEARNING_RATE = 0.001
PATIENCE = 5          # Stop training if validation loss doesn't improve for 4 straight epochs
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_and_partition_data():
    X = np.load(os.path.join(FEATURES_DIR, "X_resnet.npy"))
    y = np.load(os.path.join(FEATURES_DIR, "y_resnet.npy"))
    
    # Reorder to PyTorch style: [Samples, Channels, Height, Width]
    X = np.transpose(X, (0, 3, 1, 2))
    
    # Split 1: 25% for rigorous testing
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Split 2: 10% validation slice from the training pool
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool, test_size=0.10, random_state=42, stratify=y_train_pool
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_resnet_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 4)
    )
    return model.to(DEVICE)

def train_model():
    train_loader, val_loader, test_loader = load_and_partition_data()
    model = get_resnet_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Tracking registers for Early Stopping & Checkpointing
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training initiated on execution device: {DEVICE}")
    print(f"Early Stopping Monitor Active (Patience: {PATIENCE} epochs)\n")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # Validation Pass
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # --- EARLY STOPPING & SAVE CHECK LOGIC ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            patience_counter = 0  # Reset counter since performance improved
            print(f"   --> Validation loss improved. Optimal weights updated.")
        else:
            patience_counter += 1
            print(f"   --> No improvement. Patience counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation loss stagnated for {PATIENCE} epochs. Terminating loop.")
                break
                
    # --- ROLLBACK TO BEST WEIGHTS BEFORE THE FINAL EXAM ---
    print(f"\nRolling back to the best recorded model weights from '{MODEL_CHECKPOINT_PATH}' for rigorous testing...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    
    evaluate_testing_holdout(model, test_loader)

def evaluate_testing_holdout(model, test_loader):
    print("\n========== TESTING PHASE EXECUTION ==========")
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.append(predicted.cpu().view(-1))
            all_true_labels.append(batch_y.view(-1))
            
    final_predictions = torch.cat(all_predictions).numpy()
    final_true_labels = torch.cat(all_true_labels).numpy()
    
    target_names = ["Light", "Medium", "Heavy", "No-Drone"]
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(final_true_labels, final_predictions, target_names=target_names))
    
    print("[CONFUSION MATRIX]")
    print(confusion_matrix(final_true_labels, final_predictions))

if __name__ == "__main__":
    train_model()