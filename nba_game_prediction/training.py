import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

from model import NBAWinnerPredictor, save_model
from data_processing import load_data_from_csv, create_features_labels


def train_model(model, train_loader, criterion, optimizer, epochs=50, device='cpu'):
    """
    Train the neural network model.
    
    :param model: The PyTorch model to train.
    :param train_loader: DataLoader for the training dataset.
    :param criterion: Loss function.
    :param optimizer: Optimizer for gradient descent.
    :param epochs: Number of epochs to train.
    :param device: Device to train on ('cpu' or 'cuda').
    :return: Trained model.
    """
    print(f"Starting training for {epochs} epochs on {device}...")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)

            # Compute loss
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Backward pass and optimization
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training completed.")
    return model


def main():
    # Load processed data
    print("Loading processed data...")
    data = load_data_from_csv('./data/processed_nba_games_cleaned.csv')

    # Create features and labels
    print("Creating features and labels...")
    X, y = create_features_labels(data)

    # Check label distribution
    print("Label distribution:", y.value_counts())

    # Split into training and validation datasets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    print("Normalizing the features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save the scaler for future predictions
    joblib.dump(scaler, './models/scaler.pkl')
    print("Scaler saved to './models/scaler.pkl'")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    input_size = 6  # 3 stats for home + 3 stats for away
    model = NBAWinnerPredictor(input_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    trained_model = train_model(model, train_loader, criterion, optimizer, epochs=50)

    # Save the trained model
    save_model(trained_model, './models/nba_winner_predictor.pth')
    print("Model saved to './models/nba_winner_predictor.pth'")


if __name__ == "__main__":
    main()
