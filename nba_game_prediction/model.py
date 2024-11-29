import torch
import torch.nn as nn

class NBAWinnerPredictor(nn.Module):
    def __init__(self, input_size):
        super(NBAWinnerPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid activation here
        return x
    


def save_model(model: nn.Module, filepath: str):
    """
    Save the model's state dictionary to a file.
    
    :param model: Trained PyTorch model.
    :param filepath: Path to save the model file.
    """
    print(f"Saving model to {filepath}...")
    torch.save(model.state_dict(), filepath)
    print("Model saved successfully.")


def load_model(filepath: str, input_size: int) -> nn.Module:
    """
    Load a model's state dictionary from a file.
    
    :param filepath: Path to the saved model file.
    :param input_size: Number of input features (to initialize the model architecture).
    :return: Loaded PyTorch model.
    """
    print(f"Loading model from {filepath}...")
    model = NBAWinnerPredictor(input_size)
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor):
    """
    Evaluate the model's performance on a test dataset.
    
    :param model: Trained PyTorch model.
    :param X_test: Test features as a PyTorch tensor.
    :param y_test: Test labels as a PyTorch tensor.
    :return: Accuracy of the model on the test set.
    """
    print("Evaluating model...")
    with torch.no_grad():
        predictions = model(X_test)
        predicted_classes = (predictions > 0.5).float()  # Convert probabilities to binary classes
        accuracy = (predicted_classes == y_test).sum().item() / len(y_test)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy
