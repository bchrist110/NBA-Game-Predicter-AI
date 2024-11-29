import torch
import torch.nn as nn


class NBAWinnerPredictor(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the neural network model.
        :param input_size: Number of input features.
        """
        super(NBAWinnerPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def save_model(model, filepath):
    """
    Save the trained model to a file.
    :param model: Trained PyTorch model.
    :param filepath: Filepath to save the model.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, input_size):
    """
    Load a trained model from a file.
    :param filepath: Filepath of the saved model.
    :param input_size: Number of input features.
    :return: Loaded PyTorch model.
    """
    model = NBAWinnerPredictor(input_size)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model
