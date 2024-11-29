import torch


def predict_winner(model, home_team_stats, away_team_stats, scaler, device='cpu'):
    """
    Predict the winner of a game based on rolling averages.

    :param model: The trained PyTorch model.
    :param home_team_stats: Rolling averages for the home team.
    :param away_team_stats: Rolling averages for the away team.
    :param scaler: Scaler used during training to normalize features.
    :param device: Device to run the model on ('cpu' or 'cuda').
    :return: Probability of the home team winning.
    """
    # Combine rolling averages for home and away teams into a single feature vector
    combined_stats = [
        home_team_stats['PTS_rolling_avg'],
        home_team_stats['REB_rolling_avg'],
        home_team_stats['AST_rolling_avg'],
        away_team_stats['PTS_rolling_avg'],
        away_team_stats['REB_rolling_avg'],
        away_team_stats['AST_rolling_avg']
    ]

    # Normalize the features
    combined_stats = scaler.transform([combined_stats])

    # Convert to PyTorch tensor
    combined_stats_tensor = torch.tensor(combined_stats, dtype=torch.float32).to(device)

    # Make prediction
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        prediction = model(combined_stats_tensor)
        probability = prediction.item()

    return probability
