import torch

def predict_winner(model, input_tensor):
    """
    Predict the winner of a game based on normalized input stats.

    :param model: Trained PyTorch model.
    :param input_tensor: Normalized input feature tensor.
    :return: Probability of the home team winning.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        prediction = model(input_tensor)
        return prediction.item()





# import torch


# def predict_winner(model, home_team_stats, away_team_stats, scaler, device='cpu'):
#     """
#     Predict the winner of a game based on rolling averages.

#     :param model: The trained PyTorch model.
#     :param home_team_stats: Rolling averages for the home team.
#     :param away_team_stats: Rolling averages for the away team.
#     :param scaler: Scaler used during training to normalize features.
#     :param device: Device to run the model on ('cpu' or 'cuda').
#     :return: Probability of the home team winning.
#     """
#     # Combine rolling averages for home and away teams into a single feature vector
#     combined_stats = [
#         home_team_stats['PTS_rolling_avg'],
#         home_team_stats['REB_rolling_avg'],
#         home_team_stats['AST_rolling_avg'],
#         home_team_stats['FGM_rolling_avg'],
#         home_team_stats['FGA_rolling_avg'],
#         home_team_stats['FG_PCT_rolling_avg'],
#         home_team_stats['FG3M_rolling_avg'],
#         home_team_stats['FG3A_rolling_avg'],
#         home_team_stats['FG3_PCT_rolling_avg'],
#         home_team_stats['FTM_rolling_avg'],
#         home_team_stats['FTA_rolling_avg'],
#         home_team_stats['FT_PCT_rolling_avg'],
#         home_team_stats['OREB_rolling_avg'],
#         home_team_stats['DREB_rolling_avg'],
#         home_team_stats['STL_rolling_avg'],
#         home_team_stats['BLK_rolling_avg'],
#         home_team_stats['TOV_rolling_avg'],
#         home_team_stats['PF_rolling_avg'],
#         away_team_stats['PTS_rolling_avg'],
#         away_team_stats['REB_rolling_avg'],
#         away_team_stats['AST_rolling_avg'],
#         away_team_stats['FGM_rolling_avg'],
#         away_team_stats['FGA_rolling_avg'],
#         away_team_stats['FG_PCT_rolling_avg'],
#         away_team_stats['FG3M_rolling_avg'],
#         away_team_stats['FG3A_rolling_avg'],
#         away_team_stats['FG3_PCT_rolling_avg'],
#         away_team_stats['FTM_rolling_avg'],
#         away_team_stats['FTA_rolling_avg'],
#         away_team_stats['FT_PCT_rolling_avg'],
#         away_team_stats['OREB_rolling_avg'],
#         away_team_stats['DREB_rolling_avg'],
#         away_team_stats['STL_rolling_avg'],
#         away_team_stats['BLK_rolling_avg'],
#         away_team_stats['TOV_rolling_avg'],
#         away_team_stats['PF_rolling_avg'],
#     ]

#     # Normalize the features
#     combined_stats = scaler.transform([combined_stats])

#     # Convert to PyTorch tensor
#     combined_stats_tensor = torch.tensor(combined_stats, dtype=torch.float32).to(device)

#     # Make prediction
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():
#         prediction = model(combined_stats_tensor)
#         probability = prediction.item()

#     return probability

# 'FGM_rolling_avg',
# 'FGA_rolling_avg',
# 'FG_PCT_rolling_avg',
# 'FG3M_rolling_avg',
# 'FG3A_rolling_avg',
# 'FG3_PCT_rolling_avg',
# 'FTM_rolling_avg',
# 'FTA_rolling_avg',
# 'FT_PCT_rolling_avg',
# 'OREB_rolling_avg',
# 'DREB_rolling_avg',
# 'STL_rolling_avg',
# 'BLK_rolling_avg',
# 'TOV_rolling_avg',
# 'PF_rolling_avg'