import argparse
import joblib
import torch
from model import load_model
from prediction import predict_winner
from data_processing import load_data_from_csv, calculate_rolling_average
import pandas as pd


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="NBA Game Prediction")
    subparsers = parser.add_subparsers(dest='command')

    # Subcommand for training
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train the model")

    # Subcommand for prediction
    predict_parser = subparsers.add_parser('predict', help="Predict the winner of a game")
    predict_parser.add_argument('--home_team', type=str, required=True, help="Name of the home team")
    predict_parser.add_argument('--away_team', type=str, required=True, help="Name of the away team")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'train':
        from training import main as train_main
        train_main()

    elif args.command == 'predict':
        # Load the trained model
        print("Loading model...")
        input_size = 36  # Dynamically match to the number of input features
        model = load_model('./models/nba_winner_predictor.pth', input_size)
        print("Model loaded successfully.")

        # Load the scaler
        print("Loading scaler...")
        scaler = joblib.load('./models/scaler.pkl')
        feature_names = joblib.load('./models/feature_names.pkl')
        print("Scaler loaded successfully.")

        # Load the processed game data
        print("Loading processed game data...")
        data = load_data_from_csv('./data/processed_nba_games_cleaned.csv')

        # Normalize TEAM_NAME for matching
        data['TEAM_NAME'] = data['TEAM_NAME'].str.lower()

        # Get team inputs from user
        home_team_name = args.home_team.lower()
        away_team_name = args.away_team.lower()

        # Calculate rolling averages
        print("Calculating rolling averages...")
        stat_cols = [
            'PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF'
        ]
        data = calculate_rolling_average(data, team_col='TEAM_NAME', stat_cols=stat_cols)

        # Filter data for the specified teams
        print(f"Filtering data for teams: {args.home_team} and {args.away_team}...")
        home_team_stats = data[data['TEAM_NAME'] == home_team_name].iloc[-1]
        away_team_stats = data[data['TEAM_NAME'] == away_team_name].iloc[-1]

        # Combine stats for prediction
        combined_stats = pd.DataFrame([{
            **{f"{stat}_rolling_avg_home": home_team_stats[f"{stat}_rolling_avg"] for stat in stat_cols},
            **{f"{stat}_rolling_avg_away": away_team_stats[f"{stat}_rolling_avg"] for stat in stat_cols}
        }])

        combined_stats = combined_stats[feature_names]

        # Normalize the combined stats
        combined_stats_normalized = scaler.transform(combined_stats)

        # Convert to tensor for prediction
        combined_stats_tensor = torch.tensor(combined_stats_normalized, dtype=torch.float32)

        # Predict the outcome
        print(f"Predicting outcome for {args.home_team} vs {args.away_team}...")
        probability = predict_winner(model, combined_stats_tensor)
        print(f"The predicted probability of {args.home_team} winning is {probability:.2f}")


if __name__ == "__main__":
    main()
