import argparse
import joblib
import torch
from model import load_model
from prediction import predict_winner
from data_processing import load_data_from_csv, calculate_rolling_average


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
    predict_parser.add_argument('--home_team', type=str, required=True, help="Home team name")
    predict_parser.add_argument('--away_team', type=str, required=True, help="Away team name")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'train':
        from training import main as train_main
        train_main()

    elif args.command == 'predict':
        # Load the trained model
        input_size = 6  # 3 stats for home + 3 stats for away
        model = load_model('./models/nba_winner_predictor.pth', input_size=input_size)
        print("Model loaded successfully.")

        # Load the scaler
        scaler = joblib.load('./models/scaler.pkl')
        print("Scaler loaded successfully.")

        # Load the processed game data
        print("Loading processed game data...")
        data = load_data_from_csv('./data/processed_nba_games_cleaned.csv')

        # Calculate rolling averages for both teams
        data = calculate_rolling_average(data, team_col='TEAM_NAME', stat_cols=['PTS', 'REB', 'AST'])

        # Normalize TEAM_NAME for matching
        data['TEAM_NAME'] = data['TEAM_NAME'].str.lower()
        home_team_name = args.home_team.lower()
        away_team_name = args.away_team.lower()

        # Filter for home and away teams
        home_team_stats = data[data['TEAM_NAME'] == home_team_name]
        away_team_stats = data[data['TEAM_NAME'] == away_team_name]

        # Check for missing teams
        if home_team_stats.empty:
            raise ValueError(f"Home team '{args.home_team}' not found in the dataset. "
                             f"Available team names: {data['TEAM_NAME'].unique()}")

        if away_team_stats.empty:
            raise ValueError(f"Away team '{args.away_team}' not found in the dataset. "
                             f"Available team names: {data['TEAM_NAME'].unique()}")

        # Get the most recent game stats
        home_team_stats = home_team_stats.iloc[-1]
        away_team_stats = away_team_stats.iloc[-1]

        # Predict the winner
        print(f"Predicting outcome for {args.home_team} vs {args.away_team}...")
        probability = predict_winner(model, home_team_stats, away_team_stats, scaler)
        print(f"The predicted probability of {args.home_team} winning is {probability:.2f}")


if __name__ == "__main__":
    main()
