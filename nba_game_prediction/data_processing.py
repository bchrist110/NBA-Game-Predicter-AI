import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def fetch_nba_data(season: str = '2024-25') -> pd.DataFrame:
    """
    Fetch NBA game data for the specified season using nba_api.

    :param season: The NBA season (e.g., '2023-24').
    :return: A Pandas DataFrame containing game data.
    """
    print(f"Fetching data for season {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    print(f"Fetched {len(games)} games.")
    return games


def calculate_rolling_average(df: pd.DataFrame, team_col: str, stat_cols: list, window: int = 5) -> pd.DataFrame:
    """
    Calculate rolling averages of the specified statistics for each team.

    :param df: Input DataFrame containing game data.
    :param team_col: Column indicating the team name.
    :param stat_cols: List of statistic columns to compute rolling averages for.
    :param window: The rolling average window (default is 5 games).
    :return: DataFrame with rolling average columns added.
    """
    print("Calculating rolling averages...")
    df = df.sort_values(by=["GAME_DATE"]).reset_index(drop=True)

    for stat in stat_cols:
        rolling_col_name = f'{stat}_rolling_avg'
        df[rolling_col_name] = df.groupby(team_col)[stat].transform(lambda x: x.rolling(window, min_periods=1).mean())
        print(f"Computed rolling average for {stat}.")
    
    return df


def process_nba_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw NBA game data into a cleaned and feature-engineered format.

    :param raw_data: The raw game data DataFrame from nba_api.
    :return: Processed DataFrame ready for training or prediction.
    """
    print("Processing raw NBA game data...")

    # Convert GAME_DATE to datetime
    raw_data['GAME_DATE'] = pd.to_datetime(raw_data['GAME_DATE'])

    # Map 'W' and 'L' to binary outcomes
    raw_data['WIN'] = raw_data['WL'].map({'W': 1, 'L': 0})

    # Drop WL column as it's no longer needed
    raw_data = raw_data.drop(columns=['WL'])

    # Drop rows with missing values in critical columns
    raw_data = raw_data.dropna(subset=['WIN'])

    # Normalize TEAM_NAME to lowercase for consistency
    raw_data['TEAM_NAME'] = raw_data['TEAM_NAME'].str.lower()
    print(raw_data['TEAM_NAME'])
    print("Processed data with normalized TEAM_NAME column.")
    return raw_data


def create_features_labels(data: pd.DataFrame) -> tuple:
    """
    Create feature and label sets for training from processed data.

    :param data: The processed game data DataFrame.
    :return: Tuple of features (X) and labels (y).
    """
    print("Creating features and labels...")

    # Duplicate the dataset to simulate home and away teams
    home_data = data.copy()
    away_data = data.copy()

    # Rename columns for home and away teams
    feature_cols_home = ['PTS_rolling_avg', 'REB_rolling_avg', 'AST_rolling_avg', 'WIN']
    feature_cols_away = ['PTS_rolling_avg', 'REB_rolling_avg', 'AST_rolling_avg']

    home_data = home_data.rename(columns={col: f"{col}_home" for col in feature_cols_home})
    away_data = away_data.rename(columns={col: f"{col}_away" for col in feature_cols_away})

    # Merge home and away data
    combined_data = home_data.merge(
        away_data,
        left_index=True,
        right_index=True
    )

    # Features: Rolling averages for home and away teams
    feature_cols = [
        'PTS_rolling_avg_home', 'REB_rolling_avg_home', 'AST_rolling_avg_home',
        'PTS_rolling_avg_away', 'REB_rolling_avg_away', 'AST_rolling_avg_away'
    ]
    X = combined_data[feature_cols]

    # Labels: Binary outcome for home team winning
    y = combined_data['WIN_home']

    print(f"Created features (X) with shape {X.shape} and labels (y) with shape {y.shape}.")
    return X, y


def save_data_to_csv(data: pd.DataFrame, filepath: str):
    """
    Save the processed data to a CSV file.

    :param data: DataFrame to save.
    :param filepath: Path to save the CSV file.
    """
    print(f"Saving data to {filepath}...")
    data.to_csv(filepath, index=False)
    print("Data saved successfully.")


def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load processed data from a CSV file.

    :param filepath: Path to the CSV file.
    :return: Loaded DataFrame.
    """
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    data['TEAM_NAME'] = data['TEAM_NAME'].str.lower()  # Ensure TEAM_NAME is normalized
    print("Data loaded successfully.")
    return data


# Example usage
if __name__ == "__main__":
    # Step 1: Fetch data
    raw_games = fetch_nba_data(season='2024-25')
    
    # Step 2: Process data
    processed_data = process_nba_data(raw_games)
    
    # Step 3: Calculate rolling averages
    processed_data = calculate_rolling_average(
        processed_data,
        team_col='TEAM_NAME',
        stat_cols=['PTS', 'REB', 'AST']
    )
    
    # Step 4: Save to CSV
    save_data_to_csv(processed_data, './data/processed_nba_games_cleaned.csv')
