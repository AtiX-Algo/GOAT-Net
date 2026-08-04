# GOAT-Net Dataset Manager
# Centralized loader and normalizer for all raw datasets.

import pandas as pd
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/GOAT-Net")
DATA_RAW = ROOT / "data" / "raw"
META = ROOT / "metadata"

def load_player_mapping() -> pd.DataFrame:
    """Loads the canonical player mapping table."""
    mapping_path = META / "player_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")
    return pd.read_csv(mapping_path)

def normalize_player_names(df: pd.DataFrame, source: str, name_col: str) -> pd.DataFrame:
    """
    Attaches canonical_id and common_name to a dataframe using source-specific name matching.
    
    Parameters:
        df: The source DataFrame to normalize.
        source: Key matching column in player_mapping.csv ('statsbomb', 'fbref', 'understat', 'transfermarkt', 'fifa').
        name_col: The column in df containing the player's name.
    """
    mapping = load_player_mapping()
    source_key = f"{source.lower()}_name"
    
    if source_key not in mapping.columns:
        raise ValueError(f"Unknown source '{source}'. Must be one of: statsbomb, fbref, understat, transfermarkt, fifa")
    
    # Create lookup dictionaries
    id_map = dict(zip(mapping[source_key], mapping["canonical_id"]))
    name_map = dict(zip(mapping[source_key], mapping["common_name"]))
    
    df = df.copy()
    df["canonical_id"] = df[name_col].map(id_map)
    df["common_name"] = df[name_col].map(name_map)
    
    return df

def load_statsbomb_events() -> pd.DataFrame:
    """Loads and normalizes StatsBomb spatial event data."""
    path = DATA_RAW / "modern" / "statsbomb" / "statsbomb_panel_events.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return normalize_player_names(df, source="statsbomb", name_col="player")

def load_fbref_stats() -> pd.DataFrame:
    """Loads and normalizes all FBref season-level box scores into a single DataFrame."""
    fbref_dir = DATA_RAW / "modern" / "fbref"
    files = list(fbref_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    return normalize_player_names(combined, source="fbref", name_col="Player")

def load_understat_xg() -> pd.DataFrame:
    """Loads and normalizes Understat xG metrics."""
    path = DATA_RAW / "modern" / "understat" / "tier1_understat_xg.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    name_col = "player_name" if "player_name" in df.columns else "Player"
    return normalize_player_names(df, source="understat", name_col=name_col)

def load_transfermarkt() -> pd.DataFrame:
    """Loads and normalizes Transfermarkt player valuations."""
    players_path = DATA_RAW / "modern" / "transfermarkt" / "tier1_tm_players.parquet"
    vals_path = DATA_RAW / "modern" / "transfermarkt" / "tier1_tm_valuations.parquet"
    
    if not players_path.exists() or not vals_path.exists():
        return pd.DataFrame()
        
    players = pd.read_parquet(players_path)
    vals = pd.read_parquet(vals_path)
    
    merged = pd.merge(vals, players, on="player_id", how="left")
    return normalize_player_names(merged, source="transfermarkt", name_col="name")
