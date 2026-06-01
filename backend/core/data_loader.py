"""
Data loading utilities for NBA shot data.
Ported from the original Dash app.
"""

import tarfile
from io import BytesIO
from itertools import product
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd


# Directory where pre-built .npy cache files are stored
_CACHE_DIR = Path(__file__).parent.parent / "data"


def _npy_cache_base(season: int) -> Path:
    """Return the base path (without suffix) for a season's npy cache files."""
    return _CACHE_DIR / f"nba_shotdetail_{season}"


def _load_df_from_npy(season: int) -> "pd.DataFrame | None":
    """
    Load a cached NBA shot DataFrame from .npy files.
    Returns None if the cache does not exist.
    """
    base = _npy_cache_base(season)
    data_path = Path(str(base) + "_data.npy")
    cols_path = Path(str(base) + "_cols.txt")
    dtypes_path = Path(str(base) + "_dtypes.txt")

    if not data_path.exists():
        return None

    values = np.load(data_path, allow_pickle=True)
    columns = cols_path.read_text(encoding="utf-8").strip().splitlines()
    dtype_strs = dtypes_path.read_text(encoding="utf-8").strip().splitlines()

    df = pd.DataFrame(values, columns=columns)

    # Restore numeric dtypes
    for col, dtype_str in zip(columns, dtype_strs):
        try:
            if "int" in dtype_str:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif "float" in dtype_str:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        except Exception:
            pass

    return df


def load_nba_data(
    path: "Path | str" = Path.cwd(),
    seasons=(2022,),
    data=("shotdetail",),
    seasontype: str = "rg",
    league: str = "nba",
    in_memory: bool = True,
    use_pandas: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load NBA data from .npy cache files if available, otherwise download from GitHub.

    Args:
        path: Working directory path
        seasons: Tuple of season years
        data: Tuple of data types to load
        seasontype: "rg" (regular), "po" (playoffs), or "all"
        league: "nba" or "wnba"
        in_memory: Whether to load data in memory
        use_pandas: Whether to use pandas DataFrame
        use_cache: If True (default), try to load from .npy cache before downloading

    Returns:
        DataFrame with NBA shot data
    """
    if isinstance(path, str):
        path = Path(path).expanduser()
    if isinstance(seasons, int):
        seasons = (seasons,)
    if isinstance(data, str):
        data = (data,)

    # --- .npy cache fast path ---
    # Only supported for single shotdetail data type, regular season, NBA
    if (
        use_cache
        and league.lower() == "nba"
        and seasontype == "rg"
        and set(data) == {"shotdetail"}
    ):
        cached_frames = []
        missing_seasons = []
        for season in seasons:
            df_cached = _load_df_from_npy(season)
            if df_cached is not None:
                print(f"[cache] Loaded season {season} from .npy cache ({len(df_cached)} rows).")
                cached_frames.append(df_cached)
            else:
                missing_seasons.append(season)

        if not missing_seasons:
            # All seasons were cached
            return pd.concat(cached_frames, axis=0, ignore_index=True)

        # Some seasons missing: download them
        print(f"[cache] Seasons not cached, downloading: {missing_seasons}")
        downloaded = _download_nba_data(missing_seasons, data, seasontype, league, in_memory, use_pandas)
        all_frames = cached_frames + [downloaded]
        return pd.concat(all_frames, axis=0, ignore_index=True)

    # --- No cache: download directly ---
    return _download_nba_data(list(seasons), data, seasontype, league, in_memory, use_pandas)


def _download_nba_data(
    seasons,
    data,
    seasontype: str = "rg",
    league: str = "nba",
    in_memory: bool = True,
    use_pandas: bool = True,
) -> pd.DataFrame:
    """
    Internal: download NBA data from GitHub (original implementation).
    """
    if isinstance(seasons, int):
        seasons = (seasons,)
    if isinstance(data, str):
        data = (data,)

    if (len(data) > 1) and in_memory:
        raise ValueError("When in_memory=True, please specify only one dataset type in 'data'.")

    if seasontype == "rg":
        need_data = tuple(
            ["_".join([d, str(season)]) for (d, season) in product(data, seasons)]
        )
    elif seasontype == "po":
        need_data = tuple(
            ["_".join([d, seasontype, str(season)])
             for (d, seasontype, season) in product(data, (seasontype,), seasons)]
        )
    else:
        need_data_rg = tuple(
            ["_".join([d, str(season)]) for (d, season) in product(data, seasons)]
        )
        need_data_po = tuple(
            ["_".join([d, seasontype, str(season)])
             for (d, seasontype, season) in product(data, ("po",), seasons)]
        )
        need_data = need_data_rg + need_data_po

    if league.lower() == "wnba":
        need_data = ["wnba_" + x for x in need_data]

    # Fetch list of available datasets
    with urlopen("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt") as f:
        v = f.read().decode("utf-8").strip()

    name_v = [string.split("=")[0] for string in v.split("\n")]
    element_v = [string.split("=")[1] for string in v.split("\n")]

    need_name = [name for name in name_v if name in need_data]
    need_element = [
        element for (name, element) in zip(name_v, element_v) if name in need_data
    ]

    if not need_name:
        raise RuntimeError(
            f"Required data not found in list_data.txt. "
            f"Try changing 'seasons' or 'seasontype'."
        )

    if in_memory:
        table = pd.DataFrame() if use_pandas else []

        for name, url in zip(need_name, need_element):
            with urlopen(url) as response:
                file_content = response.read()
                with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
                    csv_file_name = "".join([name, ".csv"])
                    csv_file = tar.extractfile(csv_file_name)
                    if csv_file is None:
                        continue
                    if use_pandas:
                        df_part = pd.read_csv(csv_file)
                        table = pd.concat([table, df_part], axis=0, ignore_index=True)
                    else:
                        raise NotImplementedError("use_pandas=False is not supported here.")
        return table

    raise NotImplementedError("in_memory=False is not implemented in this app.")


def make_game_time_space_tensor_both(
    df: pd.DataFrame,
    grid_x_bins: int = 17,
    grid_y_bins: int = 16,
    time_bin_seconds: int = 720,
):
    """
    Create game × time × position × channels tensor.
    
    Args:
        df: DataFrame with NBA shot data
        grid_x_bins: Number of spatial bins in X direction
        grid_y_bins: Number of spatial bins in Y direction
        time_bin_seconds: Duration of each time bin in seconds
        
    Returns:
        tuple: (tensor, metadata_dict)
            tensor shape: (games, time_bins, spatial_cells, 5)
            channels: 0=attempts, 1=makes, 2=points, 3=efg_weights, 4=misses
    """
    required_cols = {
        "LOC_X", "LOC_Y",
        "PERIOD",
        "MINUTES_REMAINING", "SECONDS_REMAINING",
        "GAME_ID",
        "SHOT_MADE_FLAG",
        "SHOT_TYPE",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Limit to first 4 quarters
    df = df[df["PERIOD"] <= 4].copy()

    # Calculate elapsed time from tip-off
    df["ELAPSED_SEC"] = (
        (df["PERIOD"] - 1) * 720 +
        (720 - (df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]))
    )

    total_duration = 4 * 12 * 60  # 48 minutes
    num_time_bins = int(np.ceil(total_duration / time_bin_seconds))

    # Grid edges for court
    x_edges = np.linspace(-250, 250, grid_x_bins + 1)
    y_edges = np.linspace(-47.5, 422.5, grid_y_bins + 1)

    game_ids = sorted(df["GAME_ID"].unique())

    # 4D tensor (games, time, y, x, 4 channels: attempts, makes, points, efg_weights)
    data_4d = np.zeros(
        (len(game_ids), num_time_bins, grid_y_bins, grid_x_bins, 4),
        dtype=np.float32,
    )

    for g_idx, gid in enumerate(game_ids):
        game_df = df[df["GAME_ID"] == gid].copy()
        game_df["time_bin"] = (game_df["ELAPSED_SEC"] // time_bin_seconds).astype(int)
        game_df["x_bin"] = np.digitize(game_df["LOC_X"], x_edges) - 1
        game_df["y_bin"] = np.digitize(game_df["LOC_Y"], y_edges) - 1

        # Filter out-of-grid shots
        game_df = game_df[
            (game_df["x_bin"] >= 0) & (game_df["x_bin"] < grid_x_bins) &
            (game_df["y_bin"] >= 0) & (game_df["y_bin"] < grid_y_bins)
        ]

        for _, row in game_df.iterrows():
            t = int(row["time_bin"])
            y = int(row["y_bin"])
            x = int(row["x_bin"])
            if t >= num_time_bins:
                continue

            # Channel 0: Attempts
            data_4d[g_idx, t, y, x, 0] += 1.0

            # Channel 1, 2, 3: Makes, actual points, and EFG weights
            if int(row["SHOT_MADE_FLAG"]) == 1:
                data_4d[g_idx, t, y, x, 1] += 1.0
                
                shot_type = str(row["SHOT_TYPE"])
                is_3pt = "3PT" in shot_type
                pts = 3.0 if is_3pt else 2.0  # Actual point values
                efg_weight = 1.5 if is_3pt else 1.0  # EFG weights for EFG% calculation
                data_4d[g_idx, t, y, x, 2] += pts
                data_4d[g_idx, t, y, x, 3] += efg_weight

    # Add channel 4: Misses (Attempts - Makes)
    # ★変更: チャンネル数を 5 から 6 に変更
    data_6ch = np.zeros(
        (len(game_ids), num_time_bins, grid_y_bins, grid_x_bins, 6),
        dtype=np.float32,
    )
    data_6ch[:, :, :, :, :4] = data_4d
    data_6ch[:, :, :, :, 4] = data_4d[:, :, :, :, 0] - data_4d[:, :, :, :, 1]  # misses = attempts - makes
    
    # ★追加: Channel 5: Frequency (初期値としてAttemptsをコピー。後で正規化してFrequencyにする)
    data_6ch[:, :, :, :, 5] = data_4d[:, :, :, :, 0]

    # Reshape: (games, time, y, x, 6) → (games, time, y*x, 6)
    tensor = data_6ch.reshape(
        len(game_ids),
        num_time_bins,
        grid_y_bins * grid_x_bins,
        6,  # 5 -> 6
    )

    meta = {
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "game_ids": game_ids,
        "num_time_bins": num_time_bins,
        "grid_size": grid_y_bins * grid_x_bins,
        "grid_x_bins": grid_x_bins,
        "grid_y_bins": grid_y_bins,
    }

    return tensor, meta
