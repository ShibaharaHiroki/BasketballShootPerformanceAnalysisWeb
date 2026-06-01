"""
Build NBA data cache as .npy files.

Usage:
    python build_nba_cache.py [--seasons 2022 2023 ...] [--output-dir data]

This script downloads NBA shot detail data from GitHub and saves it as .npy files
so the backend API can load data instantly without network access.

Saved files (per season):
    data/nba_shotdetail_{season}_data.npy    - shot data as float32/object array
    data/nba_shotdetail_{season}_cols.txt    - column names (one per line)
    data/nba_shotdetail_{season}_dtypes.txt  - dtype hints (one per line)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make sure we can import from core/
sys.path.insert(0, str(Path(__file__).parent))
from core.data_loader import load_nba_data


def save_dataframe_as_npy(df: pd.DataFrame, base_path: Path) -> None:
    """
    Save a pandas DataFrame to .npy and .txt files.

    Files created:
        <base_path>_data.npy   - values as object array (preserves mixed types)
        <base_path>_cols.txt   - column names
        <base_path>_dtypes.txt - original dtype names for reconstruction hints
    """
    # Save column names
    cols_path = Path(str(base_path) + "_cols.txt")
    cols_path.write_text("\n".join(df.columns.tolist()), encoding="utf-8")

    # Save dtype hints
    dtypes_path = Path(str(base_path) + "_dtypes.txt")
    dtypes_path.write_text("\n".join([str(dt) for dt in df.dtypes.tolist()]), encoding="utf-8")

    # Save values as object array (handles mixed int/float/str)
    data_path = Path(str(base_path) + "_data.npy")
    np.save(data_path, df.values)

    print(f"  Saved data  : {data_path}  ({data_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Saved cols  : {cols_path}")
    print(f"  Saved dtypes: {dtypes_path}")


def load_dataframe_from_npy(base_path: Path) -> pd.DataFrame:
    """
    Load a pandas DataFrame from .npy and .txt files.
    """
    data_path = Path(str(base_path) + "_data.npy")
    cols_path = Path(str(base_path) + "_cols.txt")
    dtypes_path = Path(str(base_path) + "_dtypes.txt")

    if not data_path.exists():
        raise FileNotFoundError(f"Cache not found: {data_path}")

    values = np.load(data_path, allow_pickle=True)
    columns = cols_path.read_text(encoding="utf-8").strip().splitlines()
    dtype_strs = dtypes_path.read_text(encoding="utf-8").strip().splitlines()

    df = pd.DataFrame(values, columns=columns)

    # Restore dtypes
    for col, dtype_str in zip(columns, dtype_strs):
        try:
            if "int" in dtype_str:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif "float" in dtype_str:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            # object/str columns stay as-is
        except Exception:
            pass  # keep as-is if conversion fails

    return df


def build_cache(seasons: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        print(f"\n[Season {season}] Downloading NBA shot detail data...")
        df = load_nba_data(
            seasons=(season,),
            data=("shotdetail",),
            seasontype="rg",
            league="nba",
            in_memory=True,
            use_pandas=True,
        )
        print(f"  Downloaded {len(df)} rows, {len(df.columns)} columns.")

        base_path = output_dir / f"nba_shotdetail_{season}"
        save_dataframe_as_npy(df, base_path)

    print("\nAll seasons cached successfully.")


def main():
    parser = argparse.ArgumentParser(description="Build NBA data cache as .npy files.")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2022],
        help="Season year(s) to download (e.g. 2022 2023). Default: 2022",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save .npy files. Default: data/",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    build_cache(args.seasons, output_dir)


if __name__ == "__main__":
    main()
