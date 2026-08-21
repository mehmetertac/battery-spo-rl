"""Load DE-LU day-ahead prices from Open Power System Data (OPSD)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

OPSD_VERSION = "2020-10-06"
OPSD_CSV_URL = (
    f"https://data.open-power-system-data.org/time_series/{OPSD_VERSION}/"
    "time_series_60min_singleindex.csv"
)
DEFAULT_CACHE_PATH = Path("data/opsd/time_series_60min_singleindex.csv")
PRICE_COLUMN = "DE_LU_price_day_ahead"
TIMESTAMP_COLUMN = "cet_cest_timestamp"
MIN_HISTORY_HOURS = 168


def _find_price_column(columns: pd.Index) -> str:
    if PRICE_COLUMN in columns:
        return PRICE_COLUMN

    candidates = [
        col
        for col in columns
        if "DE_LU" in col and "price" in col.lower() and "day_ahead" in col.lower()
    ]
    if candidates:
        return candidates[0]

    raise ValueError(
        f"Could not find DE-LU day-ahead price column. "
        f"Expected {PRICE_COLUMN!r}. Available columns include: {list(columns[:10])} ..."
    )


def download_opsd_csv(cache_path: Path | None = None, force: bool = False) -> Path:
    """Download OPSD 60-min singleindex CSV if not already cached."""
    cache_path = DEFAULT_CACHE_PATH if cache_path is None else Path(cache_path)
    if cache_path.exists() and not force:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OPSD time series to {cache_path} ...")
    urllib.request.urlretrieve(OPSD_CSV_URL, cache_path)
    return cache_path


def load_opsd_prices(cache_path: Path | None = None) -> pd.DataFrame:
    """Load hourly DE-LU day-ahead prices with Europe/Berlin timestamps."""
    path = download_opsd_csv(cache_path)
    raw = pd.read_csv(path, usecols=[TIMESTAMP_COLUMN, PRICE_COLUMN], low_memory=False)

    price_col = _find_price_column(raw.columns)
    frame = raw.rename(columns={price_col: "price_eur_mwh"})
    frame["timestamp"] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    frame = frame.drop(columns=[TIMESTAMP_COLUMN])
    frame = frame.dropna(subset=["price_eur_mwh"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame = frame.set_index("timestamp")

    return frame


def load_de_day_ahead(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return DE-LU day-ahead prices, optionally sliced to [start, end)."""
    prices = load_opsd_prices(cache_path)

    if start is not None:
        start_ts = pd.Timestamp(start, tz="Europe/Berlin")
        prices = prices.loc[prices.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end, tz="Europe/Berlin")
        prices = prices.loc[prices.index < end_ts]

    if prices.empty:
        raise ValueError("No price data available for the requested date range.")

    return prices


def available_day_range(prices: pd.DataFrame | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return first and last calendar dates with complete 24h coverage."""
    if prices is None:
        prices = load_opsd_prices()

    daily_counts = prices["price_eur_mwh"].groupby(prices.index.date).count()
    complete_days = daily_counts[daily_counts == 24].index
    if len(complete_days) == 0:
        raise ValueError("No complete 24-hour days found in OPSD price data.")

    first_day = pd.Timestamp(complete_days[0], tz="Europe/Berlin")
    last_day = pd.Timestamp(complete_days[-1], tz="Europe/Berlin")
    return first_day, last_day


def extract_day_prices(prices: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    """Extract 24 hourly prices for a calendar day in Europe/Berlin."""
    day_start = pd.Timestamp(day.date(), tz="Europe/Berlin")
    day_end = day_start + pd.Timedelta(days=1)
    day_prices = prices.loc[(prices.index >= day_start) & (prices.index < day_end), "price_eur_mwh"]

    if len(day_prices) != 24:
        raise ValueError(
            f"Expected 24 hourly prices for {day_start.date()}, found {len(day_prices)}."
        )

    return day_prices.reset_index(drop=True)
