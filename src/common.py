import pathlib
from typing import Any, List

import pandas as pd
import polars as pl

from utils.io import load_pickle


def drop_multi_game_naive(
    df: pd.DataFrame, local: bool = True
) -> pd.DataFrame:
    """Drop events not occurring at the first game play.

    Parameters:
        df: input DataFrame

    Return:
        df: DataFrame with events occurring at the first game play only
    """
    if local:
        df["lv_diff"] = (
            df.groupby("session_id")
            .apply(lambda x: x["level"].diff().fillna(0))
            .values
        )
    else:
        df["lv_diff"] = df["level"].diff().fillna(0)
    reversed_lv_pts = df["lv_diff"] < 0
    df.loc[~reversed_lv_pts, "lv_diff"] = 0
    if local:
        df["multi_game_flag"] = df.groupby("session_id")["lv_diff"].cumsum()
    else:
        df["multi_game_flag"] = df["lv_diff"].cumsum()
    multi_game_mask = df["multi_game_flag"] < 0
    multi_game_rows = df[multi_game_mask].index
    df = df.drop(multi_game_rows).reset_index(drop=True)

    return df


# ====================
# Create features
# ====================
def parse_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.assign(
        level=labels["session_id"].str.extract(r"q(\d+)").astype("int64"),
        session=labels["session_id"].str.extract(r"(\d+)")[0].astype(int),
    )
    return labels


def create_features(
    data: pl.DataFrame,
    level_group: str,
    input_dir: pathlib.Path,
) -> pl.DataFrame:
    data = data.drop(["fullscreen", "hq", "music"])
    uniques_map = load_pickle(input_dir / "uniques_map.pkl")[level_group]

    columns = [
        (
            pl.col("elapsed_time")
            .diff(n=1)
            .fill_null(0)
            .clip(0, 1e9)
            .over(["session_id"])
            .alias("elapsed_time_diff")
        ),
        (
            pl.col("screen_coor_x")
            .diff(n=1)
            .abs()
            .over(["session_id"])
            .alias("location_x_diff")
        ),
        (
            pl.col("screen_coor_y")
            .diff(n=1)
            .abs()
            .over("session_id")
            .alias("location_y_diff")
        ),
        pl.col("level").cast(pl.Utf8),
        pl.col("page").cast(pl.Utf8).fill_null("page_null"),
        pl.col("event_name").fill_null("event_name_null"),
        pl.col("name").fill_null("name_null"),
        pl.col("text").fill_null("text_null"),
        pl.col("fqid").fill_null("fqid_null"),
        pl.col("room_fqid").fill_null("room_fqid_null"),
        pl.col("text_fqid").fill_null("text_fqid_null"),
    ]
    data = data.with_columns(columns)

    dialogs = [
        "that",
        "this",
        "it",
        "you",
        "find",
        "found",
        "Found",
        "notebook",
        "Wells",
        "wells",
        "help",
        "need",
        "Oh",
        "Ooh",
        "Jo",
        "flag",
        "can",
        "and",
        "is",
        "the",
        "to",
    ]

    # Categorical features.
    categorical_uniques = {
        "event_name": uniques_map["event_name"] + ["event_name_null"],
        "name": uniques_map["name"] + ["name_null"],
        "fqid": uniques_map["fqid"] + ["fiqd_null"],
        "room_fqid": uniques_map["room_fqid"] + ["room_fqid_null"],
        "text_fqid": uniques_map["text_fqid"],
        "level": uniques_map["level"],
        "page": uniques_map["page"] + ["page_null"],
    }

    agg_features: List[Any] = []

    agg_features += [
        pl.col("index").count().alias("nrows"),
    ]
    agg_features += [
        pl.col("index")
        .filter(pl.col("text").str.contains(c))
        .count()
        .alias(f"word_{c}")
        for c in dialogs
    ]
    agg_features += [
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .mean()
        .alias(f"word_mean_{c}")
        for c in dialogs
    ]
    agg_features += [
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .std()
        .alias(f"word_std_{c}")
        for c in dialogs
    ]
    agg_features += [
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .max()
        .alias(f"word_max_{c}")
        for c in dialogs
    ]
    agg_features += [
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .sum()
        .alias(f"word_sum_{c}")
        for c in dialogs
    ]
    agg_features += [
        pl.col("elapsed_time_diff")
        .filter((pl.col("text").str.contains(c)))
        .median()
        .alias(f"word_median_{c}")
        for c in dialogs
    ]
    agg_features += [
        pl.col(col).drop_nulls().n_unique().alias(f"{col}_nunique")
        for col in ["event_name", "fqid", "room_fqid", "text"]
    ]
    for col, uniques in categorical_uniques.items():
        agg_features += [
            pl.col(col)
            .filter(pl.col(col) == u)
            .count()
            .alias(f"{col}_{u}_count")
            for u in uniques
        ]

        agg_cols = [
            "elapsed_time_diff",
            "hover_duration",
        ]
        for agg_col in agg_cols:
            agg_features += [
                pl.col(agg_col)
                .filter(pl.col(col) == u)
                .sum()
                .alias(f"{col}_{u}_{agg_col}_sum")
                for u in uniques
            ]
            agg_features += [
                pl.col(agg_col)
                .filter(pl.col(col) == u)
                .mean()
                .alias(f"{col}_{u}_{agg_col}_mean")
                for u in uniques
            ]
            agg_features += [
                pl.col(agg_col)
                .filter(pl.col(col) == u)
                .std()
                .alias(f"{col}_{u}_{agg_col}_std")
                for u in uniques
            ]
            agg_features += [
                pl.col(agg_col)
                .filter(pl.col(col) == u)
                .max()
                .alias(f"{col}_{u}_{agg_col}_max")
                for u in uniques
            ]
            agg_features += [
                pl.col(agg_col)
                .filter(pl.col(col) == u)
                .min()
                .alias(f"{col}_{u}_{agg_col}_min")
                for u in uniques
            ]

    # Numeric features.
    NUMS = [
        "elapsed_time_diff",
        "hover_duration",
        "room_coor_x",
        "room_coor_y",
        "screen_coor_x",
        "screen_coor_y",
        # TESTING===================
        "location_x_diff",
        "location_y_diff",
        # ==========================
    ]

    def minmax_func(s) -> float:
        return 0.0 if s.is_empty() else s.max() - s.min()

    agg_features += [
        pl.col("elapsed_time").drop_nulls().sum().alias("elapsed_time_sum")
    ]
    agg_features += [
        pl.col(c).drop_nulls().mean().alias(f"{c}_mean") for c in NUMS
    ]
    agg_features += [
        pl.col(c).drop_nulls().std().alias(f"{c}_std") for c in NUMS
    ]
    agg_features += [
        pl.col(c).drop_nulls().sum().alias(f"{c}_sum") for c in NUMS
    ]
    # ============================
    for q_tile in [0.1, 0.2, 0.5, 0.75]:
        agg_features += [
            pl.col(c).quantile(q_tile, "nearest").alias(f"{c}_qtile_{q_tile}")
            for c in NUMS
        ]

    if level_group == "5-12":
        for col in ["elapsed_time", "index"]:
            agg_features += [
                pl.col(col)
                .filter(
                    (pl.col("text") == "Here's the log book.")
                    | (pl.col("fqid") == "logbook.page.bingo")
                )
                .apply(minmax_func)
                .alias(f"logbook_bingo_minmax_{col}"),
                pl.col(col)
                .filter(
                    (
                        (pl.col("event_name") == "navigate_click")
                        & (pl.col("fqid") == "reader")
                    )
                    | (pl.col("fqid") == "reader.paper2.bingo")
                )
                .apply(minmax_func)
                .alias(f"reader_bingo_minmax_{col}"),
                pl.col(col)
                .filter(
                    (
                        (pl.col("event_name") == "navigate_click")
                        & (pl.col("fqid") == "journals")
                    )
                    | (pl.col("fqid") == "journals.pic_2.bingo")
                )
                .apply(minmax_func)
                .alias(f"journals_bingo_minmax_{col}"),
            ]
    if level_group == "13-22":
        for col in ["elapsed_time", "index"]:
            agg_features += [
                pl.col(col)
                .filter(
                    (
                        (pl.col("event_name") == "navigate_click")
                        & (pl.col("fqid") == "reader_flag")
                    )
                    | (
                        pl.col("fqid")
                        == "tunic.library.microfiche.reader_flag.paper2.bingo"
                    )
                )
                .apply(minmax_func)
                .alias(f"reader_flag_minmax_{col}"),
                pl.col(col)
                .filter(
                    (
                        (pl.col("event_name") == "navigate_click")
                        & (pl.col("fqid") == "journals_flag")
                    )
                    | (pl.col("fqid") == "journals_flag.pic_0.bingo")
                )
                .apply(minmax_func)
                .alias(f"journalsFlag_bingo_mimmax_{col}"),
            ]

    results = (
        data.groupby("session_id", maintain_order=True)
        .agg(agg_features)
        .fill_nan(-1)
    )

    return results


def pad_sequence(seq: List, max_seq_len: int, pad_value: Any) -> List:
    if len(seq) >= max_seq_len:
        return seq[-max_seq_len:]
    else:
        return [pad_value] * (max_seq_len - len(seq)) + seq


def create_sequence_features(
    X: pd.DataFrame, y: pd.DataFrame, max_seq_len: int
) -> pd.DataFrame:
    features = X.groupby("session_id")[
        ["elapsed_time_diff", "event_name", "level", "fqid", "room_fqid"]
    ].agg(lambda x: pad_sequence(list(x), max_seq_len, 0))
    labels = y.groupby("session")["correct"].apply(list)
    data = pd.merge(
        labels, features, how="left", left_index=True, right_index=True
    )
    return data
