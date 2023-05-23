from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl


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
    uniqes_map: Dict[str, List[str]],
) -> pl.DataFrame:
    data = data.drop(["fullscreen", "hq", "music"])

    agg_groups = ["session_id"]
    columns = [
        (
            pl.col("elapsed_time")
            .diff(n=1)
            .fill_null(0)
            .clip(0, 1e9)
            .over(agg_groups)
            .alias("elapsed_time_diff")
        ),
        (
            pl.col("screen_coor_x")
            .diff(n=1)
            .abs()
            .over(agg_groups)
            .alias("location_x_diff")
        ),
        (
            pl.col("screen_coor_y")
            .diff(n=1)
            .abs()
            .over(agg_groups)
            .alias("location_y_diff")
        ),
        pl.col("event_name").fill_null("event_name_null"),
        pl.col("name").fill_null("name_null"),
        pl.col("text").fill_null("text_null"),
        pl.col("fqid").fill_null("fqid_null"),
        pl.col("room_fqid").fill_null("room_fqid_null"),
        pl.col("text_fqid").fill_null("text_fqid_null"),
    ]
    data = data.with_columns(columns)

    # Categorical features.
    categorical_uniques = {
        "event_name": uniqes_map["event_name"] + ["event_name_null"],
        # "name": uniqes_map["name"] + ["name_null"],  # NOTE: Not improve cv
        "fqid": uniqes_map["fqid"] + ["fiqd_null"],
        "room_fqid": uniqes_map["room_fqid"] + ["room_fqid_null"],
        "text_fqid": uniqes_map["text_fqid"],
        # "level": list(range(1, 23)),  # NOTE: Not improve cv
    }

    agg_features: List[Any] = []

    # NOTE: Not Improve LB score, decrease to 0.621.
    # agg_features += [
    #     pl.col("index").count().alias("nrows"),
    # ]

    # agg_features += [
    #     pl.col("index")
    #     .filter(pl.col("text").str.contains(c))
    #     .count()
    #     .alias(f"word_{c}")
    #     for c in dialogs
    # ]
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
            # "location_x_diff",
            # "location_y_diff",
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
        #     agg_features += [
        #         pl.col(agg_col)
        #         .filter(pl.col(col) == u)
        #         .std()
        #         .alias(f"{col}_{u}_{agg_col}_std")
        #         for u in uniques
        #     ]
        #     agg_features += [
        #         pl.col(agg_col)
        #         .filter(pl.col(col) == u)
        #         .max()
        #         .alias(f"{col}_{u}_{agg_col}_max")
        #         for u in uniques
        #     ]
        #     agg_features += [
        #         pl.col(agg_col)
        #         .filter(pl.col(col) == u)
        #         .min()
        #         .alias(f"{col}_{u}_{agg_col}_min")
        #         for u in uniques
        #     ]

    # Numeric features.
    NUMS = [
        "elapsed_time_diff",
        # "location_x_diff",
        # "location_y_diff",
        "hover_duration",
        "page",  # NOTE: as a categorical features???
        "room_coor_x",
        "room_coor_y",
        "screen_coor_x",
        "screen_coor_y",
        # "fullscreen",
        # "hq",
        # "music",
    ]

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
    agg_features += [
        pl.col(c).drop_nulls().median().alias(f"{c}_median") for c in NUMS
    ]
    agg_features += [
        pl.col(c).drop_nulls().min().alias(f"{c}_min") for c in NUMS
    ]
    agg_features += [
        pl.col(c).drop_nulls().max().alias(f"{c}_max") for c in NUMS
    ]
    agg_features += [
        pl.col(c).drop_nulls().first().alias(f"{c}_first") for c in NUMS
    ]
    agg_features += [
        pl.col(c).drop_nulls().last().alias(f"{c}_last") for c in NUMS
    ]
    for q_tile in [0.1, 0.2, 0.5, 0.75]:
        agg_features += [
            pl.col(c).quantile(q_tile, "nearest").alias(f"{c}_qtile_{q_tile}")
            for c in NUMS
        ]

    if level_group == "5-12":
        agg_features += [
            pl.col("elapsed_time")
            .filter(
                (pl.col("text") == "Here's the log book.")
                | (pl.col("fqid") == "logbook.page.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("logbook_bingo_duration"),
            pl.col("index")
            .filter(
                (pl.col("text") == "Here's the log book.")
                | (pl.col("fqid") == "logbook.page.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("logbook_bingo_indexCount"),
            pl.col("elapsed_time")
            .filter(
                (
                    (pl.col("event_name") == "navigate_click")
                    & (pl.col("fqid") == "reader")
                )
                | (pl.col("fqid") == "reader.paper2.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("reader_bingo_duration"),
            pl.col("index")
            .filter(
                (
                    (pl.col("event_name") == "navigate_click")
                    & (pl.col("fqid") == "reader")
                )
                | (pl.col("fqid") == "reader.paper2.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("reader_bingo_indexCount"),
            pl.col("elapsed_time")
            .filter(
                (
                    (pl.col("event_name") == "navigate_click")
                    & (pl.col("fqid") == "journals")
                )
                | (pl.col("fqid") == "journals.pic_2.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("journals_bingo_duration"),
            pl.col("index")
            .filter(
                (
                    (pl.col("event_name") == "navigate_click")
                    & (pl.col("fqid") == "journals")
                )
                | (pl.col("fqid") == "journals.pic_2.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("journals_bingo_indexCount"),
        ]
    if level_group == "13-22":
        agg_features += [
            pl.col("elapsed_time")
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
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("reader_flag_duration"),
            pl.col("index")
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
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("reader_flag_indexCount"),
            pl.col("elapsed_time")
            .filter(
                (
                    (pl.col("event_name") == "navigate_click")
                    & (pl.col("fqid") == "journals_flag")
                )
                | (pl.col("fqid") == "journals_flag.pic_0.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("journalsFlag_bingo_duration"),
            pl.col("index")
            .filter(
                (
                    (pl.col("event_name") == "navigate_click")
                    & (pl.col("fqid") == "journals_flag")
                )
                | (pl.col("fqid") == "journals_flag.pic_0.bingo")
            )
            .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
            .alias("journalsFlag_bingo_indexCount"),
        ]

    results = (
        data.groupby(agg_groups, maintain_order=True)
        .agg(agg_features)
        .fill_nan(-1)
    )
    return results
