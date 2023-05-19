import pathlib
from typing import Any, List

import pandas as pd
import polars as pl

from utils.io import load_pickle


def drop_null_columns(data: pl.DataFrame, columns: List) -> pl.DataFrame:
    return data.drop(columns)


"""
- 特徴量は規定されたlevel_groupごとに生成する
- labelは規定されたlevel_groupとは異なる範囲で分割する
- 上記の異なるlevel_groupの定義を持つデータをjoinする
"""


# ====================
# Create features
# ====================
def parse_labels(labels: pd.DataFrame) -> pl.DataFrame:
    labels.rename(columns={"session_id": "session_level"}, inplace=True)
    labels = labels.assign(
        level=labels["session_level"].str.extract(r"q(\d+)").astype("int64"),
        session_id=labels["session_level"]
        .str.extract(r"(\d+)")[0]
        .astype(int),
    )
    labels.drop(columns=["session_level"], inplace=True)

    labels_pl = pl.from_pandas(labels)
    labels_pl = labels_pl.with_columns(
        pl.when(pl.col("level") < 4)
        .then("0-4")
        .otherwise(
            pl.when(pl.col("level") < 14).then("5-12").otherwise("13-22")
        )
        .alias("level_group")
    )
    return labels_pl


def create_features(
    data: pl.DataFrame,
    uniques_dirpath: str = "./data/preprocessing",
) -> pl.DataFrame:
    agg_groups = ["session_id", "level_group"]
    columns = [
        (
            (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1))
            .fill_null(0)
            .clip(0, 1e9)
            .over(agg_groups)
            .alias("elapsed_time_diff")
        ),
        (
            (pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1))
            .abs()
            .over(agg_groups)
            .alias("location_x_diff")
        ),
        (
            (pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1))
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
    uniques_dir = pathlib.Path(uniques_dirpath)
    event_name = load_pickle(str(uniques_dir / "uniques_event_name.pkl"))
    name = load_pickle(str(uniques_dir / "uniques_name.pkl"))
    # text = load_pickle(str(uniques_dir / "uniques_text.pkl"))
    fqid = load_pickle(str(uniques_dir / "uniques_fqid.pkl"))
    room_fqid = load_pickle(str(uniques_dir / "uniques_room_fqid.pkl"))
    text_fqid = load_pickle(str(uniques_dir / "uniques_text_fqid.pkl"))
    categorical_uniques = {
        "event_name": event_name + ["event_name_null"],
        # "name": name + ["name_null"],
        # "text": text + ["text_null"],
        # "fqid": fqid + ["fiqd_null"],
        "room_fqid": room_fqid + ["room_fqid_null"],
        # "text_fqid": text_fqid,
        # "level": list(range(1, 23)),
    }

    agg_features: List[Any] = []
    # agg_features += [
    #     pl.col("index").count().alias("nrows"),
    # ]
    agg_features += [
        pl.col(col).drop_nulls().n_unique().alias(f"nunique_{col}")
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
        "elapsed_time",
        "elapsed_time_diff",
        # "location_x_diff",
        # "location_y_diff",
        "hover_duration",
        "page",
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
    # agg_features += [
    #     pl.col(c).drop_nulls().sum().alias(f"{c}_sum") for c in NUMS
    # ]
    # agg_features += [
    #     pl.col(c).drop_nulls().median().alias(f"{c}_median") for c in NUMS
    # ]
    # agg_features += [
    #     pl.col(c).drop_nulls().min().alias(f"{c}_min") for c in NUMS
    # ]
    # agg_features += [
    #     pl.col(c).drop_nulls().max().alias(f"{c}_max") for c in NUMS
    # ]
    # agg_features += [
    #     pl.col(c).drop_nulls().first().alias(f"{c}_first") for c in NUMS
    # ]
    # agg_features += [
    #     pl.col(c).drop_nulls().last().alias(f"{c}_last") for c in NUMS
    # ]
    # for q_tile in [0.1, 0.2, 0.5, 0.75]:
    #     agg_features += [
    #         pl.col(c).quantile(q_tile, "nearest").alias(f"{c}_qtile_{q_tile}")
    #         for c in NUMS
    #     ]

    # NOTE: Features of level_group="5-12".
    # agg_features += [
    #     pl.col("elapsed_time")
    #     .filter(
    #         (pl.col("text") == "Here's the log book.")
    #         | (pl.col("fqid") == "logbook.page.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("logbook_bingo_duration"),
    #     pl.col("index")
    #     .filter(
    #         (pl.col("text") == "Here's the log book.")
    #         | (pl.col("fqid") == "logbook.page.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("logbook_bingo_indexCount"),
    #     pl.col("elapsed_time")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "reader")
    #         )
    #         | (pl.col("fqid") == "reader.paper2.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("reader_bingo_duration"),
    #     pl.col("index")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "reader")
    #         )
    #         | (pl.col("fqid") == "reader.paper2.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("reader_bingo_indexCount"),
    #     pl.col("elapsed_time")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "journals")
    #         )
    #         | (pl.col("fqid") == "journals.pic_2.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("journals_bingo_duration"),
    #     pl.col("index")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "journals")
    #         )
    #         | (pl.col("fqid") == "journals.pic_2.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("journals_bingo_indexCount"),
    # ]
    # # NOTE: Features of level_group="13-12".
    # agg_features += [
    #     pl.col("elapsed_time")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "reader_flag")
    #         )
    #         | (
    #             pl.col("fqid")
    #             == "tunic.library.microfiche.reader_flag.paper2.bingo"
    #         )
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("reader_flag_duration"),
    #     pl.col("index")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "reader_flag")
    #         )
    #         | (
    #             pl.col("fqid")
    #             == "tunic.library.microfiche.reader_flag.paper2.bingo"
    #         )
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("reader_flag_indexCount"),
    #     pl.col("elapsed_time")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "journals_flag")
    #         )
    #         | (pl.col("fqid") == "journals_flag.pic_0.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("journalsFlag_bingo_duration"),
    #     pl.col("index")
    #     .filter(
    #         (
    #             (pl.col("event_name") == "navigate_click")
    #             & (pl.col("fqid") == "journals_flag")
    #         )
    #         | (pl.col("fqid") == "journals_flag.pic_0.bingo")
    #     )
    #     .apply(lambda s: 0.0 if s.is_empty() else s.max() - s.min())  # type: ignore
    #     .alias("journalsFlag_bingo_indexCount"),
    # ]

    results = (
        data.groupby(agg_groups, maintain_order=True)
        .agg(agg_features)
        .fill_nan(-1)
    )
    return results
