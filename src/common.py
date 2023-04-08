from typing import List, Union

import polars as pl

from utils.io import load_pickle


def drop_null_columns(data: pl.DataFrame, columns: List) -> pl.DataFrame:
    return data.drop(columns)


def clip_values(
    data: pl.DataFrame,
    columns: str,
    min_val: Union[int, float],
    max_val: Union[int, float],
) -> pl.DataFrame:
    data = data.with_columns([pl.col(columns).clip(min_val, max_val).alias(columns)])
    return data


def cutoff_fqid(data: pl.DataFrame) -> pl.DataFrame:
    cols_non_cutoff = [
        "worker",
        "archivist",
        "gramps",
        "wells",
        "toentry",
        "confrontation",
        "crane_ranger",
        "groupconvo",
        "flag_girl",
        "tomap",
        "tostacks",
        "tobasement",
        "archivist_glasses",
        "boss",
        "journals",
        "seescratches",
        "groupconvo_flag",
        "cs",
        "teddy",
        "expert",
        "businesscards",
        "ch3start",
        "tunic.historicalsociety",
        "tofrontdesk",
        "savedteddy",
        "plaque",
        "glasses",
        "tunic.drycleaner",
    ]
    result = data.select(
        pl.when(pl.col("fqid").is_in(cols_non_cutoff))
        .then("other")
        .otherwise(pl.col("fqid"))
    )
    return result


def split_room_fqid(data: pl.DataFrame) -> pl.DataFrame:
    room_fqid = (
        data["room_fqid"]
        .str.split_exact(".", n=2)
        .struct.rename_fields([f"room_fqid_{i}" for i in range(3)])
        .alias("fields")
        .to_frame()
        .unnest("fields")
    )
    data = pl.concat([data, room_fqid], how="horizontal").drop(
        [
            # "room_fqid",
            "room_fqid_0",
        ]
    )
    return data


def preprocessing(data: pl.DataFrame) -> pl.DataFrame:
    data = drop_null_columns(data, ["page", "hover_duration"])
    data = clip_values(data, "elapsed_time", 0, 1e7)
    # data = cutoff_fqid(data)  # NOTE: Modelがいい感じに学習してくれるかもしれないので一旦やらない
    data = split_room_fqid(data)
    return data


# ====================
# Create features
# ====================


def create_features(
    data: pl.DataFrame,
    uniques_dir: str = "./data/preprocessing",
    is_test: bool = False,
) -> pl.DataFrame:
    # TODO:
    # tf-idfでベクトル変換する <- ここまでやるならnnの方がいいかも
    #   でも今回の問題設定がlevel_groupごとの予測なので、eventがsequenceとも捉えがたい

    if is_test:
        data = preprocessing(data)

    columns = [
        (
            (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1))
            .fill_null(0)
            .clip(0, 1e7)
            .over(["session_id"])
            .alias("elapsed_time_diff")
        ),
        (
            (pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1))
            .abs()
            .over(["session_id"])
            .alias("location_x_diff")
        ),
        (
            (pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1))
            .abs()
            .over(["session_id"])
            .alias("location_y_diff")
        ),
        pl.col("event_name").fill_null("null"),
        pl.col("name").fill_null("null"),
        pl.col("fqid").fill_null("null"),
        pl.col("text_fqid").fill_null("null"),
        pl.col("room_fqid").fill_null("null"),
        pl.col("room_fqid_1").fill_null("null"),
        pl.col("room_fqid_2").fill_null("null"),
    ]
    data = data.with_columns(columns)

    CATS = [
        "event_name",
        "name",
        "level",
        "fqid",
        "text_fqid",
        "room_fqid",
        "room_fqid_1",
        "room_fqid_2",
    ]
    NUMS = [
        "elapsed_time_diff",
        "room_coor_x",
        "room_coor_y",
        "screen_coor_x",
        "screen_coor_y",
        "fullscreen",
        "hq",
        "music",
    ]

    agg_features = [
        pl.col("index").count().alias("nrows"),
    ]

    # Categorical features.
    agg_features += [
        pl.col(c).drop_nulls().n_unique().alias(f"nunique_{c}") for c in CATS
    ]
    event_names = load_pickle(f"{uniques_dir}/uniques_event_name.pkl")
    names = load_pickle(f"{uniques_dir}/uniques_name.pkl")
    fqids = load_pickle(f"{uniques_dir}/uniques_fqid.pkl")
    room_fqid = load_pickle(f"{uniques_dir}/uniques_room_fqid.pkl")
    room_fqid_1 = load_pickle(f"{uniques_dir}/uniques_room_fqid_1.pkl")
    room_fqid_2 = load_pickle(f"{uniques_dir}/uniques_room_fqid_2.pkl")

    categorical_uniques = {
        "event_name": event_names,
        "name": names,
        "fqid": fqids,
        "room_fqid": room_fqid,
        "room_fqid_1": room_fqid_1,
        "room_fqid_2": room_fqid_2,
    }
    for col, uniques in categorical_uniques.items():
        agg_features += [
            pl.col(col).filter(pl.col(col) == u).count().alias(f"{col}_{u}_count")
            for u in uniques
        ]

        agg_cols = ["elapsed_time_diff", "location_x_diff", "location_y_diff"]
        for agg_col in agg_cols:
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
    agg_features += [pl.col(c).drop_nulls().mean().alias(f"{c}_mean") for c in NUMS]
    agg_features += [pl.col(c).drop_nulls().std().alias(f"{c}_std") for c in NUMS]
    agg_features += [pl.col(c).drop_nulls().min().alias(f"{c}_min") for c in NUMS]
    agg_features += [pl.col(c).drop_nulls().max().alias(f"{c}_max") for c in NUMS]

    for q_tile in [0.25, 0.5, 0.75]:
        agg_features += [
            pl.col(c).quantile(q_tile).alias(f"{c}_qtile_{q_tile}") for c in NUMS
        ]

    results = data.groupby(["session_id"], maintain_order=True).agg(agg_features)
    return results
