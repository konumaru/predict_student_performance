from typing import List, Union

import pandas as pd
import polars as pl


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


def cutoff_fqid(data: pd.DataFrame) -> pd.DataFrame:
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
    data["fqid"].where(data["fqid"].isin(cols_non_cutoff), "other", inplace=True)
    return data


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
        ["room_fqid", "room_fqid_0"]
    )
    return data


def preprocessing(data: pl.DataFrame) -> pl.DataFrame:
    data = drop_null_columns(data, ["page", "hover_duration"])
    data = clip_values(data, "elapsed_time", 0, 1e7)
    # data = cutoff_fqid(data)  # NOTE: Modelがいい感じに学習してくれるかもしれないので一旦やらない
    data = split_room_fqid(data)
    return data
