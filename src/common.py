from typing import List, Union

import pandas as pd


def drop_null_columns(data: pd.DataFrame, columns: List) -> pd.DataFrame:
    return data.drop(columns, axis=1)


def clip_values(
    data: pd.DataFrame,
    columns: str,
    min_val: Union[int, float, None],
    max_val: Union[int, float, None],
) -> pd.DataFrame:
    data = data.assign(**{columns: lambda x: x[columns].clip(min_val, max_val)})
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


def split_room_fqid(data: pd.DataFrame) -> pd.DataFrame:
    data[[f"room_fqid_{i}" for i in range(3)]] = data["room_fqid"].str.split(
        ".", expand=True
    )
    data.drop(["room_fqid", "room_fqid_0"], axis=1, inplace=True)
    return data


def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_null_columns(
        data, ["page", "fullscreen", "hq", "music", "hover_duration"]
    )
    data = clip_values(data, "elapsed_time", 0, 1e7)
    data = cutoff_fqid(data)
    data = split_room_fqid(data)
    return data
