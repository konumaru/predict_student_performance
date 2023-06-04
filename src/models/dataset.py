from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class PSPDataset(Dataset):
    def __init__(self, data: pd.DataFrame, is_test: bool = False) -> None:
        self.elapsed_time_diff = torch.tensor(
            data["elapsed_time_diff"].tolist(), dtype=torch.float32
        )
        self.event_name = torch.tensor(
            data["event_name"].tolist(), dtype=torch.long
        )
        self.level = torch.tensor(data["level"].tolist(), dtype=torch.long)
        self.fqid = torch.tensor(data["fqid"].tolist(), dtype=torch.long)
        self.room_fqid = torch.tensor(
            data["room_fqid"].tolist(), dtype=torch.long
        )

        if is_test:
            self.labels = torch.empty(len(data), 2)
        else:
            self.labels = torch.tensor(
                data["correct"].tolist(), dtype=torch.float32
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return (
            self.labels[idx],
            {
                "elapsed_time_diff": self.elapsed_time_diff[idx],
                "event_name": self.event_name[idx],
                "level": self.level[idx],
                "fqid": self.fqid[idx],
                "room_fqid": self.room_fqid[idx],
            },
        )


def get_dataloader(
    data: pd.DataFrame,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = False,
    is_test: bool = False,
) -> DataLoader:
    dataset = PSPDataset(data, is_test)
    return DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
