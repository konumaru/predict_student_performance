from collections import defaultdict
from typing import Any, Dict, List

import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from torchmetrics.classification import F1Score


class ConvEncoder(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 64, output_dim: int = 128
    ) -> None:
        super(ConvEncoder, self).__init__()

        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    input_dim, hidden_dim, kernel_size=16, dilation=1
                ),
                torch.nn.Conv1d(
                    hidden_dim // 2, hidden_dim, kernel_size=16, dilation=2
                ),
            ]
        )
        self.dropout = torch.nn.Dropout(0.2)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),  # mean, std
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch, sequence, channel)

        Returns:
            torch.Tensor: _description_
        """
        x = x.transpose(1, 2)  # (batch, channel, sequence)

        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_filter, x_gate = x_conv.chunk(2, dim=1)
            x_conv = F.tanh(x_filter) * F.sigmoid(x_gate)
            x_convs.append(x_conv)
            x = x_conv
        x = torch.cat(x_convs, dim=2)
        x_std = torch.std(x, dim=2)
        x_avg = torch.mean(x, dim=2)
        x = torch.cat([x_avg, x_std], dim=1)
        x = self.fc(x)
        return x


class PSPLightningModule(ptl.LightningModule):
    def __init__(
        self,
        event_name_nunique: int,
        level_nunique: int,
        fqid_nunique: int,
        room_fqid_nunique: int,
        continuous_dim: int = 1,
        event_name_embedding_dim: int = 8,
        level_embedding_dim: int = 8,
        fqid_embedding_dim: int = 8,
        room_fqid_embedding_dim: int = 8,
        conv_hidden_dim: int = 128,
        conv_output_dim: int = 128,
        output_dim: int = 3,
        learning_rate: float = 1e-2,
    ) -> None:
        super(PSPLightningModule, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.event_name_embedding = torch.nn.Embedding(
            event_name_nunique, event_name_embedding_dim, padding_idx=0
        )
        self.level_embedding = torch.nn.Embedding(
            level_nunique, level_embedding_dim, padding_idx=0
        )
        self.fqid_embedding = torch.nn.Embedding(
            fqid_nunique, fqid_embedding_dim, padding_idx=0
        )
        self.room_fqid_embedding = torch.nn.Embedding(
            room_fqid_nunique, room_fqid_embedding_dim, padding_idx=0
        )

        embedding_dim = (
            continuous_dim
            + event_name_embedding_dim
            + level_embedding_dim
            + fqid_embedding_dim
            + room_fqid_embedding_dim
        )
        self.conv_encoder = ConvEncoder(
            input_dim=embedding_dim,
            hidden_dim=conv_hidden_dim,
            output_dim=conv_output_dim,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_output_dim, conv_output_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(conv_output_dim // 2, output_dim),
        )

        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.metrics = F1Score(task="binary", threshold=0.63, average="macro")
        self.train_step_outputs: Dict[str, List[torch.Tensor]] = defaultdict(
            list
        )
        self.validation_step_outputs: Dict[
            str, List[torch.Tensor]
        ] = defaultdict(list)
        self.predict_step_outputs: List[torch.Tensor] = []

    def forward(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        continuous = (
            feat["elapsed_time_diff"].unsqueeze(1).transpose(1, 2)
        )  # (B, L, 1)
        event_name = self.event_name_embedding(feat["event_name"])
        level = self.level_embedding(feat["level"])
        fqid = self.fqid_embedding(feat["fqid"])
        room_fqid = self.room_fqid_embedding(feat["room_fqid"])

        x_cat = torch.cat(
            [event_name, level, fqid, room_fqid], dim=2
        )  # (B, L, E)
        x = torch.cat([continuous, x_cat], dim=2)
        x = self.conv_encoder(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        return optimizer

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        label, feat = batch
        pred = self(feat)
        loss = self.criteria(pred, label)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.train_step_outputs["pred"].append(pred)
        self.train_step_outputs["label"].append(label)
        return loss

    def on_train_epoch_end(self) -> None:
        pred = torch.cat(self.train_step_outputs["pred"], dim=0)
        label = torch.cat(self.train_step_outputs["label"], dim=0)
        loss = self.criteria(pred, label)
        self.log("train_loss", loss, on_step=False, prog_bar=True, logger=True)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        label, feat = batch
        pred = self(feat)
        loss = self.criteria(pred, label)
        self.validation_step_outputs["pred"].append(pred)
        self.validation_step_outputs["label"].append(label)
        return loss

    def on_validation_epoch_end(self) -> None:
        pred = torch.cat(self.validation_step_outputs["pred"], dim=0)
        label = torch.cat(self.validation_step_outputs["label"], dim=0)
        loss = self.criteria(pred, label)
        self.log("valid_loss", loss, on_step=False, prog_bar=True, logger=True)
        score = self.metrics(pred, label)
        self.log(
            "valid_score", score, on_step=False, prog_bar=True, logger=True
        )

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        _, feat = batch
        pred = self(feat)
        self.predict_step_outputs.append(pred)
        return pred


def main():
    print("Hello world!")

    bs = 4
    max_seq_len = 1024
    embedding_dim = 64
    feat = torch.rand(bs, max_seq_len, embedding_dim)
    model = ConvEncoder(input_dim=embedding_dim, hidden_dim=64, output_dim=128)

    out = model(feat)
    print(out)
    print(out.shape)


if __name__ == "__main__":
    main()
