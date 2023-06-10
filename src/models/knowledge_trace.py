from typing import Any, Dict

import pytorch_lightning as ptl
import torch


class PSPEventEmbedding(torch.nn.Module):
    def __init__(
        self,
        event_name_nunique: int,
        event_name_embedding_dim: int,
        level_nunique: int,
        level_embedding_dim: int,
        fqid_nunique: int,
        fqid_embedding_dim: int,
        room_fqid_nunique: int,
        room_fqid_embedding_dim: int,
    ) -> None:
        super(PSPEventEmbedding, self).__init__()
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

        self.embedding_dim = (
            event_name_embedding_dim
            + level_embedding_dim
            + fqid_embedding_dim
            + room_fqid_embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        event_name = self.event_name_embedding(x["event_name"])
        level = self.level_embedding(x["level"])
        fqid = self.fqid_embedding(x["fqid"])
        room_fqid = self.room_fqid_embedding(x["room_fqid"])

        embeddings = torch.cat([event_name, level, fqid, room_fqid], dim=2)
        return embeddings


class PSPLightningModule(ptl.LightningModule):
    def __init__(
        self,
        continuous_dim: int,
        cat_embedding: torch.nn.Module,
        transformer_nhead: int = 4,
        transformer_num_encoder_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        output_dim: int = 1,
        learning_rate: float = 1e-2,
    ) -> None:
        super(PSPLightningModule, self).__init__()
        self.learning_rate = learning_rate

        self.cat_embedding = cat_embedding
        self.d_model = int(self.cat_embedding.embedding_dim + continuous_dim)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            nhead=transformer_nhead,
            d_model=self.d_model,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=transformer_num_encoder_layers
        )
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(self.d_model),
            torch.nn.Linear(self.d_model, output_dim),
        )

        assert self.d_model % 2 == 0, "Must be d_model is even."

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_cat = self.cat_embedding(x)  # (bs, max_seq_len, embedding_dim)
        x_continuous = x["elapsed_time_diff"].unsqueeze(
            dim=2
        )  # (bs, max_seq_len, 1)
        x = torch.cat([x_cat, x_continuous], dim=2)
        x = x.permute(1, 0, 2)  # (max_seq_len, bs, embedding_dim)
        x = self.transformer(x, mask=None)  # (max_seq_len, bs, embedding_dim)
        x = x.permute(1, 0, 2)  # (bs, max_seq_len, embedding_dim)
        x = x[:, -1, :]
        x = self.fc(x)  # (bs, max_seq_len, output_dim)
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
    bs = 16
    max_seq_len = 128

    x = {
        "elapsed_time_diff": torch.rand(bs, max_seq_len),
        "event_name": torch.randint(0, 10, (bs, max_seq_len)),
        "level": torch.randint(0, 10, (bs, max_seq_len)),
        "fqid": torch.randint(0, 10, (bs, max_seq_len)),
        "room_fqid": torch.randint(0, 10, (bs, max_seq_len)),
    }

    embedding = PSPEventEmbedding(
        event_name_nunique=10,
        event_name_embedding_dim=31,
        level_nunique=10,
        level_embedding_dim=4,
        fqid_nunique=10,
        fqid_embedding_dim=32,
        room_fqid_nunique=10,
        room_fqid_embedding_dim=16,
    )
    model = PSPLightningModule(
        continuous_dim=1,
        cat_embedding=embedding,
    )
    print(model(x).shape)


if __name__ == "__main__":
    main()
