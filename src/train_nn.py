import pathlib
import warnings
from collections import defaultdict

import hydra
import numpy as np
import polars as pl
import pytorch_lightning as ptl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from common import create_sequence_features
from metric import optimize_f1_score
from models.dataset import get_dataloader
from models.knowledge_trace import PSPEventEmbedding, PSPLightningModule
from utils import seed_everything, timer
from utils.io import load_pickle, save_pickle, save_txt

warnings.simplefilter("ignore")
torch.set_float32_matmul_precision("high")


def train(
    cfg: DictConfig, feature_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    max_seq_len = cfg.model.max_seq_len
    batch_size = cfg.model.batch_size

    X = pl.read_parquet("./data/preprocessing/train.parquet")
    uniques_map = load_pickle("./data/preprocessing/uniques_map.pkl")
    for fold in range(cfg.n_splits):
        print(f">>>> Train fold={fold}")
        y_train = pl.read_parquet(feature_dir / f"y_train_fold_{fold}.parquet")
        y_valid = pl.read_parquet(feature_dir / f"y_valid_fold_{fold}.parquet")

        suffix = f"{cfg.model.name}_fold_{fold}"
        pred_all_level_groups = []
        label_all_level_groups = []
        output_dims = {"0-4": 3, "5-12": 10, "13-22": 5}
        for level_group in ["0-4", "5-12", "13-22"]:
            _train = X.filter(X["level_group"] == level_group).to_pandas()
            _y_train = y_train.filter(
                y_train["level_group"] == level_group
            ).to_pandas()
            _y_valid = y_valid.filter(
                y_valid["level_group"] == level_group
            ).to_pandas()

            _train = _train.assign(
                elapsed_time_diff=_train["elapsed_time"]
                .fillna(0)
                .clip(0.0, 3.6e6),
            )

            _uniques_map = uniques_map[level_group]
            cols_to_encode = ["event_name", "level", "fqid", "room_fqid"]
            for col in cols_to_encode:
                _train[col] = (
                    _train[col]
                    .fillna(f"{col}_null")
                    .map(
                        {
                            k: int(i + 1)
                            for i, k in enumerate(
                                _uniques_map[col] + [f"{col}_null"]
                            )
                        }
                    )
                )

            train_data = create_sequence_features(
                _train, _y_train, max_seq_len
            )
            valid_data = create_sequence_features(
                _train, _y_valid, max_seq_len
            )

            train_dataloader = get_dataloader(
                train_data, batch_size=batch_size, shuffle=True, drop_last=True
            )
            vlaid_dataloader = get_dataloader(
                valid_data,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            logger = CSVLogger(save_dir="./data/train/logs", name="psp")
            embedding = PSPEventEmbedding(
                event_name_nunique=len(_uniques_map["event_name"]),
                event_name_embedding_dim=cfg.model.event_name_embedding_dim,
                level_nunique=len(_uniques_map["level"]),
                level_embedding_dim=cfg.model.level_embedding_dim,
                fqid_nunique=len(_uniques_map["fqid"]),
                fqid_embedding_dim=cfg.model.fqid_embedding_dim,
                room_fqid_nunique=len(_uniques_map["room_fqid"]),
                room_fqid_embedding_dim=cfg.model.room_fqid_embedding_dim,
            ).to("cuda")
            model = PSPLightningModule(
                continuous_dim=1,
                cat_embedding=embedding,
                output_dim=output_dims[level_group],
            )
            checkpoint_callback = ModelCheckpoint(
                "./data/models/ckpt",
                monitor="valid_loss",
                save_top_k=1,
                mode="min",
            )
            checkpoint_ealry_stopping = EarlyStopping(
                monitor="valid_loss", patience=3, mode="min"
            )
            trainer = ptl.Trainer(
                max_epochs=cfg.model.max_epochs,
                accelerator="gpu",
                check_val_every_n_epoch=cfg.model.check_val_every_n_epoch,
                log_every_n_steps=cfg.model.log_every_n_steps,
                logger=logger,
                callbacks=[checkpoint_callback, checkpoint_ealry_stopping],
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            trainer.fit(model, train_dataloader, vlaid_dataloader)

            _pred = trainer.predict(model, vlaid_dataloader)
            label = vlaid_dataloader.dataset.labels.numpy()  # type: ignore

            pred = torch.cat(_pred, dim=0).sigmoid().numpy()  # type: ignore
            pred_all_level_groups.append(pred)
            label_all_level_groups.append(label)

        pred_all = np.concatenate(pred_all_level_groups, axis=1)
        label_all = np.concatenate(label_all_level_groups, axis=1)
        save_pickle(output_dir / f"y_pred_{suffix}.pkl", pred_all)
        save_pickle(output_dir / f"y_true_{suffix}.pkl", label_all)


def evaluate(
    cfg: DictConfig, feature_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    oofs = []
    labels = []
    for fold in range(cfg.n_splits):
        oofs.append(
            load_pickle(
                output_dir / f"y_pred_{cfg.model.name}_fold_{fold}.pkl"
            )
        )
        labels.append(
            load_pickle(
                output_dir / f"y_true_{cfg.model.name}_fold_{fold}.pkl"
            )
        )

    oof = np.concatenate(oofs, axis=0)
    label = np.concatenate(labels, axis=0)

    threshold_levels = defaultdict(float)
    for level in range(label.shape[1]):
        score, threshold = optimize_f1_score(label[:, level], oof[:, level])
        threshold_levels[level] = threshold

        print(
            f"level-{level + 1}: F1score={score:.4f} threshold={threshold:.4f}"
        )

    save_pickle(
        output_dir / f"threshold_{cfg.model.name}.pkl", threshold_levels
    )
    score, threshold = optimize_f1_score(label.ravel(), oof.ravel())
    save_txt(output_dir / f"score_{cfg.model.name}.txt", str(score))
    print(f"overall: F1score={score:.4f} threshold={threshold:.4f}")


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    output_dir = pathlib.Path("./data/train")
    feature_dir = pathlib.Path("./data/feature")

    print("\n##### Train Model #####\n")
    train(cfg, feature_dir, output_dir)

    print("\n##### Evaluate #####\n")
    evaluate(cfg, feature_dir, output_dir)


if __name__ == "__main__":
    with timer("Train"):
        main()
