"""Evaluate frozen checkpoints on target-domain image features."""

import argparse
import os
import pprint
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from models.gdan import CVAE, Discriminator, Regressor
from models.semantic_transform import MultiAgentSemanticTransform
from utils.logger import Logger, log_args
from utils.utils import get_datetime_str, load_data, update_values


parser = argparse.ArgumentParser(description="frozen target-domain evaluation")
parser.add_argument("-cfg", "--config", metavar="YAML", default=None)
parser.add_argument("-dn", "--data_name", default="CUB")
parser.add_argument("-r", "--result", default="./result")
parser.add_argument("-f", "--logfile", default=None)
parser.add_argument("-ckpt", "--ckpt_dir", default="./checkpoints/")
parser.add_argument("-clf", "--classifier", default="knn", choices=["knn", "svc"])
parser.add_argument("-ns", "--num_samples", type=int, default=500)
parser.add_argument("-k", "--K", type=int, default=1)
parser.add_argument("-c", "--C", type=float, default=1.0)
parser.add_argument("-g", "--gpu", default="0")
parser.add_argument("--source_data_name", default="APY")
parser.add_argument("--source_data_root", default="./data/xlsa17/data")
parser.add_argument("--target_data_name", default="CUB")
parser.add_argument("--target_data_root", default="./data/xlsa17/data")

args = parser.parse_args()
if args.config is not None:
    with open(args.config, "r", encoding="utf-8") as fin:
        update_values(yaml.load(fin, Loader=yaml.SafeLoader), vars(args))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
ts = get_datetime_str()
if args.logfile is None:
    args.logfile = f"log_valtest_{args.data_name}_{ts}.txt"

target_dir = Path(args.target_data_root) / args.target_data_name
target_att_path = target_dir / "att_splits.mat"
target_res_path = target_dir / "res101.mat"
result_dir = Path(args.result)
result_dir.mkdir(parents=True, exist_ok=True)
logfile = result_dir / args.logfile
logMaster = Logger(str(logfile))
log_args(str(logfile), args)
pprint.pprint(vars(args))


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_frozen_models(checkpoint, logger):
    """Load and freeze every trainable component before target images are read."""
    states = torch.load(checkpoint, map_location="cpu")
    semantic_dim = states.get("semantic_space_dim", states["source_s_dim"])
    net = CVAE(
        x_dim=states["x_dim"],
        s_dim=states["source_s_dim"],
        z_dim=states["z_dim"],
        enc_layers=states["enc_layers"],
        dec_layers=states["dec_layers"],
    )
    discriminator = Discriminator(
        x_dim=states["x_dim"],
        s_dim=semantic_dim,
        layers=states["dis_layers"],
    )
    regressor = Regressor(
        x_dim=states["x_dim"],
        s_dim=semantic_dim,
        layers=states["reg_layers"],
    )
    semantic_transform = MultiAgentSemanticTransform(
        target_dim=states["target_s_dim"],
        source_dim=states["source_s_dim"],
        num_agents=states.get("num_agents", 3),
        hidden_dims=states.get("semantic_transform_hidden_dims", [512, 256]),
        dropout_rate=states.get("dropout_rate", 0.4),
    )

    net.load_state_dict(states["cvae"])
    discriminator.load_state_dict(states["discriminator"])
    regressor.load_state_dict(states["regressor"])
    semantic_transform.load_state_dict(states["semantic_transform"])
    modules = tuple(
        freeze(module.cuda())
        for module in (net, discriminator, regressor, semantic_transform)
    )
    logger.info(
        "loaded and froze checkpoint %s before exposing target image features",
        checkpoint,
    )
    return states, modules


def build_classifier(train_data):
    x, y = zip(*train_data)
    x = np.asarray(x)
    y = np.asarray(y)
    if args.classifier == "svc":
        classifier = LinearSVC(C=args.C)
    else:
        classifier = KNeighborsClassifier(n_neighbors=args.K)
    classifier.fit(x, y)
    return classifier


def evaluate_split(net, semantic_transform, attributes, train_data, eval_data, labels):
    samples = generate_samples(
        net,
        semantic_transform,
        args.num_samples,
        attributes[labels],
        labels,
    )
    classifier = build_classifier(train_data + samples)
    test_x, test_y = zip(*eval_data)
    predictions = classifier.predict(test_x)
    return cal_macc(truth=test_y, pred=predictions)


def main():
    logger = logMaster.get_logger("main")
    ckpt_dir = Path(args.ckpt_dir)
    best_checkpoint = ckpt_dir / "best_gdan.pkl"
    if best_checkpoint.exists():
        checkpoints = [best_checkpoint]
    else:
        checkpoints = sorted(
            ckpt_dir.glob("gdan_[0-9]*.pkl"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
    if not checkpoints:
        raise FileNotFoundError(f"no GDAN checkpoints found in {ckpt_dir}")

    val_acc = []
    test_acc = []
    model_epochs = []
    for checkpoint in checkpoints:
        states, modules = load_frozen_models(checkpoint, logger)
        net, _, _, semantic_transform = modules

        # This is the first access to target train/seen/unseen image features,
        # and it occurs only after the checkpoint above is immutable and frozen.
        logger.info("loading target images for post-freeze evaluation...")
        attributes, train_data, val_data, unseen_data, _, classes = load_data(
            att_path=target_att_path,
            res_path=target_res_path,
        )
        with torch.no_grad():
            val_score = evaluate_split(
                net,
                semantic_transform,
                attributes,
                train_data,
                val_data,
                classes["val"],
            )
            test_score = evaluate_split(
                net,
                semantic_transform,
                attributes,
                train_data,
                unseen_data,
                classes["test"],
            )
        epoch = states["epoch"]
        model_epochs.append(epoch)
        val_acc.append(val_score)
        test_acc.append(test_score)
        logger.info(
            "frozen model at epoch %d: target val_acc %.5f, test_acc %.5f",
            epoch,
            val_score,
            test_score,
        )
        del modules

    results = {
        "model_epochs": model_epochs,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "checkpoint_selection": "source_validation",
        "target_images_exposed": "after_model_freeze",
    }
    save_path = result_dir / "full_evaluation_results.pt"
    torch.save(results, str(save_path))
    logger.info("evaluation results saved to %s", save_path)


def generate_samples(net, semantic_transform, num_samples, class_emb, labels):
    data = []
    for embedding, label in zip(class_emb, labels):
        repeated = np.repeat(embedding.reshape(1, -1), num_samples, axis=0)
        semantics = torch.from_numpy(repeated).float().cuda()
        transformed, _, _, _ = semantic_transform(semantics, domain="target")
        samples = net.sample(transformed).cpu().numpy()
        data.extend((sample, label) for sample in samples)
    return data


def cal_macc(*, truth, pred):
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if len(truth) != len(pred):
        raise ValueError("truth and pred must have equal length")
    class_scores = []
    for label in np.unique(truth):
        mask = truth == label
        class_scores.append(np.mean(pred[mask] == truth[mask]))
    return float(np.mean(class_scores))


if __name__ == "__main__":
    main()
