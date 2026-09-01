"""Train MAST with the data-access and optimization protocol from the paper."""

import math
import os
import pprint
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

from models.gdan import CVAE, Discriminator, Regressor
from models.semantic_transform import MultiAgentSemanticTransform
from utils.config_gdan import parser
from utils.logger import Logger, log_args
from utils.utils import (
    get_negative_samples,
    load_data,
    load_semantic_attributes,
    update_values,
)


args = parser.parse_args()
if args.config is not None:
    with open(args.config, "r", encoding="utf-8") as fin:
        update_values(yaml.load(fin, Loader=yaml.SafeLoader), vars(args))

source_dir = Path(args.source_data_root) / args.source_data_name
target_dir = Path(args.target_data_root) / args.target_data_name
source_att_path = source_dir / "att_splits.mat"
source_res_path = source_dir / "res101.mat"
target_att_path = target_dir / "att_splits.mat"

save_dir = Path(args.ckpt_dir)
save_dir.mkdir(parents=True, exist_ok=True)
Path(args.result).mkdir(parents=True, exist_ok=True)
result_path = save_dir / "gdan_loss.txt"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pprint.pprint(vars(args))
log_path = save_dir / "gdan_log.txt"
print("log file:", log_path)
logMaster = Logger(str(log_path))
log_args(log_path, args)


def sample_image_batch(data, batch_size):
    replace = len(data) < batch_size
    indices = np.random.choice(len(data), batch_size, replace=replace)
    features = np.asarray([data[index][0] for index in indices], dtype=np.float32)
    labels = np.asarray([data[index][1] for index in indices], dtype=np.int64)
    return features, labels


def sample_target_attributes(target_attributes, batch_size):
    """Sample target categories independently of source labels."""
    target_labels = np.random.choice(len(target_attributes), batch_size, replace=True)
    return target_attributes[target_labels], target_labels


def unpaired_mmd_loss(source_projection, target_projection):
    """RBF-MMD alignment; it makes no same-index positive-pair assumption."""
    feature_dim = source_projection.size(1)
    gamma = 1.0 / max(feature_dim, 1)
    source_dist = torch.cdist(source_projection, source_projection).pow(2)
    target_dist = torch.cdist(target_projection, target_projection).pow(2)
    cross_dist = torch.cdist(source_projection, target_projection).pow(2)
    return (
        torch.exp(-gamma * source_dist).mean()
        + torch.exp(-gamma * target_dist).mean()
        - 2.0 * torch.exp(-gamma * cross_dist).mean()
    )


def routing_entropy(weights):
    return -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1).mean()


def set_frozen(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def validate_source(cvae, regressor, semantic_transform, source_attributes, val_data):
    """Validation uses source-domain images only and drives early stopping."""
    cvae.eval()
    regressor.eval()
    semantic_transform.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for start in range(0, len(val_data), args.batch):
            batch = val_data[start:start + args.batch]
            if not batch:
                continue
            x = torch.from_numpy(
                np.asarray([item[0] for item in batch], dtype=np.float32)
            ).cuda()
            labels = np.asarray([item[1] for item in batch], dtype=np.int64)
            raw_semantics = torch.from_numpy(source_attributes[labels]).float().cuda()
            semantics, _, _, _ = semantic_transform(
                raw_semantics, domain="source"
            )
            mu, log_sigma = cvae.encode(x)
            reconstructed = cvae.decoder(torch.cat([semantics, mu], dim=1))
            predicted_semantics = regressor(x)
            loss = cvae.vae_loss(x, reconstructed, mu, log_sigma)
            loss = loss + args.theta2 * F.mse_loss(predicted_semantics, semantics)
            total_loss += loss.item() * len(batch)
            total_examples += len(batch)

    cvae.train()
    regressor.train()
    semantic_transform.train()
    if total_examples == 0:
        raise ValueError("source validation split is empty; it cannot control early stopping")
    return total_loss / total_examples


def checkpoint_state(
    epoch,
    cvae,
    discriminator,
    regressor,
    semantic_transform,
    optimizers,
    metrics,
    agent_weights,
    policy_values,
):
    return {
        "epoch": epoch,
        "cvae": cvae.state_dict(),
        "discriminator": discriminator.state_dict(),
        "regressor": regressor.state_dict(),
        "semantic_transform": semantic_transform.state_dict(),
        "semantic_transform_hidden_dims": semantic_transform.hidden_dims,
        "dropout_rate": args.dropout_rate,
        "enc_layers": args.enc,
        "dec_layers": args.dec,
        "reg_layers": args.reg,
        "dis_layers": args.dis,
        "z_dim": args.z_dim,
        "x_dim": args.x_dim,
        "source_s_dim": args.source_s_dim,
        "target_s_dim": args.target_s_dim,
        "semantic_space_dim": args.source_s_dim,
        "cvae_opt": optimizers["cvae"].state_dict(),
        "dis_opt": optimizers["discriminator"].state_dict(),
        "reg_opt": optimizers["regressor"].state_dict(),
        "semantic_transform_opt": optimizers["semantic_transform"].state_dict(),
        "theta1": args.theta1,
        "theta2": args.theta2,
        "theta3": args.theta3,
        "theta4": args.theta4,
        "theta5": args.theta5,
        "theta6": args.theta6,
        "theta7": args.theta7,
        "num_agents": args.num_agents,
        "agent_weights": agent_weights.detach().cpu(),
        "policy_values": [value.detach().cpu() for value in policy_values],
        "loss_values": metrics,
        "protocol": {
            "source_target_adapters": "independent",
            "category_sampling": "independent",
            "target_images_during_training": False,
            "early_stopping_split": "source_validation",
            "routing_objective": "maximize_entropy",
            "discriminator_fake_conditioning": "source_only",
            "target_generation_objective": "generator_only",
        },
    }


def main():
    logger = logMaster.get_logger("main")
    logger.info("loading source images and public source/target semantic attributes...")

    # Only the source res101.mat is opened during optimization. Target image
    # features are intentionally absent from this training process.
    (
        source_att_feats,
        source_train_data,
        source_val_data,
        _,
        _,
        source_classes,
    ) = load_data(att_path=source_att_path, res_path=source_res_path)
    target_att_feats = load_semantic_attributes(target_att_path)

    logger.info(
        "target image features remain hidden until a frozen checkpoint is evaluated"
    )
    logger.info("building model...")

    cvae = CVAE(
        x_dim=args.x_dim,
        s_dim=args.source_s_dim,
        z_dim=args.z_dim,
        enc_layers=args.enc,
        dec_layers=args.dec,
    )
    pretrained = torch.load(args.vae_ckpt, map_location="cpu")
    cvae.load_state_dict(pretrained["model"])
    cvae.cuda()

    # All semantic conditioning is in the source-sized shared output space.
    discriminator = Discriminator(
        x_dim=args.x_dim, s_dim=args.source_s_dim, layers=args.dis
    ).cuda()
    regressor = Regressor(
        x_dim=args.x_dim, s_dim=args.source_s_dim, layers=args.reg
    ).cuda()
    semantic_transform = MultiAgentSemanticTransform(
        target_dim=args.target_s_dim,
        source_dim=args.source_s_dim,
        num_agents=args.num_agents,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
    ).cuda()

    mse_loss = nn.MSELoss()
    adam_betas = (0.8, 0.999)
    optimizers = {
        "cvae": optim.Adam(
            cvae.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=adam_betas,
        ),
        "discriminator": optim.Adam(
            discriminator.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=adam_betas,
        ),
        "regressor": optim.Adam(
            regressor.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=adam_betas,
        ),
        "semantic_transform": optim.Adam(
            semantic_transform.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=adam_betas,
        ),
    }
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizers["semantic_transform"],
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_lr,
    )

    default_steps = int(math.ceil(len(source_train_data) / args.batch))
    steps = default_steps if args.steps == -1 else args.steps
    best_val_loss = float("inf")
    no_improve_count = 0
    loss_history = []
    agent_weights_history = []
    agent_performance_history = []
    actual_epochs = []
    logger.info("start training with source-domain images only...")

    for epoch in range(args.epoch):
        epoch_start = time.time()
        cvae.train()
        discriminator.train()
        regressor.train()
        semantic_transform.train()

        totals = {
            "d": 0.0,
            "g": 0.0,
            "cycle": 0.0,
            "regression": 0.0,
            "reg_adv": 0.0,
            "vae": 0.0,
            "mmd": 0.0,
            "entropy": 0.0,
            "total": 0.0,
        }
        epoch_weights = []
        epoch_performance = []

        for _ in tqdm(range(steps), leave=False, ncols=70, unit="b"):
            for _ in range(args.d_iter):
                optimizers["discriminator"].zero_grad(set_to_none=True)
                x_np, source_labels = sample_image_batch(source_train_data, args.batch)
                negative_labels = get_negative_samples(
                    source_labels.tolist(), source_classes["train"]
                )
                x = torch.from_numpy(x_np).float().cuda()
                source_raw = torch.from_numpy(source_att_feats[source_labels]).float().cuda()
                negative_raw = torch.from_numpy(
                    source_att_feats[negative_labels]
                ).float().cuda()
                ones = torch.ones((x.size(0), 1), device=x.device)
                zeros = torch.zeros((x.size(0), 1), device=x.device)

                with torch.no_grad():
                    source_sem, _, _, _ = semantic_transform(
                        source_raw, domain="source"
                    )
                    negative_sem, _, _, _ = semantic_transform(
                        negative_raw, domain="source"
                    )
                    fake_source = cvae.sample(source_sem)
                    predicted_sem = regressor(x)

                # The discriminator is trained only with source-conditioned
                # pairs, so semantic domain cannot serve as a real/fake cue.
                d_loss = (
                    mse_loss(discriminator(x, source_sem), ones)
                    + mse_loss(discriminator(fake_source, source_sem), zeros)
                    + args.theta3 * mse_loss(discriminator(x, predicted_sem), zeros)
                    + mse_loss(discriminator(x, negative_sem), zeros)
                )
                d_loss.backward()
                optimizers["discriminator"].step()
                totals["d"] += d_loss.item()

            for _ in range(args.g_iter):
                optimizers["cvae"].zero_grad(set_to_none=True)
                optimizers["regressor"].zero_grad(set_to_none=True)
                optimizers["semantic_transform"].zero_grad(set_to_none=True)

                x_np, source_labels = sample_image_batch(source_train_data, args.batch)
                target_np, _ = sample_target_attributes(
                    target_att_feats, len(source_labels)
                )
                x = torch.from_numpy(x_np).float().cuda()
                source_raw = torch.from_numpy(source_att_feats[source_labels]).float().cuda()
                target_raw = torch.from_numpy(target_np).float().cuda()
                ones = torch.ones((x.size(0), 1), device=x.device)

                (
                    source_sem,
                    source_weights,
                    source_policy,
                    source_agent_outputs,
                ) = semantic_transform(source_raw, domain="source")
                (
                    target_sem,
                    target_weights,
                    target_policy,
                    target_agent_outputs,
                ) = semantic_transform(target_raw, domain="target")
                reconstructed, mu, log_sigma = cvae(x, source_sem)
                generated_target = cvae.sample(target_sem)
                predicted_sem = regressor(x)
                reconstructed_sem = regressor(reconstructed)
                cycle_x, _, _ = cvae(x, predicted_sem)

                vae_loss = cvae.vae_loss(x, reconstructed, mu, log_sigma)
                cycle_loss = mse_loss(reconstructed_sem, source_sem) + mse_loss(
                    cycle_x, x
                )
                # Target-conditioned generation appears only in the generator
                # objective; it is never presented to the discriminator as fake.
                generator_loss = mse_loss(
                    discriminator(generated_target, target_sem), ones
                )
                regression_loss = mse_loss(predicted_sem, source_sem)
                reg_adv_loss = mse_loss(discriminator(x, predicted_sem), ones)

                source_projection = semantic_transform.get_projection(
                    source_raw, domain="source"
                )
                target_projection = semantic_transform.get_projection(
                    target_raw, domain="target"
                )
                mmd_loss = unpaired_mmd_loss(
                    source_projection, target_projection
                )
                mean_alignment_loss = mse_loss(
                    source_projection.mean(dim=0),
                    target_projection.mean(dim=0),
                )

                # [num_agents, agent_output_dim]
                source_agent_means = source_agent_outputs.mean(dim=0)
                target_agent_means = target_agent_outputs.mean(dim=0)
                semantic_consistency_loss = F.mse_loss(
                    source_agent_means,
                    target_agent_means,
                    reduction="mean",
                )

                all_policy = source_policy + target_policy
                policy_target = torch.full_like(all_policy[0], 0.5)
                agent_loss = sum(
                    F.mse_loss(value, policy_target) for value in all_policy
                ) / len(all_policy)

                entropy = 0.5 * (
                    routing_entropy(source_weights)
                    + routing_entropy(target_weights)
                )
                # Minimization of -H explicitly maximizes routing entropy.
                high_entropy_regularizer = -entropy
                alignment_scale = min(1.0, max(0.0, (epoch - 50) / 50.0))
                base_loss = (
                    vae_loss
                    + generator_loss
                    + args.theta1 * cycle_loss
                    + args.theta2 * regression_loss
                    + args.theta3 * reg_adv_loss
                    + args.agent_weight * agent_loss
                )
                alignment_loss = (
                    args.theta4 * mmd_loss
                    + args.theta5 * mean_alignment_loss
                    + args.theta6 * semantic_consistency_loss
                    + (args.theta7 + args.entropy_weight)
                    * high_entropy_regularizer
                )
                total_loss = base_loss + alignment_scale * alignment_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    semantic_transform.parameters(), args.gradient_clip
                )
                optimizers["semantic_transform"].step()
                optimizers["cvae"].step()
                optimizers["regressor"].step()

                totals["g"] += generator_loss.item()
                totals["cycle"] += cycle_loss.item()
                totals["regression"] += regression_loss.item()
                totals["reg_adv"] += reg_adv_loss.item()
                totals["vae"] += vae_loss.item()
                totals["mmd"] += mmd_loss.item()
                totals["entropy"] += entropy.item()
                totals["total"] += total_loss.item()

                epoch_weights.append(target_weights.detach().mean(dim=0).cpu().numpy())
                performances = []
                for agent in semantic_transform.agents:
                    agent_output, _ = agent(target_raw, domain="target")
                    agent_output = semantic_transform.dim_transform(agent_output)
                    performances.append(
                        F.mse_loss(agent_output, target_sem.detach()).item()
                    )
                epoch_performance.append(np.asarray(performances))

        val_loss = validate_source(
            cvae,
            regressor,
            semantic_transform,
            source_att_feats,
            source_val_data,
        )
        scheduler.step(val_loss)
        generator_steps = max(steps * args.g_iter, 1)
        discriminator_steps = max(steps * args.d_iter, 1)
        metrics = {
            key: value / (discriminator_steps if key == "d" else generator_steps)
            for key, value in totals.items()
        }
        metrics["source_val_loss"] = val_loss
        metrics["routing_objective"] = "maximize_entropy"

        avg_weights = np.mean(epoch_weights, axis=0)
        avg_performance = np.mean(epoch_performance, axis=0)
        agent_weights_history.append(avg_weights)
        agent_performance_history.append(avg_performance)
        actual_epochs.append(epoch + 1)

        state = checkpoint_state(
            epoch + 1,
            cvae,
            discriminator,
            regressor,
            semantic_transform,
            optimizers,
            metrics,
            target_weights,
            target_policy,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(state, str(save_dir / "best_gdan.pkl"))
        else:
            no_improve_count += 1

        if (epoch + 1) % args.save_epoch == 0 or epoch == 0:
            torch.save(state, str(save_dir / f"gdan_{epoch + 1}.pkl"))

        elapsed = (time.time() - epoch_start) / 60.0
        logger.info(
            "epoch: %4d, source_val: %.5f, total: %.4f, entropy: %.4f, "
            "mmd: %.4f, minutes: %.2f",
            epoch + 1,
            val_loss,
            metrics["total"],
            metrics["entropy"],
            metrics["mmd"],
            elapsed,
        )
        loss_history.append(
            f"{epoch + 1}\t{metrics['total']:.6f}\t{val_loss:.6f}\t"
            f"{metrics['entropy']:.6f}\t{metrics['mmd']:.6f}\n"
        )

        if no_improve_count >= args.early_stopping_patience:
            logger.info(
                "early stopping at epoch %d based on source validation loss",
                epoch + 1,
            )
            break

    # The optimization phase is over before any target image feature is exposed.
    for module in (cvae, discriminator, regressor, semantic_transform):
        set_frozen(module)

    torch.save(
        {
            "agent_weights_history": agent_weights_history,
            "agent_performance_history": agent_performance_history,
            "epochs": actual_epochs,
        },
        str(save_dir / "agent_analysis.pt"),
    )
    with result_path.open("w", encoding="utf-8") as fout:
        fout.writelines(loss_history)
    logger.info("training finished; frozen checkpoints are ready for target evaluation")


if __name__ == "__main__":
    main()
