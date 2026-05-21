#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from ast import arg
import enum
from math import e
import sys
import os
import random
from datetime import datetime
import torch
import pandas as pd
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from lib.edge_coupling import sort_edge_index_with_values, sort_sparse_entries, validate_edge_index
from utils.heterophilic import get_dataset, get_fixed_splits


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])
    validate_edge_index(model.edge_index, num_nodes=args['graph_size'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            wandb.log(res_dict, step=epoch)

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    if "ODE" not in args['model']:
        # Debugging for discrete models
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args['layers']):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    last_maps = getattr(model, '_last_maps', None)
    last_laplacian = getattr(model, '_last_laplacian', None)
    last_node_representations = getattr(model, '_last_node_representations', None)
    if last_maps and last_laplacian:
        
        if args["dataset"] == "synthetic_exp":
            maps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'maps', f'{args["dataset"]}', f"normalised-{str(args['normalised']).lower()}", f'stalk_dim-{args["d"]}',f'{args["layers"]}-layers', f'{args["hidden_channels"]}-hidden', f'{args["epochs"]}-epochs'))
            lap_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'laplacians', f'{args["dataset"]}', f"normalised-{str(args['normalised']).lower()}", f'stalk_dim-{args["d"]}',f'{args["layers"]}-layers', f'{args["hidden_channels"]}-hidden', f'{args["epochs"]}-epochs'))
            repr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'representations', f'{args["dataset"]}', f"normalised-{str(args['normalised']).lower()}", f'stalk_dim-{args["d"]}',f'{args["layers"]}-layers', f'{args["hidden_channels"]}-hidden', f'{args["epochs"]}-epochs'))
        else:        
            maps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'maps', f'{args["dataset"]}', f"normalised-{str(args['normalised']).lower()}", f'stalk_dim-{args["d"]}',f'{args["layers"]}-layers', f'{args["hidden_channels"]}-hidden', f'{args["epochs"]}-epochs'))
            lap_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'laplacians', f'{args["dataset"]}', f"normalised-{str(args['normalised']).lower()}", f'stalk_dim-{args["d"]}',f'{args["layers"]}-layers', f'{args["hidden_channels"]}-hidden', f'{args["epochs"]}-epochs'))
            repr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'representations', f'{args["dataset"]}', f"normalised-{str(args['normalised']).lower()}", f'stalk_dim-{args["d"]}',f'{args["layers"]}-layers', f'{args["hidden_channels"]}-hidden', f'{args["epochs"]}-epochs'))
        
        os.makedirs(maps_dir, exist_ok=True)
        os.makedirs(lap_dir, exist_ok=True)
        os.makedirs(repr_dir, exist_ok=True)

        for layer, lap in last_laplacian.items():
            lap_indices = lap[0].detach().cpu()
            lap_values = lap[1].detach().cpu()

            if lap_indices.dim() != 2 or lap_indices.size(0) != 2:
                raise ValueError(f"Expected Laplacian indices of shape [2, N], got {tuple(lap_indices.shape)}")

            lap_indices, lap_values = sort_sparse_entries(lap_indices, lap_values)
            lap_matrix = torch.cat([lap_indices, lap_values.unsqueeze(0)], dim=0)

            if args["dataset"] == "synthetic_exp":
                lap_filename = f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}_layer{layer}_pct-hetero-{int(float(args['het_coef'])*100)}_classes-{args['num_classes']}_feats-{args['num_feats']}_seed{args['seed']}.pt"
            elif args["dataset"] == "bottleneck_exp":
                lap_filename = (
                    f"{args['model']}_graph-{args['bottleneck_graph']}"
                    f"_nodes-{args['num_nodes']}"
                    f"_left-{args['bottleneck_left']}"
                    f"_right-{args['bottleneck_right']}"
                    f"_bridge-width-{args['bridge_width']}"
                    f"_bridge-length-{args['bridge_length']}"
                    f"_sbm-intra-{args['sbm_intra_prob']}"
                    f"_sbm-inter-{args['sbm_inter_prob']}"
                    f"_features-{args['bottleneck_feature_mode']}"
                    f"_layer{layer}"
                    f"_classes-{args['num_classes']}"
                    f"_feats-{args['num_feats']}"
                    f"_fold{fold}"
                    f"_seed{args['seed']}.pt"
                )
            else:
                lap_filename = f"{args['model']}_{args['dataset']}_layer{layer}_fold{fold}_seed{args['seed']}.pt"
            
            lap_path = os.path.join(lap_dir, lap_filename)
            torch.save(lap_matrix, lap_path)
            print(f"Saved Laplacian to {lap_path} with shape {tuple(lap_matrix.shape)}")

        for layer, maps in last_maps.items():
            
            maps = maps.detach().cpu()
            map_edge_index = model.edge_index.detach().cpu()
        
            if maps.dim() == 0:
                maps = maps.unsqueeze(0)
            if maps.dim() == 1:
                maps_cols = maps.unsqueeze(1)
            else:
                maps_cols = maps.reshape(maps.shape[0], -1)

            if map_edge_index.size(1) != maps_cols.size(0):
                raise ValueError(
                    f"Expected one learned map per directed edge, got {maps_cols.size(0)} maps "
                    f"for {map_edge_index.size(1)} edges."
                )

            map_edge_index, maps_cols = sort_edge_index_with_values(map_edge_index, maps_cols)
            edge_cols = map_edge_index.t().to(maps_cols.dtype)
            maps_matrix = torch.cat([edge_cols, maps_cols], dim=1)

            if args["dataset"] == "synthetic_exp":
                maps_filename = f"{args['model']}_nodes-{args['num_nodes']}_node-deg-{args['node_degree']}_layer{layer}_pct-hetero-{int(float(args['het_coef'])*100)}_classes-{args['num_classes']}_feats-{args['num_feats']}_seed{args['seed']}.pt"
            elif args["dataset"] == "bottleneck_exp":
                maps_filename = (
                    f"{args['model']}_graph-{args['bottleneck_graph']}"
                    f"_nodes-{args['num_nodes']}"
                    f"_left-{args['bottleneck_left']}"
                    f"_right-{args['bottleneck_right']}"
                    f"_bridge-width-{args['bridge_width']}"
                    f"_bridge-length-{args['bridge_length']}"
                    f"_sbm-intra-{args['sbm_intra_prob']}"
                    f"_sbm-inter-{args['sbm_inter_prob']}"
                    f"_features-{args['bottleneck_feature_mode']}"
                    f"_layer{layer}"
                    f"_classes-{args['num_classes']}"
                    f"_feats-{args['num_feats']}"
                    f"_fold{fold}"
                    f"_seed{args['seed']}.pt"
                )
            else:
                maps_filename = f"{args['model']}_{args['dataset']}_layer{layer}_fold{fold}_seed{args['seed']}.pt"

            maps_path = os.path.join(maps_dir, maps_filename)
            torch.save(maps_matrix, maps_path)

        if last_node_representations:
            representation_payload = {
                "representations": {
                    name: value.detach().cpu() if torch.is_tensor(value) else value
                    for name, value in last_node_representations.items()
                },
                "metadata": {
                    "dataset": args["dataset"],
                    "model": args["model"],
                    "normalised": bool(args["normalised"]),
                    "stalk_dim": int(args["d"]),
                    "layers": int(args["layers"]),
                    "hidden_channels": int(args["hidden_channels"]),
                    "epochs": int(args["epochs"]),
                    "fold": int(fold),
                    "seed": int(args["seed"]),
                    "best_epoch": int(best_epoch),
                    "saved_after_epoch": int(epoch),
                },
            }

            if args["dataset"] == "bottleneck_exp":
                repr_filename = (
                    f"{args['model']}_graph-{args['bottleneck_graph']}"
                    f"_nodes-{args['num_nodes']}"
                    f"_left-{args['bottleneck_left']}"
                    f"_right-{args['bottleneck_right']}"
                    f"_bridge-width-{args['bridge_width']}"
                    f"_bridge-length-{args['bridge_length']}"
                    f"_sbm-intra-{args['sbm_intra_prob']}"
                    f"_sbm-inter-{args['sbm_inter_prob']}"
                    f"_features-{args['bottleneck_feature_mode']}"
                    f"_classes-{args['num_classes']}"
                    f"_feats-{args['num_feats']}"
                    f"_fold{fold}"
                    f"_seed{args['seed']}.pt"
                )
            else:
                repr_filename = f"{args['model']}_{args['dataset']}_fold{fold}_seed{args['seed']}.pt"

            repr_path = os.path.join(repr_dir, repr_filename)
            torch.save(representation_payload, repr_path)
            print(f"Saved node representations to {repr_path}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args.model == 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = get_dataset(args.dataset,args)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)

    # ADAPTING FROM FERRAN'S CODE
    #args.input_dim = dataset.num_features
    #args.output_dim = dataset.num_classes
    args.input_dim = dataset[0].x.shape[1]          # ← already fixed (same as my suggestion)
    try:
        args.output_dim = dataset.num_classes        # ← tries the InMemoryDataset property first
    except: 
        args.output_dim = torch.unique(dataset[0].y).shape[0]  # ← fallback for plain lists

    # I want to include the MPS Apple acceletor
    args.device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else 'cpu'
    )

    # I AM COMMENTING THIS TO TRY UNNORMALISATION
    # assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    wandb.init(project="sheaf", config=vars(args), entity=args.entity)

    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    # if hasattr(model_cls, '_last_maps') and model_cls._last_maps is not None:
    #     maps_dir = r"results/maps"
    #     os.makedirs(maps_dir, exist_ok=True)

    #     maps_filename = f"{args['model']}_{args['dataset']}_fold{fold}_seed{args['seed']}.pt"
    #     maps_path = os.path.abspath(os.path.join(maps_dir, maps_filename))
    #     torch.save(model_cls._last_maps.detach().cpu(), maps_path)
    #     print(f"Saved last restriction maps to {maps_path}")

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
