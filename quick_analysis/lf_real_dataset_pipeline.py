from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.decomposition import PCA

from definitions import ROOT_DIR
from lib.edge_coupling import laplacian_matrix_to_edge_weights, undirected_edge_set
from utils.heterophilic import get_dataset


REAL_DATASET_CLASS_NAMES = {
    "texas": ["student", "project", "course", "staff", "faculty"],
    "wisconsin": ["student", "project", "course", "staff", "faculty"],
    "cornell": ["student", "project", "course", "staff", "faculty"],
}


@dataclass
class RealDatasetExperimentConfig:
    dataset: str
    layers: int = 2
    epochs: int = 500
    folds: int = 10
    seed: int = 43
    d: int = 1
    hidden_channels: int = 16
    model: str = "DiagSheaf"
    normalised: bool = True
    deg_normalised: bool = False
    lr: float = 0.02
    weight_decay: float = 5e-3
    sheaf_decay: float | None = None
    input_dropout: float = 0.0
    dropout: float = 0.7
    early_stopping: int = 200
    left_weights: bool = True
    right_weights: bool = True
    use_act: bool = True # default True, try using False
    add_hp: bool = False
    add_lp: bool = False
    edge_weights: bool = True
    sparse_learner: bool = False
    orth: str = "householder"
    sheaf_act: str = "tanh"
    entity: str = "local"
    wandb_mode: str = "disabled"


def default_real_dataset_configs(layers=2, epochs=500, folds=10, seed=43):
    base = {
        "layers": int(layers),
        "epochs": int(epochs),
        "folds": int(folds),
        "seed": int(seed),
    }
    return [
        RealDatasetExperimentConfig(
            dataset="texas",
            weight_decay=5e-3,
            dropout=0.7,
            input_dropout=0.0,
            sparse_learner=True,
            edge_weights=True,
            **base,
        ),
        RealDatasetExperimentConfig(
            dataset="wisconsin",
            weight_decay=0.0006685729356079199,
            dropout=0.7276458263736642,
            input_dropout=0.0,
            sparse_learner=False,
            edge_weights=True,
            **base,
        ),
        RealDatasetExperimentConfig(
            dataset="cornell",
            weight_decay=0.0006914841722570725,
            sheaf_decay=0.00031764232712732976,
            dropout=0.7,
            input_dropout=0.2,
            sparse_learner=False,
            edge_weights=True,
            **base,
        ),
    ]


def with_real_dataset_layers(configs, layers):
    return [replace(config, layers=int(layers)) for config in configs]


def bool_arg(value):
    return "True" if bool(value) else "False"


def real_dataset_run_command(config):
    command = [
        sys.executable,
        "-m",
        "exp.run",
        "--dataset", config.dataset,
        "--d", str(config.d),
        "--layers", str(config.layers),
        "--hidden_channels", str(config.hidden_channels),
        "--left_weights", bool_arg(config.left_weights),
        "--right_weights", bool_arg(config.right_weights),
        "--lr", str(config.lr),
        "--epochs", str(config.epochs),
        "--folds", str(config.folds),
        "--weight_decay", str(config.weight_decay),
        "--input_dropout", str(config.input_dropout),
        "--dropout", str(config.dropout),
        "--use_act", bool_arg(config.use_act),
        "--model", config.model,
        "--normalised", bool_arg(config.normalised),
        "--deg_normalised", bool_arg(config.deg_normalised),
        "--sparse_learner", bool_arg(config.sparse_learner),
        "--edge_weights", bool_arg(config.edge_weights),
        "--add_hp", bool_arg(config.add_hp),
        "--add_lp", bool_arg(config.add_lp),
        "--orth", config.orth,
        "--sheaf_act", config.sheaf_act,
        "--early_stopping", str(config.early_stopping),
        "--entity", config.entity,
    ]
    if config.sheaf_decay is not None:
        command += ["--sheaf_decay", str(config.sheaf_decay)]
    return command


def run_real_dataset_experiment(config, project_root=ROOT_DIR, check=True):
    env = os.environ.copy()
    env["WANDB_MODE"] = config.wandb_mode
    return subprocess.run(
        real_dataset_run_command(config),
        cwd=project_root,
        env=env,
        check=check,
        text=True,
    )


def materialise_real_dataset(name):
    return get_dataset(name.lower(), args=None)[0]


def real_dataset_summary(names=("texas", "wisconsin", "cornell")):
    rows = []
    for name in names:
        data = materialise_real_dataset(name)
        rows.append({
            "dataset": name,
            "nodes": int(data.x.size(0)),
            "features": int(data.x.size(1)),
            "classes": int(torch.unique(data.y).numel()),
            "directed_edges": int(data.edge_index.size(1)),
            "undirected_edges": int(len(undirected_edge_set(data.edge_index))),
        })
    return pd.DataFrame(rows)


def real_node_frame(data, dataset):
    x = data.x.detach().cpu()
    y = data.y.detach().cpu().long()
    names = REAL_DATASET_CLASS_NAMES.get(dataset.lower())
    if names and int(y.max()) < len(names):
        class_names = [names[int(label)] for label in y]
    else:
        class_names = [str(int(label)) for label in y]
    return pd.DataFrame({
        "node": np.arange(x.size(0), dtype=int),
        "class": y.numpy().astype(int),
        "class_name": class_names,
        "feature_norm": torch.linalg.norm(x.float(), dim=1).numpy(),
    })


def laplacian_dir_for_config(config, project_root=ROOT_DIR):
    return (
        Path(project_root)
        / "results"
        / "laplacians"
        / config.dataset
        / f"normalised-{str(config.normalised).lower()}"
        / f"stalk_dim-{config.d}"
        / f"{config.layers}-layers"
        / f"{config.hidden_channels}-hidden"
        / f"{config.epochs}-epochs"
    )


def trained_laplacian_path(config, layer="last", fold=0, project_root=ROOT_DIR):
    if layer == "last":
        layer = int(config.layers) - 1
    layer = int(layer)
    path = laplacian_dir_for_config(config, project_root) / (
        f"{config.model}_{config.dataset}_layer{layer}_fold{int(fold)}_seed{config.seed}.pt"
    )
    return path


def representation_dir_for_config(config, project_root=ROOT_DIR):
    return (
        Path(project_root)
        / "results"
        / "representations"
        / config.dataset
        / f"normalised-{str(config.normalised).lower()}"
        / f"stalk_dim-{config.d}"
        / f"{config.layers}-layers"
        / f"{config.hidden_channels}-hidden"
        / f"{config.epochs}-epochs"
    )


def trained_representations_path(config, fold=0, project_root=ROOT_DIR):
    return representation_dir_for_config(config, project_root) / (
        f"{config.model}_{config.dataset}_fold{int(fold)}_seed{config.seed}.pt"
    )


def load_trained_representations(config, fold=0, project_root=ROOT_DIR):
    path = trained_representations_path(config, fold=fold, project_root=project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained representation artifact: {path}. "
            "Run the real-dataset experiment cell after updating exp.run, or adjust REAL_DIAGNOSTIC_FOLD."
        )
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict) and "representations" in payload:
        return payload["representations"], payload.get("metadata", {}), path
    return payload, {}, path


def representation_key_order(representations):
    def key_rank(name):
        if name == "input":
            return (-2, -1)
        if name == "encoded":
            return (-1, -1)
        if str(name).startswith("layer"):
            try:
                return (0, int(str(name).replace("layer", "")))
            except ValueError:
                return (0, 10**9)
        if name == "pre_logits":
            return (1, -1)
        if name == "logits":
            return (2, -1)
        return (3, str(name))

    return sorted(representations, key=key_rank)


def load_trained_edge_weights(config, data, layer="last", fold=0, project_root=ROOT_DIR):
    path = trained_laplacian_path(config, layer=layer, fold=fold, project_root=project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained Laplacian artifact: {path}. "
            "Run the real-dataset experiment cell first, or adjust REAL_DIAGNOSTIC_LAYER/FOLD."
        )
    laplacian = torch.load(path, weights_only=False)
    edge_weights = laplacian_matrix_to_edge_weights(
        laplacian,
        num_nodes=data.x.size(0),
        stalk_dim=config.d,
        edge_index=data.edge_index,
        strict_edge_index=True,
    )
    return edge_weights, path


def build_weighted_graph(edge_weights, num_nodes, min_distance=1e-12):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    for (u, v), value in sorted(edge_weights.items()):
        distance = max(abs(float(value)), min_distance)
        graph.add_edge(int(u), int(v), weight=distance, laplacian_weight=distance)
    return graph


def load_layer_weight_graphs(config, data, layers=None, fold=0, project_root=ROOT_DIR):
    layers = list(range(config.layers)) if layers is None else list(layers)
    layer_graphs = {}
    layer_paths = {}
    for layer in layers:
        edge_weights, path = load_trained_edge_weights(
            config,
            data,
            layer=int(layer),
            fold=fold,
            project_root=project_root,
        )
        layer_graphs[int(layer)] = build_weighted_graph(edge_weights, num_nodes=data.x.size(0))
        layer_paths[int(layer)] = path
    return layer_graphs, layer_paths


def edge_plot_values(graph, curvature_dict):
    rows = []
    for u, v, attrs in graph.edges(data=True):
        edge = tuple(sorted((int(u), int(v))))
        curvature = curvature_dict.get(edge, attrs.get("ricciCurvature", np.nan))
        if np.isfinite(curvature):
            rows.append((edge, float(attrs["laplacian_weight"]), float(curvature)))
    rows = sorted(rows, key=lambda item: item[0])
    if not rows:
        return [], np.array([]), np.array([])
    edges, weights, curvatures = zip(*rows)
    return list(edges), np.asarray(weights), np.asarray(curvatures)


def graph_positions_for_flow(graph, seed=43):
    return nx.spring_layout(graph, seed=seed, weight=None, iterations=80)


def plot_weight_flow_by_layer(layer_graphs, dataset, positions=None, bins=25):
    if not layer_graphs:
        print(f"No layer graphs available for {dataset}.")
        return

    layers = sorted(layer_graphs)
    if positions is None:
        positions = graph_positions_for_flow(layer_graphs[layers[0]])
    all_weights = np.concatenate([
        np.asarray([attrs["laplacian_weight"] for _, _, attrs in layer_graphs[layer].edges(data=True)], dtype=float)
        for layer in layers
    ])
    vmax = float(np.nanmax(all_weights)) if all_weights.size else None
    fig, axes = plt.subplots(
        2,
        len(layers),
        figsize=(4.2 * len(layers), 7.2),
        squeeze=False,
        constrained_layout=True,
    )

    for col, layer in enumerate(layers):
        graph = layer_graphs[layer]
        edges = list(graph.edges())
        weights = np.asarray([graph.edges[edge]["laplacian_weight"] for edge in edges], dtype=float)
        nx.draw_networkx_nodes(graph, positions, node_size=10, node_color="black", alpha=0.45, ax=axes[0, col])
        edge_collection = nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            edge_color=weights,
            edge_cmap=plt.cm.viridis,
            edge_vmin=0,
            edge_vmax=vmax,
            width=1.15,
            alpha=0.85,
            ax=axes[0, col],
        )
        axes[0, col].set_title(f"Layer {layer}: weights")
        axes[0, col].axis("off")
        plt.colorbar(edge_collection, ax=axes[0, col], fraction=0.046, pad=0.02, label="|L_ij|")

        sns.histplot(weights, bins=bins, kde=True, color="#2563eb", ax=axes[1, col])
        axes[1, col].set_title(f"Layer {layer}: weight distribution")
        axes[1, col].set_xlabel("|L_ij|")
        axes[1, col].set_ylabel("edge count")

    fig.suptitle(f"{dataset.title()} trained NSD weight flow by layer", fontsize=14)
    plt.show()


def plot_curvature_flow_by_layer(layer_curvature_results, dataset, positions=None, tanh_scale=True, bins=25):
    if not layer_curvature_results:
        print(f"No layer curvature results available for {dataset}.")
        return

    layers = sorted(layer_curvature_results)
    first_graph = layer_curvature_results[layers[0]]["graph"]
    if positions is None:
        positions = graph_positions_for_flow(first_graph)
    curvature_label = "tanh(OR curvature)" if tanh_scale else "OR curvature"

    fig, axes = plt.subplots(
        2,
        len(layers),
        figsize=(4.2 * len(layers), 7.2),
        squeeze=False,
        constrained_layout=True,
    )

    for col, layer in enumerate(layers):
        result = layer_curvature_results[layer]
        graph = result["graph"]
        edges, _, curvatures = edge_plot_values(graph, result["edge_curvature"])
        shown_curvatures = np.tanh(curvatures) if tanh_scale else curvatures

        nx.draw_networkx_nodes(graph, positions, node_size=10, node_color="black", alpha=0.45, ax=axes[0, col])
        edge_collection = nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            edge_color=shown_curvatures,
            edge_cmap=plt.cm.coolwarm,
            edge_vmin=-1 if tanh_scale else None,
            edge_vmax=1 if tanh_scale else None,
            width=1.15,
            alpha=0.85,
            ax=axes[0, col],
        )
        axes[0, col].set_title(f"Layer {layer}: curvature")
        axes[0, col].axis("off")
        plt.colorbar(edge_collection, ax=axes[0, col], fraction=0.046, pad=0.02, label=curvature_label)

        sns.histplot(shown_curvatures, bins=bins, kde=True, color="#dc2626", ax=axes[1, col])
        axes[1, col].axvline(0, color="#0f172a", linestyle="--", linewidth=1)
        axes[1, col].set_title(f"Layer {layer}: curvature distribution")
        axes[1, col].set_xlabel(curvature_label)
        axes[1, col].set_ylabel("edge count")

    fig.suptitle(f"{dataset.title()} trained NSD Ollivier-Ricci flow by layer", fontsize=14)
    plt.show()


def build_real_feature_edge_frame(data, dataset, curvature_result, normalised, layer, tanh_scale=True):
    x = data.x.detach().cpu().float().numpy()
    node_df = real_node_frame(data, dataset)
    class_names = node_df.set_index("node")["class_name"].to_dict()
    edges, weights, curvatures = edge_plot_values(curvature_result["graph"], curvature_result["edge_curvature"])
    shown_curvatures = np.tanh(curvatures) if tanh_scale else curvatures

    rows = []
    for (u, v), weight, curvature, shown_curvature in zip(edges, weights, curvatures, shown_curvatures):
        source_norm = float(np.linalg.norm(x[u]))
        target_norm = float(np.linalg.norm(x[v]))
        denom = source_norm * target_norm
        rows.append({
            "dataset": dataset,
            "normalised": bool(normalised),
            "layer": int(layer),
            "source": int(u),
            "target": int(v),
            "source_class": class_names[int(u)],
            "target_class": class_names[int(v)],
            "class_pair": "-".join(sorted((class_names[int(u)], class_names[int(v)]))),
            "feature_norm_diff": abs(source_norm - target_norm),
            "feature_distance": float(np.linalg.norm(x[u] - x[v])),
            "feature_cosine": float(np.dot(x[u], x[v]) / denom) if denom > 0 else np.nan,
            "weight": float(weight),
            "curvature": float(curvature),
            "shown_curvature": float(shown_curvature),
        })
    return pd.DataFrame(rows)


def build_local_real_class_frame(edge_df, node_df):
    rows = []
    for node, class_name in node_df[["node", "class_name"]].itertuples(index=False):
        incident = edge_df[(edge_df["source"] == node) | (edge_df["target"] == node)].copy()
        neighbour_classes = []
        for row in incident.itertuples(index=False):
            neighbour_classes.append(row.target_class if int(row.source) == int(node) else row.source_class)
        neighbour_classes = np.asarray(neighbour_classes, dtype=object)
        local_heterogeneity = float((neighbour_classes != class_name).mean()) if len(neighbour_classes) else np.nan
        rows.append({
            "node": int(node),
            "class_name": class_name,
            "laplacian_degree": int(len(incident)),
            "local_class_heterogeneity": local_heterogeneity,
            "avg_1hop_weight": float(incident["weight"].mean()) if len(incident) else np.nan,
            "avg_1hop_curvature": float(incident["shown_curvature"].mean()) if len(incident) else np.nan,
        })
    local_df = pd.DataFrame(rows)
    local_df["class_heterogeneity_bin"] = pd.cut(
        local_df["local_class_heterogeneity"],
        bins=[-0.001, 0.0, 0.25, 0.5, 0.75, 1.0],
        labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]"],
    )
    return local_df


def pca_coordinates(x, n_components=3, return_model=False):
    from sklearn.decomposition import PCA

    x = x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=float)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(x)
    if return_model:
        return coords, pca
    return coords


def pca_diagnostic_frame(coords, y, explained_variance_ratio):
    frame = pd.DataFrame({
        "node": np.arange(coords.shape[0]),
        "class": y,
    })
    for component in range(coords.shape[1]):
        frame[f"pc{component + 1}"] = coords[:, component]
        frame[f"pc{component + 1}_explained_variance_ratio"] = float(explained_variance_ratio[component])
    frame.attrs["explained_variance_ratio"] = explained_variance_ratio.tolist()
    frame.attrs["explained_variance_ratio_total"] = float(explained_variance_ratio.sum())
    return frame


def pca_axis_label(component, explained_variance_ratio):
    return f"PC{component + 1} ({100 * explained_variance_ratio[component]:.1f}%)"


def plot_pca_3d(data, dataset, title=None, x=None, representation_name="input"):
    source_x = data.x if x is None else x
    coords, pca = pca_coordinates(source_x, n_components=3, return_model=True)
    explained = pca.explained_variance_ratio_
    y = data.y.detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=y, cmap="tab10", s=22, alpha=0.85)
    ax.set_xlabel(pca_axis_label(0, explained))
    ax.set_ylabel(pca_axis_label(1, explained))
    ax.set_zlabel(pca_axis_label(2, explained))
    total_explained = 100 * explained.sum()
    ax.set_title(title or f"{dataset.title()} {representation_name} PCA (3D, {total_explained:.1f}% var.)")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="class")
    plt.tight_layout()
    plt.show()
    return pca_diagnostic_frame(coords, y, explained)

def plot_pca_2d(data, dataset, title=None, x=None, representation_name="input"):
    source_x = data.x if x is None else x
    coords, pca = pca_coordinates(source_x, n_components=2, return_model=True)
    explained = pca.explained_variance_ratio_
    y = data.y.detach().cpu().numpy()
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=22, alpha=0.85)
    ax.set_xlabel(pca_axis_label(0, explained))
    ax.set_ylabel(pca_axis_label(1, explained))
    total_explained = 100 * explained.sum()
    ax.set_title(title or f"{dataset.title()} {representation_name} PCA (2D, {total_explained:.1f}% var.)")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="class")
    plt.tight_layout()
    plt.show()
    return pca_diagnostic_frame(coords, y, explained)


def plot_umap_2d(data, dataset, n_neighbors=15, min_dist=0.1, random_state=43, x=None, representation_name="input"):
    try:
        import umap
    except ImportError:
        print("UMAP is not installed. Install umap-learn to enable this diagnostic.")
        return pd.DataFrame()

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    source_x = data.x if x is None else x
    source_x = source_x.detach().cpu().float().numpy() if isinstance(source_x, torch.Tensor) else np.asarray(source_x, dtype=float)
    coords = reducer.fit_transform(source_x)
    y = data.y.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=24, alpha=0.85, edgecolors="black", linewidths=0.2)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"{dataset.title()} {representation_name} UMAP")
    fig.colorbar(scatter, ax=ax, label="class")
    plt.tight_layout()
    plt.show()
    return pd.DataFrame({"node": np.arange(coords.shape[0]), "umap1": coords[:, 0], "umap2": coords[:, 1], "class": y})


def plot_representation_pca_umap(
    data,
    dataset,
    representations,
    representation_names,
    n_neighbors=15,
    min_dist=0.1,
    random_state=43,
    plot_umap=True,
):
    pca_tables = {}
    umap_tables = {}
    for name in representation_names:
        if name not in representations:
            print(f"Representation {name!r} not found for {dataset}. Available: {list(representations)}")
            continue
        x = representations[name]
        pca_tables[name] = plot_pca_3d(data, dataset, x=x, representation_name=name)
        if plot_umap:
            umap_tables[name] = plot_umap_2d(
                data,
                dataset,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                x=x,
                representation_name=name,
            )
    return pca_tables, umap_tables


def plot_representation_flow_by_layer(
    data,
    dataset,
    representations,
    representation_names,
    n_neighbors=15,
    min_dist=0.1,
    random_state=43,
):
    available_names = [name for name in representation_names if name in representations]
    missing = [name for name in representation_names if name not in representations]
    for name in missing:
        print(f"Representation {name!r} not found for {dataset}. Available: {list(representations)}")
    if not available_names:
        print(f"No requested representations available for {dataset}.")
        return {}, {}

    try:
        import umap
    except ImportError:
        umap = None
        print("UMAP is not installed. The UMAP row will be left empty.")

    y = data.y.detach().cpu().numpy()
    pca_tables = {}
    umap_tables = {}
    fig, axes = plt.subplots(
        2,
        len(available_names),
        figsize=(4.4 * len(available_names), 8.0),
        squeeze=False,
        constrained_layout=True,
    )

    for col, name in enumerate(available_names):
        x = representations[name]
        coords, pca = pca_coordinates(x, n_components=2, return_model=True)
        explained = pca.explained_variance_ratio_
        pca_tables[name] = pca_diagnostic_frame(coords, y, explained)
        axes[0, col].scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=24, alpha=0.85, edgecolors="black", linewidths=0.2)
        axes[0, col].set_title(f"{name}: PCA ({100 * explained.sum():.1f}% var.)")
        axes[0, col].set_xlabel(pca_axis_label(0, explained))
        axes[0, col].set_ylabel(pca_axis_label(1, explained))

        if umap is not None:
            source_x = x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=float)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            umap_coords = reducer.fit_transform(source_x)
            umap_tables[name] = pd.DataFrame({
                "node": np.arange(umap_coords.shape[0]),
                "umap1": umap_coords[:, 0],
                "umap2": umap_coords[:, 1],
                "class": y,
            })
            axes[1, col].scatter(umap_coords[:, 0], umap_coords[:, 1], c=y, cmap="tab10", s=24, alpha=0.85, edgecolors="black", linewidths=0.2)
            axes[1, col].set_title(f"{name}: UMAP")
            axes[1, col].set_xlabel("UMAP1")
            axes[1, col].set_ylabel("UMAP2")
        else:
            axes[1, col].axis("off")

    fig.suptitle(f"{dataset.title()} representation flow", fontsize=14)
    plt.show()
    return pca_tables, umap_tables


def plot_real_graph_views(data, dataset, curvature_result, title=None, tanh_scale=True, seed=43):
    graph = curvature_result["graph"]
    node_df = real_node_frame(data, dataset)
    edges, weights, curvatures = edge_plot_values(graph, curvature_result["edge_curvature"])
    if not edges:
        print(f"No graph-view edge diagnostics available for {dataset}.")
        return

    shown_curvatures = np.tanh(curvatures) if tanh_scale else curvatures
    curvature_label = "tanh(OR curvature)" if tanh_scale else "OR curvature"
    positions = nx.spring_layout(graph, seed=seed, weight=None, iterations=80)

    fig, ax = plt.subplot_mosaic(
        [["weights_graph", "curvature_graph", "classes"]],
        figsize=(18, 5),
        constrained_layout=True,
    )
    nx.draw_networkx_nodes(graph, positions, node_size=10, node_color="black", alpha=0.45, ax=ax["weights_graph"])
    edge_collection = nx.draw_networkx_edges(
        graph, positions, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.viridis,
        width=1.2, alpha=0.85, ax=ax["weights_graph"],
    )
    ax["weights_graph"].set_title("Weights on edges")
    ax["weights_graph"].axis("off")
    plt.colorbar(edge_collection, ax=ax["weights_graph"], fraction=0.046, pad=0.02, label="|L_ij|")

    nx.draw_networkx_nodes(graph, positions, node_size=10, node_color="black", alpha=0.45, ax=ax["curvature_graph"])
    edge_collection = nx.draw_networkx_edges(
        graph, positions, edgelist=edges, edge_color=shown_curvatures, edge_cmap=plt.cm.coolwarm,
        edge_vmin=-1 if tanh_scale else None, edge_vmax=1 if tanh_scale else None,
        width=1.2, alpha=0.85, ax=ax["curvature_graph"],
    )
    ax["curvature_graph"].set_title("Curvature on edges")
    ax["curvature_graph"].axis("off")
    plt.colorbar(edge_collection, ax=ax["curvature_graph"], fraction=0.046, pad=0.02, label=curvature_label)

    class_by_node = node_df.set_index("node").reindex(list(graph.nodes()))["class"].to_numpy()
    nx.draw_networkx_edges(graph, positions, edge_color="#cbd5e1", width=.7, alpha=0.45, ax=ax["classes"])
    nodes = nx.draw_networkx_nodes(
        graph, positions, node_color=class_by_node, cmap=plt.cm.tab10,
        node_size=22, edgecolors="#0f172a", linewidths=0.2, ax=ax["classes"],
    )
    ax["classes"].set_title("Classes on nodes")
    ax["classes"].axis("off")
    plt.colorbar(nodes, ax=ax["classes"], fraction=0.046, pad=0.02, label="class")
    fig.suptitle(title or f"{dataset.title()} trained NSD graph views", fontsize=14)
    plt.show()


def _linear_fit(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0:
        return np.nan
    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=2)
    r = np.corrcoef(x[valid], y[valid])[0, 1] if np.std(y[valid]) > 0 else np.nan
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha="left", va="top")
    return float(r)


def plot_real_classical_diagnostics(edge_df, local_df, dataset, title=None):
    if edge_df.empty:
        print(f"No edge diagnostics available for {dataset}.")
        return

    fig, ax = plt.subplot_mosaic(
        [["weight_hist", "curvature_hist", "correlation"], ["edge_weight", "edge_curvature", "node_curvature"]],
        figsize=(18, 9),
        constrained_layout=True,
    )
    sns.histplot(edge_df["weight"], bins=25, kde=True, color="#2563eb", ax=ax["weight_hist"])
    sns.histplot(edge_df["shown_curvature"], bins=25, kde=True, color="#dc2626", ax=ax["curvature_hist"])
    sns.scatterplot(data=edge_df, x="weight", y="shown_curvature", s=35, color="#059669", edgecolor="white", ax=ax["correlation"])
    r = _linear_fit(ax["correlation"], edge_df["weight"], edge_df["shown_curvature"])

    class_pair_order = sorted(edge_df["class_pair"].dropna().unique())
    sns.boxplot(data=edge_df, x="class_pair", y="weight", order=class_pair_order, color="#93c5fd", ax=ax["edge_weight"])
    sns.boxplot(data=edge_df, x="class_pair", y="shown_curvature", order=class_pair_order, color="#fca5a5", ax=ax["edge_curvature"])
    sns.boxplot(
        data=local_df.dropna(subset=["class_heterogeneity_bin"]),
        x="class_heterogeneity_bin",
        y="avg_1hop_curvature",
        color="#fca5a5",
        ax=ax["node_curvature"],
    )

    ax["weight_hist"].set(title="Weight distribution", xlabel="|L_ij|", ylabel="edge count")
    ax["curvature_hist"].axvline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["curvature_hist"].set(title="Curvature distribution", xlabel="tanh(OR curvature)", ylabel="edge count")
    ax["correlation"].axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["correlation"].set(title=f"Weights vs curvature (r={r:.3f})", xlabel="|L_ij|", ylabel="tanh(OR curvature)")
    ax["edge_weight"].set(title="Edge class pair vs weight", xlabel="edge class pair", ylabel="|L_ij|")
    ax["edge_curvature"].axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["edge_curvature"].set(title="Edge class pair vs curvature", xlabel="edge class pair", ylabel="shown curvature")
    ax["node_curvature"].axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["node_curvature"].set(title="1-hop class heterogeneity vs avg curvature", xlabel="local class heterogeneity", ylabel="avg incident curvature")

    for name in ["edge_weight", "edge_curvature"]:
        ax[name].tick_params(axis="x", rotation=25)
    fig.suptitle(title or f"{dataset.title()} trained NSD diagnostics", fontsize=14)
    plt.show()
