from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from definitions import ROOT_DIR
from utils.heterophilic import get_dataset


CIRCLE_CLASS_NAMES = {
    0: "left",
    1: "right",
    2: "upper_lower",
}


@dataclass
class CircleExperimentConfig:
    name: str = "circle"
    circle_topology: str = "circle"
    circle_nodes: int = 64
    circle_k: int = 1
    circle_cross_stride: int = 2
    circle_side_margin: float = 0.35
    circle_feature_mode: str = "coords"
    circle_feature_noise: float = 0.05
    num_feats: int = 8
    seed: int = 43


def make_circle_args(config):
    return SimpleNamespace(
        circle_topology=config.circle_topology,
        circle_nodes=int(config.circle_nodes),
        circle_k=int(config.circle_k),
        circle_cross_stride=int(config.circle_cross_stride),
        circle_side_margin=float(config.circle_side_margin),
        circle_feature_mode=config.circle_feature_mode,
        circle_feature_noise=float(config.circle_feature_noise),
        num_feats=int(config.num_feats),
        seed=int(config.seed),
    )


def clear_circle_cache(project_root=ROOT_DIR):
    project_root = Path(project_root)
    for path in [
        project_root / "datasets" / "circle_exp" / "processed",
        project_root / "datasets" / "circle_exp" / "raw",
    ]:
        if path.exists():
            shutil.rmtree(path)
    for split_path in (project_root / "splits").glob("circle_exp_split_0.6_0.2_*.npz"):
        split_path.unlink()


def materialise_circle_data(config, refresh=True, project_root=ROOT_DIR):
    if refresh:
        clear_circle_cache(project_root)
    dataset = get_dataset("circle_exp", make_circle_args(config))
    data = dataset[0]
    data.topology = str(getattr(data, "circle_topology", config.circle_topology))
    data.num_classes = 3
    return data


def default_circle_experiment_configs(seed=43):
    return [
        CircleExperimentConfig(name="circle", circle_topology="circle", circle_k=1, seed=seed),
        CircleExperimentConfig(name="crossed_circle", circle_topology="crossed_circle", circle_k=1, seed=seed),
        CircleExperimentConfig(name="knn_circle_k2", circle_topology="knn_circle", circle_k=2, seed=seed),
    ]


def circle_node_frame(data):
    x = data.x.detach().cpu()
    y = data.y.detach().cpu().long()
    pos = getattr(data, "pos", None)
    if pos is None:
        angles = torch.linspace(0, 2 * torch.pi, x.size(0) + 1)[:-1]
        pos = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    pos = pos.detach().cpu()
    return pd.DataFrame({
        "node": np.arange(x.size(0), dtype=int),
        "class": y.numpy().astype(int),
        "class_name": [CIRCLE_CLASS_NAMES[int(label)] for label in y],
        "role": [CIRCLE_CLASS_NAMES[int(label)] for label in y],
        "role_id": y.numpy().astype(int),
        "pos_x": pos[:, 0].numpy(),
        "pos_y": pos[:, 1].numpy(),
        "feature_norm": torch.linalg.norm(x, dim=1).numpy(),
    })


def edge_plot_values(graph, curvature_dict):
    rows = []
    for u, v, attrs in graph.edges(data=True):
        edge = tuple(sorted((int(u), int(v))))
        curvature = curvature_dict.get(edge, attrs.get("ricciCurvature", np.nan))
        weight = attrs.get("laplacian_weight", attrs.get("weight", 1.0))
        if np.isfinite(curvature):
            rows.append((edge, float(weight), float(curvature)))
    rows = sorted(rows, key=lambda item: item[0])
    if not rows:
        return [], np.array([]), np.array([])
    edges, weights, curvatures = zip(*rows)
    return list(edges), np.asarray(weights), np.asarray(curvatures)


def build_circle_feature_edge_frame(data, curvature_result, topology, normalised, layer, tanh_scale=True):
    x = data.x.detach().cpu().numpy()
    node_df = circle_node_frame(data)
    class_names = node_df.set_index("node")["class_name"].to_dict()
    edges, weights, curvatures = edge_plot_values(curvature_result["graph"], curvature_result["edge_curvature"])
    shown_curvatures = np.tanh(curvatures) if tanh_scale else curvatures

    rows = []
    for (u, v), weight, curvature, shown_curvature in zip(edges, weights, curvatures, shown_curvatures):
        source_norm = float(np.linalg.norm(x[u]))
        target_norm = float(np.linalg.norm(x[v]))
        denom = source_norm * target_norm
        rows.append({
            "topology": topology,
            "normalised": bool(normalised),
            "layer": int(layer),
            "source": int(u),
            "target": int(v),
            "source_class": class_names[int(u)],
            "target_class": class_names[int(v)],
            "class_pair": "-".join(sorted((class_names[int(u)], class_names[int(v)]))),
            "source_feature_norm": source_norm,
            "target_feature_norm": target_norm,
            "feature_norm_diff": abs(source_norm - target_norm),
            "feature_distance": float(np.linalg.norm(x[u] - x[v])),
            "feature_cosine": float(np.dot(x[u], x[v]) / denom) if denom > 0 else np.nan,
            "weight": float(weight),
            "curvature": float(curvature),
            "shown_curvature": float(shown_curvature),
        })
    return pd.DataFrame(rows)


def build_local_circle_class_frame(edge_df, node_df):
    rows = []
    for node, class_name in node_df[["node", "class_name"]].itertuples(index=False):
        incident = edge_df[(edge_df["source"] == node) | (edge_df["target"] == node)].copy()
        neighbour_classes = []
        for row in incident.itertuples(index=False):
            if int(row.source) == int(node):
                neighbour_classes.append(row.target_class)
            else:
                neighbour_classes.append(row.source_class)
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


def plot_basic_circle_diagnostics(data, node_df, curvature_result, title, tanh_scale=True, bins=25):
    graph = curvature_result["graph"]
    edges, weights, curvatures = edge_plot_values(graph, curvature_result["edge_curvature"])
    if len(edges) == 0:
        print(f"No edge values available for {title}.")
        return

    shown_curvatures = np.tanh(curvatures) if tanh_scale else curvatures
    curvature_label = "tanh(OR curvature)" if tanh_scale else "OR curvature"
    positions = {int(row.node): (float(row.pos_x), float(row.pos_y)) for row in node_df.itertuples(index=False)}

    fig, ax = plt.subplot_mosaic(
        [["weights_graph", "curvature_graph", "classes"], ["weight_hist", "curvature_hist", "correlation"]],
        figsize=(18, 9),
        constrained_layout=True,
    )

    nx.draw_networkx_nodes(graph, positions, node_size=10, node_color="black", alpha=0.4, ax=ax["weights_graph"])
    edge_collection = nx.draw_networkx_edges(
        graph, positions, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.viridis,
        width=1.6, ax=ax["weights_graph"],
    )
    ax["weights_graph"].set_title("Weights on edges")
    ax["weights_graph"].axis("off")
    plt.colorbar(edge_collection, ax=ax["weights_graph"], fraction=0.046, pad=0.02, label="|L_ij|")

    nx.draw_networkx_nodes(graph, positions, node_size=10, node_color="black", alpha=0.4, ax=ax["curvature_graph"])
    edge_collection = nx.draw_networkx_edges(
        graph, positions, edgelist=edges, edge_color=shown_curvatures, edge_cmap=plt.cm.coolwarm,
        edge_vmin=-1 if tanh_scale else None, edge_vmax=1 if tanh_scale else None,
        width=1.6, ax=ax["curvature_graph"],
    )
    ax["curvature_graph"].set_title("Curvature on edges")
    ax["curvature_graph"].axis("off")
    plt.colorbar(edge_collection, ax=ax["curvature_graph"], fraction=0.046, pad=0.02, label=curvature_label)

    class_by_node = node_df.set_index("node").reindex(list(graph.nodes()))["class"].to_numpy()
    nx.draw_networkx_edges(graph, positions, edge_color="#cbd5e1", width=.8, alpha=0.6, ax=ax["classes"])
    nodes = nx.draw_networkx_nodes(
        graph, positions, node_color=class_by_node, cmap=plt.cm.Set2, vmin=0, vmax=2,
        node_size=34, edgecolors="#0f172a", linewidths=0.35, ax=ax["classes"],
    )
    ax["classes"].set_title("Classes on nodes")
    ax["classes"].axis("off")
    cbar = plt.colorbar(nodes, ax=ax["classes"], fraction=0.046, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([CIRCLE_CLASS_NAMES[i] for i in [0, 1, 2]])
    cbar.set_label("class")

    sns.histplot(weights, bins=bins, kde=True, color="#2563eb", ax=ax["weight_hist"])
    sns.histplot(shown_curvatures, bins=bins, kde=True, color="#dc2626", ax=ax["curvature_hist"])
    sns.scatterplot(x=weights, y=shown_curvatures, s=35, color="#059669", edgecolor="white", ax=ax["correlation"])
    pearson_r = _linear_fit(ax["correlation"], weights, shown_curvatures)

    ax["weight_hist"].set(title="Weight distribution", xlabel="|L_ij|", ylabel="edge count")
    ax["curvature_hist"].axvline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["curvature_hist"].set(title="Curvature distribution", xlabel=curvature_label, ylabel="edge count")
    ax["correlation"].axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["correlation"].set(title=f"Weights vs curvature (r={pearson_r:.3f})", xlabel="|L_ij|", ylabel=curvature_label)
    fig.suptitle(title, fontsize=14)
    plt.show()


def plot_local_circle_class_diagnostics(edge_df, local_df, title):
    if edge_df.empty or local_df.empty:
        print(f"No local class diagnostic data available for {title}.")
        return

    class_pair_order = sorted(edge_df["class_pair"].dropna().unique())
    fig, ax = plt.subplot_mosaic(
        [["edge_weight", "edge_curvature"], ["node_weight", "node_curvature"]],
        figsize=(13, 10),
        constrained_layout=True,
    )
    sns.boxplot(data=edge_df, x="class_pair", y="weight", order=class_pair_order, color="#93c5fd", ax=ax["edge_weight"])
    sns.boxplot(data=edge_df, x="class_pair", y="shown_curvature", order=class_pair_order, color="#fca5a5", ax=ax["edge_curvature"])
    sns.boxplot(data=local_df.dropna(subset=["class_heterogeneity_bin"]), x="class_heterogeneity_bin", y="avg_1hop_weight", color="#93c5fd", ax=ax["node_weight"])
    sns.boxplot(data=local_df.dropna(subset=["class_heterogeneity_bin"]), x="class_heterogeneity_bin", y="avg_1hop_curvature", color="#fca5a5", ax=ax["node_curvature"])

    ax["edge_weight"].set(title="Edge class pair vs weight", xlabel="edge class pair", ylabel="|L_ij|")
    ax["edge_curvature"].axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["edge_curvature"].set(title="Edge class pair vs curvature", xlabel="edge class pair", ylabel="shown curvature")
    ax["node_weight"].set(title="1-hop class heterogeneity bin vs avg weight", xlabel="local class heterogeneity", ylabel="avg incident |L_ij|")
    ax["node_curvature"].axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax["node_curvature"].set(title="1-hop class heterogeneity bin vs avg curvature", xlabel="local class heterogeneity", ylabel="avg incident curvature")
    for name in ["edge_weight", "edge_curvature"]:
        ax[name].tick_params(axis="x", rotation=20)
    fig.suptitle(title, fontsize=14)
    plt.show()


def raw_vs_processed_circle_summary(project_root=ROOT_DIR):
    project_root = Path(project_root)
    raw_path = project_root / "datasets" / "circle_exp" / "raw" / "circle_node_classification_data.pt"
    processed_path = project_root / "datasets" / "circle_exp" / "processed" / "data.pt"
    raw = torch.load(raw_path, weights_only=False) if raw_path.exists() else None
    processed = torch.load(processed_path, weights_only=False)[0] if processed_path.exists() else None
    rows = []
    for name, data in [("raw", raw), ("processed", processed)]:
        if data is None:
            rows.append({"stage": name, "exists": False})
            continue
        edge_index = data.edge_index.detach().cpu()
        edge_pairs = [tuple(edge) for edge in edge_index.t().numpy().tolist()]
        rows.append({
            "stage": name,
            "exists": True,
            "nodes": int(data.x.size(0)),
            "features": int(data.x.size(1)),
            "directed_edges": int(edge_index.size(1)),
            "self_loops": int((edge_index[0] == edge_index[1]).sum().item()),
            "unique_directed_edges": int(len(set(edge_pairs))),
            "has_pos": bool(hasattr(data, "pos")),
            "has_node_role": bool(hasattr(data, "node_role")),
        })
    return pd.DataFrame(rows)
