"""
Phase 3d — Feature Correlation Network

Builds a network graph where nodes = dQ/dV features and edges connect
pairs with |r| > threshold. The Louvain algorithm identifies feature
communities.

References:
    - Blondel et al. (2008), J. Statistical Mechanics, P10008.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain  # python-louvain package

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def build_correlation_network(corr_matrix: pd.DataFrame,
                               threshold: float = config.NETWORK_EDGE_THRESHOLD
                               ) -> nx.Graph:
    """
    Build a NetworkX graph from the correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Feature-by-feature Pearson correlation matrix.
    threshold : float
        Minimum |r| to create an edge.

    Returns
    -------
    nx.Graph
        Weighted graph with |r| as edge weights.
    """
    G = nx.Graph()
    features = corr_matrix.columns.tolist()
    G.add_nodes_from(features)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                G.add_edge(features[i], features[j],
                           weight=abs(r), correlation=r)
    return G


def detect_communities(G: nx.Graph) -> dict:
    """
    Apply Louvain community detection.

    Parameters
    ----------
    G : nx.Graph
        Correlation network.

    Returns
    -------
    dict
        Mapping of node → community ID.
    """
    return community_louvain.best_partition(G, random_state=42)


def plot_network(G: nx.Graph, partition: dict,
                 save_path=None) -> plt.Figure:
    """
    Visualize the correlation network with community coloring.
    """
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)

    pos = nx.spring_layout(G, seed=42, k=2)

    # Edge widths proportional to |r|
    edges = G.edges(data=True)
    widths = [d["weight"] * 3 for _, _, d in edges]
    edge_colors = ["red" if d["correlation"] < 0 else "steelblue"
                   for _, _, d in edges]

    # Node colors by community
    communities = set(partition.values())
    cmap = plt.cm.Set2
    node_colors = [cmap(partition[n] / max(len(communities), 1))
                   for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                           edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Edge weight labels
    edge_labels = {(u, v): f"{d['correlation']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    ax.set_title("dQ/dV Feature Correlation Network", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Network plot saved → {save_path}")
    return fig


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 3d: Correlation Network")
    print("=" * 35)

    df = pd.read_csv(config.SAMPLE_DATA)
    corr = df[config.ALL_FEATURES].corr()

    G = build_correlation_network(corr)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = detect_communities(G)
    n_communities = len(set(partition.values()))
    print(f"Communities detected: {n_communities}")

    for comm_id in sorted(set(partition.values())):
        members = [n for n, c in partition.items() if c == comm_id]
        print(f"  Community {comm_id}: {members}")

    plot_network(G, partition,
                 config.FIGURES_DIR / f"correlation_network.{config.FIGURE_FORMAT}")
    plt.close("all")
