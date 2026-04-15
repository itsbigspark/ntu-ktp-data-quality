"""
Page 13: Entity Graph -- Interactive network visualisation of entity resolution matches.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("ENTITY GRAPH", "Interactive network visualisation of entity matches")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
use_from_er_tab = False
if "er_pairs" in st.session_state and st.session_state["er_pairs"] is not None and len(st.session_state["er_pairs"]) > 0:
    use_from_er_tab = st.checkbox("Use results from Entity Resolution page", value=True)

if use_from_er_tab:
    pdf = st.session_state["er_pairs"]
    datasets_for_preview = st.session_state.get("er_datasets")
    st.success(f"Loaded {len(pdf)} matches from Entity Resolution")
else:
    pair_df_file = st.file_uploader(
        "Upload ER pair list (CSV)", type=["csv"], key="graph_pair_upload",
        help="Upload the CSV downloaded from Entity Resolution",
    )
    if pair_df_file is None:
        st.info("Upload an ER pairs CSV file or run Entity Resolution first.")
        st.stop()
    pdf = pd.read_csv(pair_df_file)
    datasets_for_preview = None

if pdf is None or len(pdf) == 0:
    st.warning("No data to visualise")
    st.stop()

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
import networkx as nx
from core.entity_resolution import analyze_clusters

G = nx.Graph()

for r in pdf.itertuples(index=False):
    A = getattr(r, "A", None)
    B = getattr(r, "B", None)
    A_row = getattr(r, "A_row", None)
    B_row = getattr(r, "B_row", None)
    score = getattr(r, "score", None)

    if A is None or B is None or A_row is None or B_row is None:
        continue

    try:
        w = float(score) if score is not None else 0.0
    except Exception:
        w = 0.0

    G.add_edge(f"{A}:{A_row}", f"{B}:{B_row}", weight=w)

# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------
section_header("// Graph Statistics")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Nodes", G.number_of_nodes())
s2.metric("Total Edges", G.number_of_edges())
num_components = nx.number_connected_components(G)
s3.metric("Clusters", num_components)
if G.number_of_nodes() > 0:
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    s4.metric("Avg Connections", f"{avg_degree:.1f}")
else:
    s4.metric("Avg Connections", "0")

# Cluster analysis
if "er_cluster_stats" in st.session_state:
    cluster_stats = st.session_state["er_cluster_stats"]
else:
    cluster_stats = analyze_clusters(pdf)
    st.session_state["er_cluster_stats"] = cluster_stats

with st.expander("Detailed Cluster Analysis"):
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Avg Size", f"{cluster_stats['avg_cluster_size']:.1f}")
        st.metric("Max Size", cluster_stats["max_cluster_size"])
    with d2:
        st.metric("Multi-Node", cluster_stats["multi_node_clusters"])
        st.metric("Singletons", cluster_stats["singleton_clusters"])
    with d3:
        if cluster_stats["golden_candidates"]:
            st.metric("Golden Candidates", len(cluster_stats["golden_candidates"]))
        else:
            st.metric("Golden Candidates", 0)

st.markdown("---")

# ---------------------------------------------------------------------------
# Visualisation options
# ---------------------------------------------------------------------------
section_header("// Visualisation Options")

v1, v2, v3 = st.columns(3)
with v1:
    graph_height = st.selectbox("Graph Height", ["500px", "650px", "800px", "1000px"], index=1, key="graph_height_select")
with v2:
    color_by = st.selectbox("Color Nodes By", ["Dataset", "Cluster", "Degree"])
with v3:
    show_labels = st.checkbox("Show Labels", value=True)

min_cluster_size = st.slider(
    "Minimum Cluster Size to Display", 1,
    max(cluster_stats["cluster_sizes"]) if cluster_stats["cluster_sizes"] else 1, 1,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Render graph
# ---------------------------------------------------------------------------
section_header("// Interactive Graph")

try:
    from pyvis.network import Network
    import streamlit.components.v1 as components

    # Filter by cluster size
    if min_cluster_size > 1:
        clusters = list(nx.connected_components(G))
        nodes_to_keep = set()
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                nodes_to_keep.update(cluster)
        G_filtered = G.subgraph(nodes_to_keep).copy()
        st.caption(f"Filtered to {G_filtered.number_of_nodes()} nodes in {nx.number_connected_components(G_filtered)} clusters")
    else:
        G_filtered = G

    net = Network(height=graph_height, width="100%", directed=False, bgcolor="#0a0a0a", font_color="#00ff41")
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
        }
    }
    """)

    colors_palette = ["#00ff41", "#00e5ff", "#ff6600", "#ff0066", "#9467bd",
                      "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#ffff00"]

    if color_by == "Dataset":
        datasets_set = set()
        for node in G_filtered.nodes():
            datasets_set.add(node.split(":")[0] if ":" in node else "unknown")
        dataset_colors = {ds: colors_palette[i % len(colors_palette)] for i, ds in enumerate(sorted(datasets_set))}

        for node in G_filtered.nodes():
            dataset = node.split(":")[0] if ":" in node else "unknown"
            net.add_node(node, label=node if show_labels else "", color=dataset_colors.get(dataset, "#888"), title=node)

    elif color_by == "Cluster":
        clusters_list = list(nx.connected_components(G_filtered))
        node_to_cluster = {}
        for idx, cluster in enumerate(clusters_list):
            for node in cluster:
                node_to_cluster[node] = idx

        for node in G_filtered.nodes():
            cidx = node_to_cluster.get(node, 0)
            net.add_node(node, label=node if show_labels else "", color=colors_palette[cidx % len(colors_palette)], title=node)

    elif color_by == "Degree":
        degrees = dict(G_filtered.degree())
        max_degree = max(degrees.values()) if degrees else 1
        for node in G_filtered.nodes():
            degree = degrees.get(node, 0)
            intensity = int(255 * (1 - degree / max_degree))
            color = f"#{intensity:02x}{intensity:02x}ff"
            size = 10 + (degree * 2)
            net.add_node(node, label=node if show_labels else "", color=color, title=f"{node} (degree: {degree})", size=size)

    for u, v, d in G_filtered.edges(data=True):
        weight = d.get("weight", 0.0)
        net.add_edge(u, v, value=max(1, int(weight * 5)), title=f"Similarity: {weight:.3f}")

    out_html = os.path.join(_ROOT, "output", "er_graph.html")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    net.save_graph(out_html)

    with open(out_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=int(graph_height.replace("px", "")))
    st.caption("Click and drag nodes to explore. Zoom with mouse wheel.")

except ImportError:
    st.error("PyVis not installed. Install with: pip install pyvis networkx")
except Exception as e:
    st.error(f"Graph rendering failed: {e}")

# ---------------------------------------------------------------------------
# Cluster details
# ---------------------------------------------------------------------------
if cluster_stats["multi_node_clusters"] > 0:
    st.markdown("---")
    section_header("// Cluster Details")

    clusters = list(nx.connected_components(G))
    multi_node_clusters = [c for c in clusters if len(c) > 1]

    if multi_node_clusters:
        cluster_options = [f"Cluster {i+1} ({len(c)} nodes)" for i, c in enumerate(multi_node_clusters)]
        selected_cluster_idx = st.selectbox(
            "Select Cluster to Inspect",
            range(len(cluster_options)),
            format_func=lambda x: cluster_options[x],
            key="cluster_inspect_select",
        )

        selected_cluster = multi_node_clusters[selected_cluster_idx]
        st.markdown(f"**Size:** {len(selected_cluster)} nodes")

        for node in sorted(selected_cluster):
            st.markdown(f"- `{node}`")

        subgraph = G.subgraph(selected_cluster)
        centrality = nx.degree_centrality(subgraph)
        best_node = max(centrality, key=centrality.get)

        st.markdown(f"**Suggested Golden Record:** `{best_node}` (centrality: {centrality[best_node]:.3f})")

        if datasets_for_preview:
            try:
                from core.entity_resolution import get_record_preview
                with st.expander("View Record Details"):
                    record_data = get_record_preview(datasets_for_preview, best_node)
                    if "data" in record_data:
                        st.markdown(f"**Dataset:** {record_data['dataset']} | **Row:** {record_data['row']}")
                        record_df = pd.DataFrame([record_data["data"]]).T
                        record_df.columns = ["Value"]
                        st.dataframe(record_df, use_container_width=True)
                    else:
                        st.warning("Record details not available")
            except ImportError:
                pass
