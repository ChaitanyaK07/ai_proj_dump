import numpy as np
import osmnx as ox

G = ox.load_graphml("data/jubilee_3500.graphml", node_dtypes={"elevation": float})

# 1. Map of node elevations
nc = ox.plot.get_node_colors_by_attr(G, "elevation", cmap="terrain")
ox.plot_graph(G, node_color=nc, node_size=4, edge_linewidth=0.2, bgcolor="white",
              show=False, save=True, filepath="fitness/jubilee_3500_elevation.png", dpi=200)

# 2. Spike check: how far each node sits from the median of its neighbours
U = G.to_undirected()
elev = {n: d["elevation"] for n, d in G.nodes(data=True)}
dev = []
for n in U.nodes:
    nb = [elev[m] for m in U.neighbors(n) if m != n]
    if len(nb) >= 2:
        dev.append(abs(elev[n] - np.median(nb)))
dev = np.array(dev)

print(f"elevation range: {min(elev.values()):.0f} - {max(elev.values()):.0f} m")
print(f"node vs neighbour-median difference: p50 {np.percentile(dev, 50):.1f} m, "
      f"p90 {np.percentile(dev, 90):.1f} m, p99 {np.percentile(dev, 99):.1f} m")
print(f"spikes > 10 m: {(dev > 10).sum()} nodes ({100 * (dev > 10).mean():.2f}%)")
print("map saved to fitness/jubilee_3500_elevation.png")
