"""Load the study-area network (preprocessed arrays if present, otherwise the raw GraphML)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wayfinder.config import Config, load_config
from wayfinder.cost import CostModel
from wayfinder.graph.road_graph import RoadGraph
from wayfinder.graph.turn_graph import TurnGraph

PROCESSED_GRAPH = "graph.npz"


@dataclass
class Network:
    rg: RoadGraph
    tg: TurnGraph
    cm: CostModel


def processed_graph_path(cfg: Config) -> Path:
    return cfg.path("processed_dir") / PROCESSED_GRAPH


def build_road_graph(cfg: Config) -> RoadGraph:
    return RoadGraph.from_graphml(cfg.path("raw_graph"), largest_scc=True)


def load_network(cfg: Config | None = None, rg: RoadGraph | None = None) -> Network:
    cfg = cfg or load_config()
    if rg is None:
        p = processed_graph_path(cfg)
        rg = RoadGraph.load(p) if p.exists() else build_road_graph(cfg)
    tg = TurnGraph(rg)
    return Network(rg=rg, tg=tg, cm=CostModel(rg, tg))
