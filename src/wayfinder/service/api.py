"""FastAPI routing service and static web interface.

    GET  /                    map with the five sliders
    GET  /study               user-study page
    GET  /api/health
    GET  /api/meta            study area, sliders, presets, algorithms
    POST /api/route           personalised route
    GET  /api/study/trials    study trials (routes unlabelled; the page shuffles them per participant)
    POST /api/study/response  record one choice
"""
from __future__ import annotations

import random
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wayfinder import WEIGHT_KEYS
from wayfinder.config import ROOT, load_config
from wayfinder.learning.weights import PROFILES
from wayfinder.service.planner import ALGORITHMS, RoutePlanner
from wayfinder.study.user_study import load_trials, record_response

WEB_DIR = ROOT / "web"


class Point(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Weights(BaseModel):
    time: float = Field(1.0, ge=0, le=1)
    junction: float = Field(0.0, ge=0, le=1)
    traffic: float = Field(0.0, ge=0, le=1)
    canopy: float = Field(0.0, ge=0, le=1)
    climb: float = Field(0.0, ge=0, le=1)


class RouteRequest(BaseModel):
    origin: Point
    destination: Point
    weights: Weights = Weights()
    algorithm: str = "astar"
    W: float = Field(1.5, ge=1.0, le=5.0)
    compare_fastest: bool = True
    verify: bool = False


class StudyResponse(BaseModel):
    participant_id: str = Field(min_length=1, max_length=64)
    trial_id: str = Field(min_length=1, max_length=64)
    choice: str = Field(pattern="^(profile|fastest)$")
    shown_first: str = Field(pattern="^(profile|fastest)$")
    seconds: float | None = None


def create_app(planner: RoutePlanner | None = None, cfg=None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="Routes That Fit You", version="0.1.0")
    state = {"planner": planner}

    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        # browsers must re-check pages and scripts on every load, so an updated app.js is never served stale
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response
    planner_lock, responses_lock = threading.Lock(), threading.Lock()

    def get_planner() -> RoutePlanner:
        if state["planner"] is None:
            with planner_lock:                      # load the graph and models once, even under concurrent requests
                if state["planner"] is None:
                    state["planner"] = RoutePlanner.from_config(cfg)
        return state["planner"]

    study_dir = cfg.path("study_dir")
    responses_dir = cfg.path("results_dir") / "user_study"

    @app.get("/api/health")
    def health():
        p = get_planner()
        return {"status": "ok", "nodes": p.net.rg.n, "edges": p.net.rg.m, "guide": p.guide_name}

    @app.get("/api/meta")
    def meta():
        p = get_planner()
        area = cfg["study_area"]
        return {"study_area": area["name"], "center": [area["center_lat"], area["center_lon"]],
                "bounds": p.net.rg.bounds_latlon(), "sliders": list(WEIGHT_KEYS), "algorithms": list(ALGORITHMS),
                "default_algorithm": cfg["service"]["default_algorithm"], "default_W": cfg["service"]["default_W"],
                "guide": p.guide_name,
                "presets": {"fastest": {"time": 1.0}, **{k: v["weights"] for k, v in PROFILES.items()}}}

    @app.post("/api/route")
    def route(req: RouteRequest):
        if req.algorithm not in ALGORITHMS:
            raise HTTPException(400, f"algorithm must be one of {list(ALGORITHMS)}")
        try:
            return get_planner().route((req.origin.lat, req.origin.lon), (req.destination.lat, req.destination.lon),
                                       req.weights.model_dump(), req.algorithm, req.W, req.compare_fastest,
                                       req.verify)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/study/trials")
    def study_trials():
        trials = load_trials(study_dir)
        if not trials:
            raise HTTPException(404, "no study trials prepared; run `python -m wayfinder study-prepare`")
        out = []
        for t in trials:
            out.append({"trial_id": t["trial_id"], "profile": t["profile"], "scenario": t["scenario"],
                        "origin": t["origin"], "destination": t["destination"],
                        "routes": {k: {"latlon": v["latlon"]} for k, v in t["routes"].items()}})
        random.shuffle(out)
        return {"trials": out}

    @app.post("/api/study/response")
    def study_response(resp: StudyResponse):
        trials = {t["trial_id"]: t for t in load_trials(study_dir)}
        if resp.trial_id not in trials:
            raise HTTPException(404, "unknown trial")
        with responses_lock:                        # one writer at a time on the responses file
            record_response(responses_dir, {**resp.model_dump(), "profile": trials[resp.trial_id]["profile"]})
        return {"status": "recorded"}

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/study")
    def study_page():
        return FileResponse(WEB_DIR / "study.html")

    if Path(WEB_DIR).exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    return app


def serve(cfg=None, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    cfg = cfg or load_config()
    app = create_app(cfg=cfg)
    uvicorn.run(app, host=host or cfg["service"]["host"], port=port or cfg["service"]["port"])
