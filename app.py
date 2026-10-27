import os
import math
import tempfile
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    import cv2
except Exception:
    cv2 = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


st.set_page_config(
    page_title="Field Mission Autonomy Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

WORLD_W, WORLD_H = 10.0, 6.0
VIDEO_W, VIDEO_H = 480, 270

SCALE = min(VIDEO_W / WORLD_W, VIDEO_H / WORLD_H)
OFFSET_X = (VIDEO_W - WORLD_W * SCALE) / 2.0
OFFSET_Y = (VIDEO_H - WORLD_H * SCALE) / 2.0

AGENT_R = 0.18
OBJ_HALF = 0.22
DT = 0.07

DEFAULT_FPS = 20
DEFAULT_SCENE_FRAMES = 32

NAV_WORLD_SCENES = {
    "navigation",
    "reward",
    "imitation",
    "curriculum",
    "regression",
    "domain",
    "safety",
    "monitor",
}

REWARD_SENSITIVE_SCENES = {
    "navigation",
    "reward",
    "imitation",
    "curriculum",
    "regression",
    "domain",
    "safety",
    "monitor",
}

REWARD_BEHAVIOR = {
    "sparse": {
        "noise_mult": 2.0,
        "goal_gain_mult": 0.85,
        "avoidance_mult": 0.90,
        "safety_mult": 1.0,
    },
    "shaped": {
        "noise_mult": 1.0,
        "goal_gain_mult": 1.0,
        "avoidance_mult": 1.0,
        "safety_mult": 1.0,
    },
    "dense": {
        "noise_mult": 0.70,
        "goal_gain_mult": 1.18,
        "avoidance_mult": 0.70,
        "safety_mult": 0.60,
    },
}


try:
    _font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"), fallback_to_default=True)
    FONT = ImageFont.truetype(_font_path, 12)
    FONT_SMALL = ImageFont.truetype(_font_path, 10)
    FONT_TINY = ImageFont.truetype(_font_path, 9)
except Exception:
    FONT = ImageFont.load_default()
    FONT_SMALL = FONT
    FONT_TINY = FONT


SCENES = [
    {
        "key": "vision",
        "title": "Vision-Based Navigation Policy",
        "sim": "navigation",
        "seed_offset": 0,
        "desc": "Train and evaluate an image-style navigation policy in a repeatable simulated corridor with measurable success, path length, and collision rate.",
        "overrides": {},
    },
    {
        "key": "pushing",
        "title": "Object-Pushing Skill Learner",
        "sim": "pushing",
        "seed_offset": 100,
        "desc": "Learn a tabletop pushing skill by coupling agent motion to object motion, with reward shaping toward a target region.",
        "overrides": {},
    },
    {
        "key": "reward",
        "title": "Reward Design Comparison Lab",
        "sim": "navigation",
        "seed_offset": 200,
        "desc": "Compare sparse, shaped, and dense rewards. Detect unintended shortcuts by monitoring clearance, violations, and behavior class.",
        "overrides": {"reward_variant": "shaped"},
    },
    {
        "key": "imitation",
        "title": "Imitation Learning from Demonstrations",
        "sim": "imitation",
        "seed_offset": 300,
        "desc": "Capture scripted demonstrations, normalize observations/actions, train a behavior-cloning baseline, and validate on held-out demos.",
        "overrides": {"reward_variant": "shaped"},
    },
    {
        "key": "safety",
        "title": "Safe Exploration Constraint Layer",
        "sim": "safety",
        "seed_offset": 400,
        "desc": "Command intentionally unsafe actions and show an action shield projecting them into the nearest safe tangent while logging violations.",
        "overrides": {"reward_variant": "shaped"},
    },
    {
        "key": "domain",
        "title": "Sim-to-Sim Domain Generalization",
        "sim": "domain",
        "seed_offset": 500,
        "desc": "Evaluate the same policy under randomized lighting, friction, and obstacle jitter to estimate generalization gap and worst-case performance.",
        "overrides": {"reward_variant": "shaped"},
    },
    {
        "key": "monitor",
        "title": "Adaptive Controller Performance Monitor",
        "sim": "monitor",
        "seed_offset": 600,
        "desc": "Inject disturbances, detect residual error against a reference model, and switch to a conservative fallback controller.",
        "overrides": {"reward_variant": "shaped"},
    },
    {
        "key": "multi",
        "title": "Multi-Agent Cooperative Navigation",
        "sim": "multi",
        "seed_offset": 700,
        "desc": "Coordinate multiple agents with crossed goals, local avoidance, conflict detection, and throughput measurement.",
        "overrides": {"reward_variant": "shaped"},
    },
    {
        "key": "curriculum",
        "title": "Skill Curriculum Generator",
        "sim": "curriculum",
        "seed_offset": 800,
        "desc": "Generate progressively harder tasks from measured performance and recommend the next curriculum level using a mastery threshold.",
        "overrides": {"reward_variant": "shaped", "use_curriculum": True},
    },
    {
        "key": "regression",
        "title": "Policy Regression and Reproducibility Suite",
        "sim": "regression",
        "seed_offset": 900,
        "desc": "Compare baseline and candidate policy versions on fixed seeded scenarios and enforce release gates for success, collision, and violations.",
        "overrides": {"reward_variant": "shaped", "policy_version": "candidate"},
    },
]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def clip_speed(v, max_speed):
    v = np.asarray(v, dtype=float)
    s = norm(v)
    if s <= max_speed or s < 1e-9:
        return v
    return v / s * max_speed


def w2p(pt):
    x, y = float(pt[0]), float(pt[1])
    return (
        int(round(OFFSET_X + x * SCALE)),
        int(round(OFFSET_Y + (WORLD_H - y) * SCALE)),
    )


def color_scale(rgb, lighting=1.0):
    return tuple(int(max(0, min(255, channel * lighting))) for channel in rgb)


def hash_params(p):
    return hashlib.md5(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()


def set_progress(progress_obj, value, text=None):
    value = float(max(0.0, min(1.0, value)))
    try:
        if text is None:
            progress_obj.progress(value)
        else:
            progress_obj.progress(value, text=text)
    except TypeError:
        progress_obj.progress(value)


def fmt_pct(v):
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.0%}"
    except Exception:
        return "—"


def fmt_num(v, digits=2):
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "—"


def get_or_compute(key, producer):
    if key in st.session_state and st.session_state[key] is not None:
        return st.session_state[key]
    try:
        value = producer()
        st.session_state[key] = value
        return value
    except Exception as exc:
        st.warning(f"Could not compute optional data for `{key}`: {exc}")
        return None


# ----------------------------------------------------------------------------
# World and state initialization
# ----------------------------------------------------------------------------

def make_world(seed, scene, p):
    rng = np.random.default_rng(seed)

    dr = float(p.get("domain_randomization", 0.0)) if scene == "domain" else 0.0
    friction = float(np.clip(1.0 + dr * rng.uniform(-0.35, 0.35), 0.55, 1.45))
    lighting = float(np.clip(1.0 + dr * rng.uniform(-0.35, 0.25), 0.60, 1.25))
    jitter = dr * 0.22

    def jit(x, y, r):
        xx = float(x + rng.normal(0.0, jitter))
        yy = float(y + rng.normal(0.0, jitter))
        rr = float(max(0.18, r * (1.0 + 0.08 * rng.normal(0.0, jitter))))
        return xx, yy, rr

    if scene == "pushing":
        obstacles = [jit(5.0, 1.35, 0.45), jit(5.0, 4.25, 0.45)]
        return {
            "scene": scene,
            "start": (1.0, 2.8),
            "object": (2.9, 2.8),
            "target": (8.0, 2.8),
            "target_r": 0.45,
            "obstacles": obstacles,
            "friction": friction,
            "lighting": lighting,
        }

    if scene == "multi":
        n = int(np.clip(p.get("agent_count", 3), 2, 5))
        ys_by_n = {
            2: [1.6, 4.4],
            3: [0.8, 2.8, 4.8],
            4: [0.7, 2.1, 3.5, 4.9],
            5: [0.55, 1.65, 2.8, 3.95, 5.05],
        }
        ys = ys_by_n.get(n, ys_by_n[3])[:n]
        starts = [(0.8, y) for y in ys]
        goals = [(9.2, ys[-1 - i]) for i in range(n)]
        obstacles = [
            jit(4.4, 1.9, 0.42),
            jit(4.4, 3.9, 0.42),
            jit(6.6, 2.9, 0.38),
        ]
        return {
            "scene": scene,
            "start": starts[0],
            "agents": list(zip(starts, goals)),
            "goal_r": 0.32,
            "obstacles": obstacles,
            "friction": friction,
            "lighting": lighting,
        }

    # Navigation-like worlds.
    start = (0.8, 2.8)
    goal = (9.2, 2.8)
    goal_r = 0.35
    obstacles = [
        (3.7, 1.55, 0.55),
        (3.7, 4.05, 0.55),
        (6.1, 2.8, 0.45),
    ]

    if scene == "safety":
        start = (1.2, 2.8)
        goal = (7.6, 2.8)
        obstacles = [(4.3, 2.8, 0.95)]

    elif scene == "curriculum":
        level = int(np.clip(p.get("curriculum_level", 3), 1, 5))
        obstacles = [(3.4, 1.45, 0.48), (3.4, 4.15, 0.48)]
        count = level + 2
        for i in range(count):
            x = 2.4 + 6.6 * (i + 0.5) / count
            y = 2.8 + rng.uniform(-1.7, 1.7)
            if abs(y - 2.8) < 0.75:
                y += 1.15 * (1.0 if y >= 2.8 else -1.0)
            r = rng.uniform(0.32, 0.52)
            obstacles.append((x, y, r))

    elif scene == "regression" and p.get("policy_version") == "candidate":
        obstacles = [(x + 0.13, y + 0.05, r) for x, y, r in obstacles]

    obstacles = [jit(x, y, r) for x, y, r in obstacles]

    clean = []
    start_np = np.array(start, dtype=float)
    goal_np = np.array(goal, dtype=float)
    for x, y, r in obstacles:
        o = np.array([x, y], dtype=float)
        if (
            norm(o - start_np) > r + AGENT_R + 0.35
            and norm(o - goal_np) > r + AGENT_R + 0.35
        ):
            clean.append((x, y, r))
    if clean:
        obstacles = clean

    return {
        "scene": scene,
        "start": start,
        "goal": goal,
        "goal_r": goal_r,
        "obstacles": obstacles,
        "friction": friction,
        "lighting": lighting,
    }


def init_state(scene, world, p):
    if scene == "multi":
        if world.get("agents"):
            start_pos = np.array(world["agents"][0][0], dtype=float)
        else:
            start_pos = np.array(world.get("start", (0.8, 2.8)), dtype=float)
    else:
        start_pos = np.array(world["start"], dtype=float)

    s = {
        "scene": scene,
        "t": 0.0,
        "pos": start_pos,
        "vel": np.zeros(2, dtype=float),
        "history": [],
        "collision": False,
        "success": False,
        "violations": 0,
        "step_violated": False,
        "reward": 0.0,
        "path_length": 0.0,
        "done": False,
        "min_clearance": 10.0,
        "friction": float(world.get("friction", 1.0)),
        "lighting": float(world.get("lighting", 1.0)),
        "speed_scale": float(p.get("speed_scale", 1.0)),
    }
    s["history"].append(s["pos"].copy())

    if scene in NAV_WORLD_SCENES:
        s["prev_goal_dist"] = norm(s["pos"] - np.array(world["goal"], dtype=float))

    if scene == "pushing":
        s["obj_pos"] = np.array(world["object"], dtype=float)
        s["obj_vel"] = np.zeros(2, dtype=float)
        s["obj_history"] = [s["obj_pos"].copy()]
        s["prev_obj_dist"] = norm(s["obj_pos"] - np.array(world["target"], dtype=float))

    if scene == "multi":
        s["agents"] = []
        for idx, (sp, gp) in enumerate(world["agents"]):
            sp = np.array(sp, dtype=float)
            gp = np.array(gp, dtype=float)
            s["agents"].append(
                {
                    "id": idx,
                    "pos": sp.copy(),
                    "goal": gp.copy(),
                    "vel": np.zeros(2, dtype=float),
                    "history": [sp.copy()],
                    "done": False,
                }
            )
        s["completed_agents"] = 0

    if scene == "monitor":
        s["ref_pos"] = s["pos"].copy()
        s["disturbance"] = np.zeros(2, dtype=float)
        s["disturbance_applied"] = False
        s["disturbance_time"] = None
        s["error_ewma"] = 0.0
        s["fallback"] = False
        s["fallback_ever"] = False
        s["false_switch"] = False
        s["detection_delay"] = None
        s["recovery_count"] = 0
        s["switch_events"] = []
        s["series"] = []

    return s


# ----------------------------------------------------------------------------
# Controllers, shielding, and rewards
# ----------------------------------------------------------------------------

def effective_params(scene, params):
    p = params.copy()

    p.setdefault("goal_gain", 1.0)
    p.setdefault("avoidance_gain", 1.8)
    p.setdefault("safety_margin", 0.28)
    p.setdefault("noise", 0.02)

    variant = p.get("reward_variant", "shaped")
    b = REWARD_BEHAVIOR.get(variant, {})

    if scene in REWARD_SENSITIVE_SCENES:
        p["noise"] = float(p["noise"]) * float(b.get("noise_mult", 1.0))
        p["goal_gain"] = float(p["goal_gain"]) * float(b.get("goal_gain_mult", 1.0))
        p["avoidance_gain"] = float(p["avoidance_gain"]) * float(b.get("avoidance_mult", 1.0))
        p["safety_margin"] = max(0.03, float(p["safety_margin"]) * float(b.get("safety_mult", 1.0)))

    if p.get("policy_version") == "candidate":
        p["avoidance_gain"] *= 0.82
        p["noise"] *= 1.25
        p["goal_gain"] *= 1.05

    if scene == "imitation":
        p["noise"] *= 0.5

    return p


def seek(pos, target, gain, max_speed):
    d = np.array(target, dtype=float) - np.asarray(pos, dtype=float)
    dist = norm(d)
    if dist < 1e-6:
        return np.zeros(2, dtype=float)
    return normalize(d) * (max_speed * gain)


def repulsion(pos, circles, margin, gain, influence=0.85):
    pos = np.asarray(pos, dtype=float)
    rep = np.zeros(2, dtype=float)
    for c in circles:
        center = np.array(c[:2], dtype=float)
        r = float(c[2])
        d = pos - center
        dist = norm(d)
        min_d = r + AGENT_R + margin
        if dist < min_d + influence:
            if dist < 1e-5:
                angle = math.atan2(pos[1] - WORLD_H / 2.0, pos[0] - WORLD_W / 2.0) + 0.1
                d = np.array([math.cos(angle), math.sin(angle)], dtype=float)
                dist = 1e-3
            strength = gain * ((min_d + influence - dist) / influence) ** 2
            rep += normalize(d) * strength
    return rep


def shield_velocity(desired, pos, circles, margin, dt, max_speed):
    pos = np.asarray(pos, dtype=float)
    v = clip_speed(np.asarray(desired, dtype=float).copy(), max_speed)

    def unsafe(vel):
        nxt = pos + np.asarray(vel, dtype=float) * dt
        for c in circles:
            center = np.array(c[:2], dtype=float)
            r = float(c[2])
            if norm(nxt - center) < r + AGENT_R + margin * 0.5:
                return True
        return False

    violated = bool(unsafe(v))

    for _ in range(3):
        for c in circles:
            center = np.array(c[:2], dtype=float)
            r = float(c[2])
            R = r + AGENT_R + margin
            d = pos - center
            dist = norm(d)
            if dist < 1e-6:
                continue
            n = d / dist
            rhs = (R * R - dist * dist) / (2.0 * dt)
            nv = float(np.dot(n, v))
            if nv < rhs:
                v = v + (rhs - nv) * n
                violated = True
        v = clip_speed(v, max_speed)

    if unsafe(v):
        original = v.copy()
        found = False
        for alpha in np.linspace(1.0, 0.0, 14):
            trial = original * alpha
            if not unsafe(trial):
                v = trial
                found = True
                break
        if not found:
            v = np.zeros(2, dtype=float)
        violated = True

    return v, violated


def check_collision(pos, circles):
    pos = np.asarray(pos, dtype=float)
    for c in circles:
        center = np.array(c[:2], dtype=float)
        r = float(c[2])
        if norm(pos - center) < r + AGENT_R - 1e-4:
            return True
    return False


def check_object_collision(pos, circles):
    pos = np.asarray(pos, dtype=float)
    for c in circles:
        center = np.array(c[:2], dtype=float)
        r = float(c[2])
        if norm(pos - center) < r + OBJ_HALF - 1e-4:
            return True
    return False


def navigation_reward(state, world, p, prev_d, d):
    r = 0.0
    variant = p.get("reward_variant", "shaped")

    if state.get("success", False):
        r += 10.0
    if state.get("collision", False):
        r -= 8.0
    if state.get("step_violated", False):
        r -= 0.35

    vel = state.get("vel", np.zeros(2, dtype=float))

    if variant == "sparse":
        pass
    elif variant == "shaped":
        r += 2.0 * (prev_d - d) - 0.04 * d * DT
    else:
        r += 3.0 * (prev_d - d) + 0.25 * (1.0 if d < 0.9 else 0.0)
        r -= 0.02 * norm(vel) * DT

    r -= 0.01 * norm(vel) * DT
    return float(r)


def pushing_reward(state, world, p, prev_d, d):
    r = 0.0
    variant = p.get("reward_variant", "shaped")

    if state.get("success", False):
        r += 10.0
    if state.get("collision", False):
        r -= 8.0
    if state.get("step_violated", False):
        r -= 0.35

    vel = state.get("vel", np.zeros(2, dtype=float))

    if variant == "sparse":
        pass
    elif variant == "shaped":
        r += 2.5 * (prev_d - d) - 0.05 * d * DT
    else:
        r += 3.5 * (prev_d - d) - 0.03 * norm(vel) * DT

    r -= 0.01 * norm(vel) * DT
    return float(r)


def step_navigation(state, world, p, desired_override=None, unsafe_mode=False):
    pos = np.asarray(state["pos"], dtype=float)
    goal = np.array(world["goal"], dtype=float)
    max_speed = float(p["max_speed"]) * float(state.get("speed_scale", 1.0))
    state["step_violated"] = False

    if desired_override is not None:
        desired = np.asarray(desired_override, dtype=float)
    elif unsafe_mode:
        desired = normalize(goal - pos) * max_speed * 1.65
        tangent = np.array([-desired[1], desired[0]], dtype=float)
        if norm(tangent) < 1e-6:
            tangent = np.array([0.0, 1.0], dtype=float)
        desired += normalize(tangent) * max_speed * 0.38
    else:
        desired = seek(pos, goal, p.get("goal_gain", 1.0), max_speed)
        desired += repulsion(
            pos,
            world["obstacles"],
            p.get("safety_margin", 0.28),
            p.get("avoidance_gain", 1.8),
        )

    if state.get("disturbance") is not None:
        desired = desired + np.asarray(state["disturbance"], dtype=float)

    safe_v, violated = shield_velocity(
        desired,
        pos,
        world["obstacles"],
        p.get("safety_margin", 0.28),
        DT,
        max_speed,
    )

    if violated:
        state["violations"] += 1
        state["step_violated"] = True

    state["last_action"] = safe_v.copy()

    old = pos.copy()
    noise = state["rng"].normal(0.0, p.get("noise", 0.02), 2) * DT
    new = old + safe_v * state.get("friction", 1.0) * DT + noise

    state["pos"] = new
    state["vel"] = safe_v.copy()
    state["path_length"] += norm(new - old)
    state["history"].append(new.copy())
    if len(state["history"]) > 90:
        state["history"].pop(0)

    d = norm(new - goal)
    prev = state.get("prev_goal_dist", d)

    if d < float(world.get("goal_r", 0.35)):
        state["success"] = True
        state["done"] = True

    if check_collision(new, world["obstacles"]):
        state["collision"] = True
        state["done"] = True

    state["reward"] += navigation_reward(state, world, p, prev, d)
    state["prev_goal_dist"] = d


def step_pushing(state, world, p):
    pos = np.asarray(state["pos"], dtype=float)
    obj = np.asarray(state["obj_pos"], dtype=float)
    target = np.array(world["target"], dtype=float)
    max_speed = float(p["max_speed"]) * float(state.get("speed_scale", 1.0))
    contact = AGENT_R + OBJ_HALF + 0.04

    state["step_violated"] = False
    d_obj = norm(obj - pos)

    if d_obj > contact:
        desired = seek(pos, obj, p.get("goal_gain", 1.0), max_speed)
        desired += repulsion(
            pos,
            world["obstacles"],
            p.get("safety_margin", 0.28),
            p.get("avoidance_gain", 1.8),
        )
    else:
        desired = seek(pos, target, p.get("push_gain", 1.1), max_speed)
        desired += 0.35 * repulsion(
            pos,
            world["obstacles"],
            p.get("safety_margin", 0.28),
            p.get("avoidance_gain", 1.8),
        )

    safe_v, violated = shield_velocity(
        desired,
        pos,
        world["obstacles"],
        p.get("safety_margin", 0.28),
        DT,
        max_speed,
    )

    if violated:
        state["violations"] += 1
        state["step_violated"] = True

    old = pos.copy()
    noise = state["rng"].normal(0.0, p.get("noise", 0.02), 2) * DT
    new = old + safe_v * state.get("friction", 1.0) * DT + noise

    state["pos"] = new
    state["vel"] = safe_v.copy()
    state["path_length"] += norm(new - old)
    state["history"].append(new.copy())
    if len(state["history"]) > 90:
        state["history"].pop(0)

    if norm(new - obj) <= contact + 0.10:
        push_vel = (new - old) / DT
        transfer = float(p.get("push_transfer", 0.85))
        obj_next = obj + push_vel * transfer * state.get("friction", 1.0) * DT

        if check_object_collision(obj_next, world["obstacles"]):
            chosen = obj.copy()
            for alpha in np.linspace(1.0, 0.0, 10):
                trial = obj + push_vel * transfer * state.get("friction", 1.0) * DT * alpha
                if not check_object_collision(trial, world["obstacles"]):
                    chosen = trial
                    break
            obj_next = chosen

        state["obj_pos"] = obj_next
        state["obj_history"].append(obj_next.copy())
        if len(state["obj_history"]) > 90:
            state["obj_history"].pop(0)

    obj_d = norm(state["obj_pos"] - target)
    prev = state.get("prev_obj_dist", obj_d)

    if obj_d < float(world.get("target_r", 0.45)):
        state["success"] = True
        state["done"] = True

    if (
        check_collision(state["pos"], world["obstacles"])
        or check_object_collision(state["obj_pos"], world["obstacles"])
    ):
        state["collision"] = True
        state["done"] = True

    state["reward"] += pushing_reward(state, world, p, prev, obj_d)
    state["prev_obj_dist"] = obj_d


def step_multi(state, world, p):
    state["step_violated"] = False
    agents = state["agents"]
    obstacles = world["obstacles"]
    max_speed = float(p["max_speed"]) * float(state.get("speed_scale", 1.0))

    for a in agents:
        if a["done"]:
            continue

        others = [
            (float(b["pos"][0]), float(b["pos"][1]), AGENT_R)
            for b in agents
            if b is not a
        ]

        desired = seek(a["pos"], a["goal"], p.get("goal_gain", 1.0), max_speed)
        desired += repulsion(
            a["pos"],
            obstacles,
            p.get("safety_margin", 0.28),
            p.get("avoidance_gain", 1.8),
        )
        desired += repulsion(
            a["pos"],
            others,
            p.get("safety_margin", 0.28),
            p.get("avoidance_gain", 1.8) * 1.25,
        )

        circles = list(obstacles) + others
        safe_v, violated = shield_velocity(
            desired,
            a["pos"],
            circles,
            p.get("safety_margin", 0.28),
            DT,
            max_speed,
        )

        if violated:
            state["violations"] += 1
            state["step_violated"] = True

        old = a["pos"].copy()
        noise = state["rng"].normal(0.0, p.get("noise", 0.02), 2) * DT
        new = old + safe_v * state.get("friction", world.get("friction", 1.0)) * DT + noise

        a["pos"] = new
        a["vel"] = safe_v.copy()
        a["history"].append(new.copy())
        if len(a["history"]) > 80:
            a["history"].pop(0)

        state["path_length"] += norm(new - old)

        if norm(new - np.array(a["goal"], dtype=float)) < float(world.get("goal_r", 0.32)):
            a["done"] = True

    for a in agents:
        if check_collision(a["pos"], obstacles):
            state["collision"] = True
            state["done"] = True

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if norm(agents[i]["pos"] - agents[j]["pos"]) < 2.0 * AGENT_R - 1e-4:
                state["collision"] = True
                state["done"] = True

    completed = sum(1 for a in agents if a["done"])
    state["completed_agents"] = completed

    if completed == len(agents) and not state["collision"]:
        state["success"] = True
        state["done"] = True

    state["reward"] += (
        (10.0 if state["success"] else 0.0)
        - (8.0 if state["collision"] else 0.0)
        - (0.25 if state["step_violated"] else 0.0)
        + 0.08 * completed * DT
    )


def step_monitor(state, world, p):
    state["step_violated"] = False
    goal = np.array(world["goal"], dtype=float)
    max_speed = float(p["max_speed"]) * float(state.get("speed_scale", 1.0))

    # Reference model: nominal controller without disturbance.
    ref = np.asarray(state["ref_pos"], dtype=float)
    ref_desired = seek(ref, goal, p.get("goal_gain", 1.0), max_speed)
    ref_desired += repulsion(
        ref,
        world["obstacles"],
        p.get("safety_margin", 0.28),
        p.get("avoidance_gain", 1.8),
    )
    ref_v, _ = shield_velocity(
        ref_desired,
        ref,
        world["obstacles"],
        p.get("safety_margin", 0.28),
        DT,
        max_speed,
    )
    state["ref_pos"] = ref + ref_v * DT

    # Inject disturbance once.
    if not state["disturbance_applied"] and state["t"] >= float(p.get("disturbance_start", 0.8)):
        state["disturbance"] = state["rng"].normal(0.0, float(p.get("disturbance_mag", 0.55)), 2)
        state["disturbance_applied"] = True
        state["disturbance_time"] = state["t"]

    desired = seek(state["pos"], goal, p.get("goal_gain", 1.0), max_speed)
    desired += repulsion(
        state["pos"],
        world["obstacles"],
        p.get("safety_margin", 0.28),
        p.get("avoidance_gain", 1.8),
    )

    if state["disturbance_applied"]:
        desired = desired + np.asarray(state["disturbance"], dtype=float)

    if state["fallback"]:
        desired *= float(p.get("fallback_gain", 0.55))
        desired += repulsion(
            state["pos"],
            world["obstacles"],
            p.get("safety_margin", 0.28),
            p.get("avoidance_gain", 1.8) * 1.6,
        )

    safe_v, violated = shield_velocity(
        desired,
        state["pos"],
        world["obstacles"],
        p.get("safety_margin", 0.28),
        DT,
        max_speed,
    )

    if violated:
        state["violations"] += 1
        state["step_violated"] = True

    old = state["pos"].copy()
    noise = state["rng"].normal(0.0, p.get("noise", 0.02), 2) * DT
    new = old + safe_v * state.get("friction", 1.0) * DT + noise

    state["pos"] = new
    state["vel"] = safe_v.copy()
    state["path_length"] += norm(new - old)
    state["history"].append(new.copy())
    if len(state["history"]) > 90:
        state["history"].pop(0)

    error = norm(new - state["ref_pos"])
    beta = float(p.get("monitor_beta", 0.75))
    state["error_ewma"] = beta * state["error_ewma"] + (1.0 - beta) * error

    threshold = float(p.get("monitor_threshold", 0.22))

    if not state["fallback"]:
        if state["error_ewma"] > threshold:
            state["fallback"] = True
            state["fallback_ever"] = True
            if state["disturbance_applied"]:
                state["detection_delay"] = state["t"] - state["disturbance_time"]
            else:
                state["false_switch"] = True
            state["switch_events"].append(("engage", state["t"]))
            state["recovery_count"] = 0
    else:
        if state["error_ewma"] < threshold * 0.65:
            state["recovery_count"] += 1
            if state["recovery_count"] >= 8:
                state["fallback"] = False
                state["switch_events"].append(("recover", state["t"]))
        else:
            state["recovery_count"] = 0

    state["series"].append(
        {
            "t": float(state["t"]),
            "error": float(error),
            "ewma": float(state["error_ewma"]),
            "fallback": bool(state["fallback"]),
            "disturbed": bool(state["disturbance_applied"]),
        }
    )

    d = norm(new - goal)
    prev = state.get("prev_goal_dist", d)

    if d < float(world.get("goal_r", 0.35)):
        state["success"] = True
        state["done"] = True

    if check_collision(new, world["obstacles"]):
        state["collision"] = True
        state["done"] = True

    state["reward"] += navigation_reward(state, world, p, prev, d)
    state["prev_goal_dist"] = d


def update_clearance(state, world, p):
    mc = float(state.get("min_clearance", 10.0))

    if state["scene"] == "multi":
        for a in state["agents"]:
            for o in world["obstacles"]:
                c = norm(a["pos"] - np.array(o[:2], dtype=float)) - float(o[2]) - AGENT_R
                if c < mc:
                    mc = c
    else:
        for o in world["obstacles"]:
            c = norm(state["pos"] - np.array(o[:2], dtype=float)) - float(o[2]) - AGENT_R
            if c < mc:
                mc = c

        if state["scene"] == "pushing":
            for o in world["obstacles"]:
                c = norm(state["obj_pos"] - np.array(o[:2], dtype=float)) - float(o[2]) - OBJ_HALF
                if c < mc:
                    mc = c

    state["min_clearance"] = mc


def summarize_episode(state, world, p, include_series=False):
    mc = state.get("min_clearance", 0.0)
    if not np.isfinite(mc):
        mc = 0.0

    m = {
        "success": bool(state.get("success", False)),
        "collision": bool(state.get("collision", False)),
        "violations": int(state.get("violations", 0)),
        "path_length": float(state.get("path_length", 0.0)),
        "reward": float(state.get("reward", 0.0)),
        "time": float(state.get("t", 0.0)),
        "min_clearance": float(mc),
    }

    if state["scene"] == "pushing":
        m["object_distance"] = float(norm(state["obj_pos"] - np.array(world["target"], dtype=float)))

    if state["scene"] == "multi":
        completed = int(sum(1 for a in state["agents"] if a["done"]))
        m["completed_agents"] = completed
        m["throughput"] = float(completed / max(state["t"], 1e-6))

    if state["scene"] == "monitor":
        m["detection_delay"] = state.get("detection_delay")
        m["fallback_ever"] = bool(state.get("fallback_ever", False))
        m["false_switch"] = bool(state.get("false_switch", False))
        if include_series:
            m["series"] = state.get("series", [])

    return m


def run_episode(scene, seed, params, max_steps=130, render=False, title=None, frame_total=None, include_series=False):
    p = effective_params(scene, params)
    world = make_world(seed, scene, p)
    state = init_state(scene, world, p)
    state["rng"] = np.random.default_rng(int(seed) + 104729)

    frames = []
    total = int(frame_total or max_steps)

    for step in range(int(max_steps)):
        if scene == "pushing":
            step_pushing(state, world, p)
        elif scene == "multi":
            step_multi(state, world, p)
        elif scene == "monitor":
            step_monitor(state, world, p)
        else:
            step_navigation(state, world, p, unsafe_mode=(scene == "safety"))

        update_clearance(state, world, p)
        state["t"] += DT

        if render:
            rgb = render_frame(state, world, title or scene, step, total, p)
            frames.append(rgb)

        if state["done"]:
            break

    if render:
        if not frames:
            frames.append(render_frame(state, world, title or scene, 0, total, p))
        while len(frames) < total:
            frames.append(frames[-1].copy())
        if len(frames) > total:
            frames = frames[:total]

    m = summarize_episode(state, world, p, include_series=include_series)
    m["scene"] = scene
    m["seed"] = int(seed)
    return m, frames


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def draw_goal(draw, goal, r, cfunc, small=False):
    gx, gy = w2p(goal)
    rr = float(r) * SCALE
    if small:
        rr *= 0.8

    draw.ellipse(
        [gx - rr, gy - rr, gx + rr, gy + rr],
        outline=cfunc((60, 160, 90)),
        width=2,
    )
    draw.ellipse(
        [gx - rr * 0.55, gy - rr * 0.55, gx + rr * 0.55, gy + rr * 0.55],
        fill=cfunc((120, 210, 150)),
    )
    draw.ellipse(
        [gx - rr * 0.20, gy - rr * 0.20, gx + rr * 0.20, gy + rr * 0.20],
        fill=cfunc((35, 110, 65)),
    )


def render_frame(state, world, title, step, total, p):
    lighting = float(np.clip(state.get("lighting", world.get("lighting", 1.0)), 0.55, 1.30))

    def c(rgb):
        return color_scale(rgb, lighting)

    img = Image.new("RGB", (VIDEO_W, VIDEO_H), c((247, 248, 250)))
    draw = ImageDraw.Draw(img)

    top_left = w2p((0, WORLD_H))
    bottom_right = w2p((WORLD_W, 0))

    draw.rectangle([top_left, bottom_right], outline=c((180, 186, 194)), width=1)

    for x in range(0, int(WORLD_W) + 1):
        px = w2p((x, 0))[0]
        draw.line([(px, top_left[1]), (px, bottom_right[1])], fill=c((226, 230, 235)), width=1)

    for y in range(0, int(WORLD_H) + 1):
        py = w2p((0, y))[1]
        draw.line([(top_left[0], py), (bottom_right[0], py)], fill=c((226, 230, 235)), width=1)

    for ox, oy, orad in world["obstacles"]:
        cx, cy = w2p((ox, oy))
        rr = float(orad) * SCALE
        draw.ellipse(
            [cx - rr, cy - rr, cx + rr, cy + rr],
            fill=c((92, 112, 145)),
            outline=c((45, 60, 85)),
        )

    if world.get("goal") is not None and state["scene"] != "multi":
        draw_goal(draw, world["goal"], world.get("goal_r", 0.35), c)

    if world.get("target") is not None:
        tx, ty = w2p(world["target"])
        tr = float(world.get("target_r", 0.45)) * SCALE
        draw.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], outline=c((60, 160, 90)), width=2)
        draw.text((tx - 16, ty - tr - 14), "TARGET", font=FONT_TINY, fill=c((40, 100, 60)))

    if state["scene"] == "pushing":
        if len(state["obj_history"]) > 1:
            pts = [w2p(pt) for pt in state["obj_history"]]
            draw.line(pts, fill=c((230, 150, 60)), width=2)

        ox, oy = w2p(state["obj_pos"])
        half = OBJ_HALF * SCALE
        draw.rectangle(
            [ox - half, oy - half, ox + half, oy + half],
            fill=c((235, 170, 80)),
            outline=c((150, 90, 30)),
        )

    if state["scene"] == "multi":
        palette = [
            (220, 80, 80),
            (70, 130, 220),
            (80, 180, 120),
            (210, 140, 50),
            (150, 90, 200),
        ]
        for idx, a in enumerate(state["agents"]):
            col = palette[idx % len(palette)]
            if len(a["history"]) > 1:
                pts = [w2p(pt) for pt in a["history"]]
                draw.line(pts, fill=c(col), width=2)

            draw_goal(draw, a["goal"], world.get("goal_r", 0.32), c, small=True)

            cx, cy = w2p(a["pos"])
            rr = AGENT_R * SCALE
            draw.ellipse(
                [cx - rr, cy - rr, cx + rr, cy + rr],
                fill=c(col),
                outline=c((30, 30, 30)),
            )
            draw.text((cx - 4, cy - 5), str(idx + 1), font=FONT_TINY, fill=(255, 255, 255))
    else:
        if len(state["history"]) > 1:
            pts = [w2p(pt) for pt in state["history"]]
            draw.line(pts, fill=c((220, 80, 80)), width=2)

        if state["scene"] in ("safety", "domain", "monitor") or p.get("show_margin", True):
            mx, my = w2p(state["pos"])
            mr = (AGENT_R + float(p.get("safety_margin", 0.28))) * SCALE
            draw.ellipse(
                [mx - mr, my - mr, mx + mr, my + mr],
                outline=c((240, 180, 80)),
                width=1,
            )

        cx, cy = w2p(state["pos"])
        rr = AGENT_R * SCALE
        agent_color = (70, 130, 220) if state.get("fallback") else (220, 80, 80)
        draw.ellipse(
            [cx - rr, cy - rr, cx + rr, cy + rr],
            fill=c(agent_color),
            outline=c((25, 25, 25)),
        )

        vel = state.get("vel", np.zeros(2, dtype=float))
        if norm(vel) > 0.05:
            ex, ey = w2p(state["pos"] + normalize(vel) * 0.45)
            draw.line([(cx, cy), (ex, ey)], fill=c((25, 25, 25)), width=2)

    draw.rectangle([0, 0, VIDEO_W, 24], fill=(255, 255, 255))
    draw.text((6, 5), str(title), font=FONT, fill=(25, 30, 40))
    draw.text((VIDEO_W - 92, 5), f"step {min(int(step) + 1, int(total))}/{int(total)}", font=FONT_SMALL, fill=(80, 80, 80))

    status = []
    if state.get("success"):
        status.append("SUCCESS")
    if state.get("collision"):
        status.append("COLLISION")
    if int(state.get("violations", 0)) > 0:
        status.append(f"shield activations: {int(state.get('violations', 0))}")
    if state.get("fallback"):
        status.append("FALLBACK ACTIVE")
    if state.get("disturbance_applied"):
        status.append("disturbance injected")
    if state["scene"] == "pushing":
        status.append(f"obj-goal {norm(state['obj_pos'] - np.array(world['target'], dtype=float)):.2f}m")
    if state["scene"] == "multi":
        done = sum(1 for a in state["agents"] if a["done"])
        status.append(f"done {done}/{len(state['agents'])}")

    status_text = " | ".join(status) if status else "running"

    draw.rectangle([0, VIDEO_H - 18, VIDEO_W, VIDEO_H], fill=(255, 255, 255))
    draw.text((6, VIDEO_H - 16), status_text[:82], font=FONT_SMALL, fill=(40, 45, 55))

    return np.array(img, dtype=np.uint8)


# ----------------------------------------------------------------------------
# Evaluation harness
# ----------------------------------------------------------------------------

def aggregate_df(df):
    agg = {}
    agg["success_rate"] = float(df["success"].mean())
    agg["collision_rate"] = float(df["collision"].mean())
    agg["violation_rate"] = float((df["violations"] > 0).mean())
    agg["avg_violations"] = float(df["violations"].mean())
    agg["avg_path_length"] = float(df["path_length"].mean())
    agg["avg_reward"] = float(df["reward"].mean())

    clearance = pd.to_numeric(df["min_clearance"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    agg["avg_min_clearance"] = float(clearance.mean()) if clearance.notna().any() else float("nan")

    agg["avg_time"] = float(df["time"].mean())

    succ = df[df["success"]]
    agg["time_to_success"] = float(succ["time"].mean()) if len(succ) else float("nan")

    if "object_distance" in df:
        agg["final_object_distance"] = float(pd.to_numeric(df["object_distance"], errors="coerce").mean())

    if "throughput" in df:
        agg["throughput"] = float(pd.to_numeric(df["throughput"], errors="coerce").mean())

    if "completed_agents" in df:
        agg["completed_agents"] = float(pd.to_numeric(df["completed_agents"], errors="coerce").mean())

    if "detection_delay" in df:
        dd = pd.to_numeric(df["detection_delay"], errors="coerce")
        agg["detection_delay"] = float(dd.mean()) if dd.notna().any() else float("nan")

    if "fallback_ever" in df:
        agg["fallback_rate"] = float(df["fallback_ever"].mean())

    if "false_switch" in df:
        agg["false_switch_rate"] = float(df["false_switch"].mean())

    return agg


def evaluate_sim(scene, params, episodes, seed_offset=0, max_steps=130):
    rows = []
    episodes = max(1, int(episodes))
    for ep in range(episodes):
        seed = int(params["base_seed"]) + ep * 137 + int(seed_offset)
        m, _ = run_episode(scene, seed, params, max_steps=max_steps, render=False, include_series=False)
        m["episode"] = ep
        rows.append(m)
    df = pd.DataFrame(rows)
    return aggregate_df(df), df


@st.cache_data(show_spinner=False)
def evaluate_all_scenes(params):
    rows = []
    details = {}
    episodes = int(params.get("episodes", 10))

    for cfg in SCENES:
        p = params.copy()
        p.update(cfg.get("overrides", {}))
        agg, df = evaluate_sim(cfg["sim"], p, episodes=episodes, seed_offset=cfg["seed_offset"], max_steps=130)
        agg["Project"] = cfg["title"]
        agg["Key"] = cfg["key"]
        rows.append(agg)
        details[cfg["key"]] = {"agg": agg, "df": df, "description": cfg["desc"]}

    return pd.DataFrame(rows), details


# ----------------------------------------------------------------------------
# Imitation learning micro-lab
# ----------------------------------------------------------------------------

def bc_feature_vector(state, world, p):
    pos = np.asarray(state["pos"], dtype=float)
    goal = np.array(world["goal"], dtype=float)
    d = goal - pos
    dist = norm(d)

    if world["obstacles"]:
        nearest = min(world["obstacles"], key=lambda o: norm(pos - np.array(o[:2], dtype=float)) - float(o[2]))
    else:
        nearest = (0.0, 0.0, 0.0)

    od = pos - np.array(nearest[:2], dtype=float)
    odist = max(norm(od) - float(nearest[2]), 0.0)
    vel = state.get("vel", np.zeros(2, dtype=float))

    return np.array(
        [
            pos[0] / WORLD_W,
            pos[1] / WORLD_H,
            d[0],
            d[1],
            dist,
            od[0],
            od[1],
            odist,
            vel[0],
            vel[1],
        ],
        dtype=float,
    )


def collect_bc_dataset(params, n_demos=28, max_steps=70):
    X, Y, ids = [], [], []

    for i in range(int(n_demos)):
        seed = int(params["base_seed"]) + 3141 + i * 37
        p = effective_params("imitation", params)
        p["noise"] = min(float(p["noise"]), 0.012)

        world = make_world(seed, "navigation", p)
        state = init_state("navigation", world, p)
        state["rng"] = np.random.default_rng(seed + 77)

        for _ in range(int(max_steps)):
            pos = np.asarray(state["pos"], dtype=float)
            goal = np.array(world["goal"], dtype=float)
            max_speed = float(p["max_speed"])

            desired = seek(pos, goal, p.get("goal_gain", 1.0), max_speed)
            desired += repulsion(
                pos,
                world["obstacles"],
                p.get("safety_margin", 0.28),
                p.get("avoidance_gain", 1.8),
            )
            safe_v, _ = shield_velocity(
                desired,
                pos,
                world["obstacles"],
                p.get("safety_margin", 0.28),
                DT,
                max_speed,
            )

            X.append(bc_feature_vector(state, world, p))
            Y.append(safe_v)
            ids.append(i)

            old = pos.copy()
            noise = state["rng"].normal(0.0, p.get("noise", 0.02), 2) * DT
            new = old + safe_v * state.get("friction", 1.0) * DT + noise
            state["pos"] = new
            state["vel"] = safe_v.copy()
            state["history"].append(new.copy())

            if check_collision(new, world["obstacles"]):
                break
            if norm(new - goal) < float(world.get("goal_r", 0.35)):
                break

    return np.array(X, dtype=float), np.array(Y, dtype=float), np.array(ids, dtype=int)


def train_ridge_policy(X, Y, lam=1e-2):
    if len(X) == 0:
        return np.zeros((1, 2), dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=float)])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=float)
    try:
        W = np.linalg.solve(A, Xb.T @ Y)
    except Exception:
        W = np.linalg.pinv(Xb) @ Y
    return W


def predict_policy(W, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=float)])
    return Xb @ W


def rollout_bc(W, seed, params, max_steps=90):
    p = effective_params("imitation", params)
    world = make_world(seed, "navigation", p)
    state = init_state("navigation", world, p)
    state["rng"] = np.random.default_rng(int(seed) + 91)

    for _ in range(int(max_steps)):
        x = bc_feature_vector(state, world, p).reshape(1, -1)
        pred = predict_policy(W, x)[0]
        step_navigation(state, world, p, desired_override=pred)
        if state["done"]:
            break

    return summarize_episode(state, world, p, include_series=False)


@st.cache_data(show_spinner=False)
def run_imitation_experiment(params):
    n_demos = 28
    X, Y, ids = collect_bc_dataset(params, n_demos=n_demos, max_steps=70)

    unique_ids = np.unique(ids)
    test_count = max(4, n_demos // 4)
    train_ids = unique_ids[:-test_count]
    test_ids = unique_ids[-test_count:]

    train_mask = np.isin(ids, train_ids)
    test_mask = np.isin(ids, test_ids)

    counts = sorted(
        set(
            np.maximum(
                2,
                (np.linspace(0.15, 1.0, 6) * len(train_ids)).astype(int),
            ).tolist()
        )
    )

    curve_rows = []
    for n in counts:
        n = int(min(n, len(train_ids)))
        subset_ids = train_ids[:n]
        mask = np.isin(ids, subset_ids)
        W = train_ridge_policy(X[mask], Y[mask])

        pred = predict_policy(W, X[test_mask])
        mse = float(np.mean(np.sum((pred - Y[test_mask]) ** 2, axis=1)))

        successes = []
        for tid in test_ids:
            seed = int(params["base_seed"]) + 3141 + int(tid) * 37
            m = rollout_bc(W, seed, params, max_steps=80)
            successes.append(float(m["success"]))

        curve_rows.append(
            {
                "demos": n,
                "success_rate": float(np.mean(successes)),
                "action_mse": mse,
            }
        )

    curve_df = pd.DataFrame(curve_rows)

    W_final = train_ridge_policy(X[train_mask], Y[train_mask])
    final_rows = []
    for tid in test_ids:
        seed = int(params["base_seed"]) + 3141 + int(tid) * 37
        final_rows.append(rollout_bc(W_final, seed, params, max_steps=90))

    final_df = pd.DataFrame(final_rows)
    final_success = float(final_df["success"].mean()) if len(final_df) else float("nan")

    pred_test = predict_policy(W_final, X[test_mask])
    final_mse = float(np.mean(np.sum((pred_test - Y[test_mask]) ** 2, axis=1)))

    de_series = curve_df.loc[curve_df["success_rate"] >= 0.75, "demos"]
    data_efficiency = float(de_series.min()) if len(de_series) else float("nan")

    return {
        "curve": curve_df,
        "final_success": final_success,
        "final_mse": final_mse,
        "data_efficiency_demos": data_efficiency,
        "heldout_episodes": int(len(test_ids)),
        "final_df": final_df,
    }


# ----------------------------------------------------------------------------
# Specialized labs
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_reward_comparison(params):
    rows = []
    episodes = max(8, int(params.get("episodes", 10) * 0.8))

    for variant in ["sparse", "shaped", "dense"]:
        p = params.copy()
        p["reward_variant"] = variant
        agg, _ = evaluate_sim("navigation", p, episodes=episodes, seed_offset=200, max_steps=130)

        success = agg.get("success_rate", float("nan"))
        violation = agg.get("violation_rate", float("nan"))
        clearance = agg.get("avg_min_clearance", float("nan"))

        succ = 0.0 if pd.isna(success) else float(success)
        viol = 0.0 if pd.isna(violation) else float(violation)
        clr = 0.0 if pd.isna(clearance) else float(clearance)

        hack_score = succ - 0.8 * viol - 1.5 * max(0.0, 0.12 - clr)

        if clr < 0.06 and viol > 0.15:
            label = "shortcut / reward hack"
        elif succ > 0.70 and viol < 0.25:
            label = "safe direct"
        elif succ < 0.40:
            label = "underfit / exploratory"
        else:
            label = "conservative"

        rows.append(
            {
                "Variant": variant,
                "Success": success,
                "Collision": agg.get("collision_rate", float("nan")),
                "Violation rate": violation,
                "Avg clearance": clearance,
                "Avg reward": agg.get("avg_reward", float("nan")),
                "Path length": agg.get("avg_path_length", float("nan")),
                "Behavior label": label,
                "Hack score": hack_score,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_curriculum_experiment(params):
    rows = []
    episodes = max(6, int(params.get("episodes", 10) // 2))

    for lvl in range(1, 6):
        p = params.copy()
        p["curriculum_level"] = lvl
        p["use_curriculum"] = True
        agg, _ = evaluate_sim("curriculum", p, episodes=episodes, seed_offset=800 + lvl * 13, max_steps=130)
        rows.append(
            {
                "Level": lvl,
                "Success": agg.get("success_rate", float("nan")),
                "Collision": agg.get("collision_rate", float("nan")),
                "Violation": agg.get("violation_rate", float("nan")),
                "Path length": agg.get("avg_path_length", float("nan")),
                "Reward": agg.get("avg_reward", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    mastery = 0.70
    unmastered = df[df["Success"] < mastery]["Level"]
    next_level = int(unmastered.min()) if len(unmastered) else 5

    return {"table": df, "next_level": next_level, "mastery_threshold": mastery}


@st.cache_data(show_spinner=False)
def run_domain_experiment(params):
    episodes = int(params.get("episodes", 10))

    nominal_p = params.copy()
    nominal_p["domain_randomization"] = 0.0

    domain_p = params.copy()
    domain_p["domain_randomization"] = float(params.get("domain_randomization", 0.65))

    nom_agg, _ = evaluate_sim("navigation", nominal_p, episodes=episodes, seed_offset=500, max_steps=130)
    dom_agg, dom_df = evaluate_sim("domain", domain_p, episodes=episodes, seed_offset=500, max_steps=130)

    n_groups = 5
    group_size = max(1, len(dom_df) // n_groups)
    dom_df = dom_df.copy()
    dom_df["env_group"] = dom_df["episode"] // group_size
    group_success = dom_df.groupby("env_group")["success"].mean().reset_index(name="success")

    worst = float(group_success["success"].min()) if len(group_success) else dom_agg.get("success_rate", float("nan"))
    gap = float(nom_agg.get("success_rate", 0.0) - dom_agg.get("success_rate", 0.0))

    return {
        "nominal": nom_agg,
        "domain": dom_agg,
        "worst_case": worst,
        "gap": gap,
        "group_success": group_success,
    }


@st.cache_data(show_spinner=False)
def run_multi_experiment(params):
    rows = []
    episodes = max(5, int(params.get("episodes", 10) // 2))

    for n in [2, 3, 4, 5]:
        p = params.copy()
        p["agent_count"] = n
        agg, _ = evaluate_sim("multi", p, episodes=episodes, seed_offset=700 + n * 17, max_steps=130)
        rows.append(
            {
                "Agents": n,
                "Success": agg.get("success_rate", float("nan")),
                "Collision": agg.get("collision_rate", float("nan")),
                "Violation": agg.get("violation_rate", float("nan")),
                "Throughput": agg.get("throughput", float("nan")),
                "Completion time": agg.get("time_to_success", float("nan")),
                "Path length": agg.get("avg_path_length", float("nan")),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_regression_experiment(params):
    episodes = int(params.get("episodes", 10))

    base_p = params.copy()
    base_p["policy_version"] = "baseline"

    cand_p = params.copy()
    cand_p["policy_version"] = "candidate"

    base_agg, _ = evaluate_sim("navigation", base_p, episodes=episodes, seed_offset=900, max_steps=130)
    cand_agg, _ = evaluate_sim("navigation", cand_p, episodes=episodes, seed_offset=900, max_steps=130)

    success_drop = float(base_agg.get("success_rate", 0.0) - cand_agg.get("success_rate", 0.0))
    collision_increase = float(cand_agg.get("collision_rate", 0.0) - base_agg.get("collision_rate", 0.0))
    violation_increase = float(cand_agg.get("violation_rate", 0.0) - base_agg.get("violation_rate", 0.0))

    gate = {
        "success_gate": bool(success_drop <= 0.05),
        "collision_gate": bool(collision_increase <= 0.02),
        "violation_gate": bool(violation_increase <= 0.03),
    }
    release = all(gate.values())

    table = pd.DataFrame(
        [
            {
                "Version": "v1 baseline",
                "Success": base_agg.get("success_rate", float("nan")),
                "Collision": base_agg.get("collision_rate", float("nan")),
                "Violation": base_agg.get("violation_rate", float("nan")),
                "Path length": base_agg.get("avg_path_length", float("nan")),
                "Reward": base_agg.get("avg_reward", float("nan")),
            },
            {
                "Version": "v2 candidate",
                "Success": cand_agg.get("success_rate", float("nan")),
                "Collision": cand_agg.get("collision_rate", float("nan")),
                "Violation": cand_agg.get("violation_rate", float("nan")),
                "Path length": cand_agg.get("avg_path_length", float("nan")),
                "Reward": cand_agg.get("avg_reward", float("nan")),
            },
        ]
    )

    return {
        "table": table,
        "gate": gate,
        "release": release,
        "success_drop": success_drop,
        "collision_increase": collision_increase,
        "violation_increase": violation_increase,
    }


@st.cache_data(show_spinner=False)
def run_monitor_series(params):
    seed = int(params.get("base_seed", 7)) + 600
    m, _ = run_episode("monitor", seed, params, max_steps=130, render=False, include_series=True)
    series = pd.DataFrame(m.get("series", []))
    metrics = {k: v for k, v in m.items() if k != "series"}
    return series, metrics


# ----------------------------------------------------------------------------
# Video generation
# ----------------------------------------------------------------------------

def open_video_writer(base_path, fps, size):
    if cv2 is None:
        return None, None

    base, _ = os.path.splitext(base_path)
    for name in ("mp4v", "avc1", "H264"):
        path = base + ".mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*name)
            writer = cv2.VideoWriter(path, fourcc, float(fps), (int(size[0]), int(size[1])))
            if writer.isOpened():
                return writer, path
            writer.release()
        except Exception:
            pass

    return None, None


def generate_demo_video(params, progress_callback=None):
    out_dir = tempfile.mkdtemp(prefix="field_mission_")
    base_path = os.path.join(out_dir, "field_mission_demo.mp4")

    frames_per_scene = int(params.get("scene_frames", DEFAULT_SCENE_FRAMES))
    fps = int(params.get("fps", DEFAULT_FPS))
    total_frames = len(SCENES) * frames_per_scene

    writer, video_path = open_video_writer(base_path, fps, (VIDEO_W, VIDEO_H))
    fallback_frames = [] if writer is None else None
    video_metrics = []
    frame_idx = 0

    for cfg in SCENES:
        p = params.copy()
        p.update(cfg.get("overrides", {}))
        p["speed_scale"] = float(p.get("speed_scale", 1.0)) * 1.12

        seed = int(p["base_seed"]) + int(cfg["seed_offset"])
        m, frames = run_episode(
            cfg["sim"],
            seed,
            p,
            max_steps=frames_per_scene,
            render=True,
            title=cfg["title"],
            frame_total=frames_per_scene,
            include_series=False,
        )

        video_metrics.append(
            {
                "Project": cfg["title"],
                "success": bool(m.get("success", False)),
                "collision": bool(m.get("collision", False)),
                "violations": int(m.get("violations", 0)),
                "path_length": float(m.get("path_length", 0.0)),
                "reward": float(m.get("reward", 0.0)),
                "min_clearance": float(m.get("min_clearance", 0.0)),
            }
        )

        for rgb in frames:
            if writer is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            else:
                fallback_frames.append(rgb)

            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx / max(1, total_frames), f"Rendering {cfg['title']} ({frame_idx}/{total_frames})")

    if writer is not None:
        writer.release()
        return video_path, pd.DataFrame(video_metrics)

    if fallback_frames:
        if imageio is not None:
            try:
                gif_path = os.path.join(out_dir, "field_mission_demo.gif")
                imageio.mimsave(gif_path, fallback_frames, format="GIF")
                return gif_path, pd.DataFrame(video_metrics)
            except Exception:
                pass

        try:
            preview_path = os.path.join(out_dir, "field_mission_demo_preview.png")
            Image.fromarray(fallback_frames[0]).save(preview_path)
            return preview_path, pd.DataFrame(video_metrics)
        except Exception:
            pass

    return None, pd.DataFrame(video_metrics)


# ----------------------------------------------------------------------------
# Streamlit controls and UI
# ----------------------------------------------------------------------------

def get_controls():
    with st.sidebar:
        st.header("🎛️ Mission Controls")

        base_seed = int(st.number_input("Deterministic seed", min_value=0, max_value=100000, value=7, step=1))
        episodes = int(st.slider("Evaluation episodes / condition", min_value=5, max_value=50, value=10, step=1))
        max_speed = float(st.slider("Policy speed", min_value=1.0, max_value=6.0, value=3.2, step=0.1))
        safety_margin = float(st.slider("Safety margin", min_value=0.0, max_value=0.8, value=0.28, step=0.01))
        avoidance_gain = float(st.slider("Avoidance gain", min_value=0.0, max_value=4.0, value=1.8, step=0.1))
        noise = float(st.slider("Action noise", min_value=0.0, max_value=0.15, value=0.02, step=0.005))
        reward_variant = st.selectbox("Reward design", ["shaped", "sparse", "dense"], index=0)
        domain_randomization = float(st.slider("Domain randomization strength", min_value=0.0, max_value=1.0, value=0.65, step=0.05))
        agent_count = int(st.slider("Multi-agent count", min_value=2, max_value=5, value=3, step=1))
        curriculum_level = int(st.slider("Curriculum level", min_value=1, max_value=5, value=3, step=1))
        disturbance_mag = float(st.slider("Disturbance magnitude", min_value=0.0, max_value=1.5, value=0.55, step=0.05))
        monitor_threshold = float(st.slider("Fallback error threshold", min_value=0.05, max_value=0.8, value=0.22, step=0.01))
        scene_frames = int(st.slider("Video frames per scene", min_value=20, max_value=70, value=DEFAULT_SCENE_FRAMES, step=1))
        fps = int(st.slider("Video FPS", min_value=10, max_value=30, value=DEFAULT_FPS, step=1))

        auto_regenerate = st.checkbox("Auto-regenerate when controls change", value=False)
        generate_button = st.button("🔁 Generate / refresh live demo", type="primary", use_container_width=True)

        st.caption("No API key required. Runs locally with deterministic seeds.")

    params = {
        "base_seed": base_seed,
        "episodes": episodes,
        "max_speed": max_speed,
        "safety_margin": safety_margin,
        "avoidance_gain": avoidance_gain,
        "noise": noise,
        "reward_variant": reward_variant,
        "domain_randomization": domain_randomization,
        "agent_count": agent_count,
        "curriculum_level": curriculum_level,
        "disturbance_mag": disturbance_mag,
        "monitor_threshold": monitor_threshold,
        "scene_frames": scene_frames,
        "fps": fps,
        "goal_gain": 1.0,
        "push_gain": 1.1,
        "push_transfer": 0.85,
        "disturbance_start": 0.8,
        "fallback_gain": 0.55,
        "monitor_beta": 0.75,
        "policy_version": "baseline",
        "speed_scale": 1.0,
        "use_curriculum": False,
        "show_margin": True,
    }

    return params, generate_button, auto_regenerate


def main():
    st.title("🤖 Field Mission Autonomy Lab")
    st.caption(
        "A local Streamlit mission harness for repeatable policy evaluation, safety constraints, "
        "reward-design audits, imitation learning, domain generalization, monitoring, multi-agent coordination, "
        "curriculum generation, and regression gates. The app auto-renders a live simulation demo on startup."
    )

    params, generate_button, auto_regenerate = get_controls()
    full_hash = hash_params(params)
    analytics_params = {k: v for k, v in params.items() if k not in ("scene_frames", "fps")}

    default_keys = {
        "generated_full_hash": None,
        "video_path": None,
        "video_metrics": None,
        "overall_data": None,
        "reward_df": None,
        "imitation": None,
        "curriculum": None,
        "domain": None,
        "multi_df": None,
        "regression": None,
        "monitor_data": None,
        "error": None,
    }
    for k, v in default_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

    need_generate = (
        st.session_state.generated_full_hash is None
        or generate_button
        or (auto_regenerate and st.session_state.generated_full_hash != full_hash)
    )

    if need_generate:
        progress = st.progress(0.0, text="Starting local simulation...")
        video_path = None

        try:
            set_progress(progress, 0.03, "Rendering live simulation video...")
            video_path, video_metrics = generate_demo_video(
                params,
                lambda v, t: set_progress(progress, 0.03 + 0.55 * float(v), t),
            )

            st.session_state.video_path = video_path
            st.session_state.video_metrics = video_metrics

            set_progress(progress, 0.62, "Evaluating repeatable policies...")
            overall_data = evaluate_all_scenes(analytics_params)
            st.session_state.overall_data = overall_data

            set_progress(progress, 0.70, "Comparing reward designs...")
            reward_df = run_reward_comparison(analytics_params)
            st.session_state.reward_df = reward_df

            set_progress(progress, 0.76, "Training behavior-cloning baseline...")
            imitation = run_imitation_experiment(analytics_params)
            st.session_state.imitation = imitation

            set_progress(progress, 0.82, "Running curriculum and domain generalization...")
            curriculum = run_curriculum_experiment(analytics_params)
            domain = run_domain_experiment(analytics_params)
            multi_df = run_multi_experiment(analytics_params)
            st.session_state.curriculum = curriculum
            st.session_state.domain = domain
            st.session_state.multi_df = multi_df

            set_progress(progress, 0.91, "Checking regression gates and monitor telemetry...")
            regression = run_regression_experiment(analytics_params)
            monitor_data = run_monitor_series(analytics_params)
            st.session_state.regression = regression
            st.session_state.monitor_data = monitor_data

            st.session_state.error = None
            st.session_state.generated_full_hash = full_hash
            set_progress(progress, 1.0, "Live demo ready.")

        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
            set_progress(progress, 1.0, "Generation stopped with an error. Analytics may still be available below.")

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    if st.session_state.get("generated_full_hash") is not None and st.session_state.generated_full_hash != full_hash:
        st.info("Controls changed. Click **Generate / refresh live demo** to update the video and metrics.")

    video_path = st.session_state.get("video_path")
    video_metrics = st.session_state.get("video_metrics")

    if video_path and os.path.exists(video_path):
        st.subheader("🎥 Live simulation demo")
        col_video, col_telemetry = st.columns([2.1, 1])

        with col_video:
            ext = os.path.splitext(video_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".gif"):
                st.image(video_path, use_container_width=True)
            else:
                try:
                    st.video(video_path, format="video/mp4", loop=True, autoplay=True, muted=True)
                except TypeError:
                    try:
                        st.video(video_path, loop=True, autoplay=True)
                    except TypeError:
                        st.video(video_path)

            with open(video_path, "rb") as f:
                data = f.read()

            mime = {
                ".mp4": "video/mp4",
                ".gif": "image/gif",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }.get(ext, "application/octet-stream")

            st.download_button(
                "⬇️ Download demo file",
                data=data,
                file_name=os.path.basename(video_path),
                mime=mime,
                use_container_width=True,
            )

        with col_telemetry:
            st.markdown("**Rendered episode telemetry**")
            if video_metrics is not None and len(video_metrics) > 0:
                st.metric("Scenes rendered", len(video_metrics))
                st.metric("Successes", int(video_metrics["success"].sum()))
                st.metric("Collisions", int(video_metrics["collision"].sum()))
                st.metric("Shield activations", int(video_metrics["violations"].sum()))
                st.dataframe(
                    video_metrics[
                        ["Project", "success", "collision", "violations", "path_length", "reward"]
                    ]
                )
            else:
                st.write("No telemetry available.")
    else:
        st.warning(
            "Video renderer unavailable on this system. The app still loads deterministic analytics and interactive modules below."
        )

    overall_data = get_or_compute("overall_data", lambda: evaluate_all_scenes(analytics_params))

    if overall_data is None:
        st.warning("Analytics could not be computed.")
        return

    overall_df, scene_details = overall_data

    st.subheader("Measurable success and safety constraints")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Mean success", fmt_pct(overall_df["success_rate"].mean()))
    k2.metric("Mean collision", fmt_pct(overall_df["collision_rate"].mean()))
    k3.metric("Violation rate", fmt_pct(overall_df["violation_rate"].mean()))
    k4.metric("Avg path length", fmt_num(overall_df["avg_path_length"].mean(), 2) + " m")
    k5.metric("Avg reward", fmt_num(overall_df["avg_reward"].mean(), 2))
    k6.metric("Avg clearance", fmt_num(overall_df["avg_min_clearance"].mean(), 3) + " m")

    success_vals = overall_df["success_rate"].fillna(0).astype(float).values
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.38 * len(overall_df) + 1.0)))
    colors = ["#2a7fbf" if v > 0.55 else "#e08b2c" if v > 0.25 else "#c44e52" for v in success_vals]
    ax.barh(overall_df["Project"].values, success_vals, color=colors)
    ax.set_xlim(0, 1.03)
    ax.invert_yaxis()
    ax.set_xlabel("Success rate")
    ax.set_title("Policy success across mission modules")
    for i, v in enumerate(success_vals):
        ax.text(min(v + 0.02, 0.98), i, fmt_pct(v), va="center", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

    tab_list = st.tabs([cfg["title"] for cfg in SCENES])

    for tab, cfg in zip(tab_list, SCENES):
        with tab:
            row = overall_df[overall_df["Key"] == cfg["key"]].iloc[0]

            st.markdown(f"**Goal:** {cfg['desc']}")

            mc = st.columns(5)
            mc[0].metric("Success", fmt_pct(row.get("success_rate")))
            mc[1].metric("Collision", fmt_pct(row.get("collision_rate")))
            mc[2].metric("Violations", fmt_pct(row.get("violation_rate")))
            mc[3].metric("Path length", fmt_num(row.get("avg_path_length"), 2) + " m")
            mc[4].metric("Reward", fmt_num(row.get("avg_reward"), 2))

            if scene_details and cfg["key"] in scene_details:
                df = scene_details[cfg["key"]]["df"]
                preferred_cols = {
                    "vision": ["episode", "success", "collision", "violations", "path_length", "reward", "min_clearance"],
                    "pushing": ["episode", "success", "collision", "object_distance", "path_length", "reward"],
                    "reward": ["episode", "success", "collision", "violations", "min_clearance", "reward"],
                    "imitation": ["episode", "success", "collision", "path_length", "reward", "min_clearance"],
                    "safety": ["episode", "violations", "collision", "success", "min_clearance", "reward"],
                    "domain": ["episode", "success", "collision", "path_length", "reward", "min_clearance"],
                    "monitor": ["episode", "success", "collision", "detection_delay", "fallback_ever", "false_switch"],
                    "multi": ["episode", "success", "collision", "completed_agents", "throughput", "path_length"],
                    "curriculum": ["episode", "success", "collision", "violations", "path_length", "reward"],
                    "regression": ["episode", "success", "collision", "violations", "path_length", "reward"],
                }
                cols = [c for c in preferred_cols.get(cfg["key"], []) if c in df.columns]
                if cols:
                    st.write("Sample seeded episodes")
                    st.dataframe(df[cols].head(8))

            if cfg["key"] == "pushing":
                st.metric("Final object distance", fmt_num(row.get("final_object_distance"), 3) + " m")

            if cfg["key"] == "reward":
                reward_df = get_or_compute("reward_df", lambda: run_reward_comparison(analytics_params))
                if reward_df is not None:
                    st.write("Reward variant comparison")
                    st.dataframe(reward_df)

                    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
                    axes[0].bar(reward_df["Variant"], reward_df["Success"].fillna(0), color="#2a7fbf")
                    axes[0].set_ylim(0, 1)
                    axes[0].set_title("Success")

                    axes[1].bar(reward_df["Variant"], reward_df["Violation rate"].fillna(0), color="#c44e52")
                    axes[1].set_title("Violation rate")

                    axes[2].bar(reward_df["Variant"], reward_df["Avg clearance"].fillna(0), color="#4daf4a")
                    axes[2].set_title("Avg clearance (m)")

                    for ax in axes:
                        ax.tick_params(axis="x", rotation=20)
                        for label in ax.get_xticklabels():
                            label.set_fontsize(8)

                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        "The dense reward can look attractive while producing tighter clearances and more shield activations — "
                        "a typical reward-hacking signal."
                    )

            if cfg["key"] == "imitation":
                imitation = get_or_compute("imitation", lambda: run_imitation_experiment(analytics_params))
                if imitation is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Held-out success", fmt_pct(imitation["final_success"]))
                    c2.metric("Action MSE", fmt_num(imitation["final_mse"], 4))
                    c3.metric(
                        "Demos to 75% success",
                        "—" if pd.isna(imitation["data_efficiency_demos"]) else str(int(imitation["data_efficiency_demos"])),
                    )

                    curve = imitation["curve"]
                    fig, ax1 = plt.subplots(figsize=(6.2, 3.0))
                    ax1.plot(curve["demos"], curve["success_rate"], marker="o", color="#2a7fbf", label="Success")
                    ax1.set_xlabel("Demonstrations used")
                    ax1.set_ylabel("Success rate")
                    ax1.set_ylim(0, 1)

                    ax2 = ax1.twinx()
                    ax2.plot(curve["demos"], curve["action_mse"], marker="s", color="#d95f02", label="Action MSE")
                    ax2.set_ylabel("MSE")

                    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(curve)

            if cfg["key"] == "safety":
                st.markdown(
                    "The shield projects commanded velocities into the nearest safe tangent space. "
                    "Unsafe commands are logged as constraint violations instead of being executed blindly."
                )
                if scene_details and "safety" in scene_details:
                    sdf = scene_details["safety"]["df"]
                    if len(sdf) > 0:
                        fig, ax = plt.subplots(figsize=(6.2, 2.9))
                        colors = ["#c44e52" if bool(c) else "#2a7fbf" for c in sdf["collision"]]
                        ax.scatter(sdf["violations"], sdf["min_clearance"], c=colors, alpha=0.75, edgecolors="none")
                        ax.set_xlabel("Shield activations")
                        ax.set_ylabel("Min clearance (m)")
                        ax.set_title("Violations are logged; collisions remain controlled")
                        st.pyplot(fig)
                        plt.close(fig)

            if cfg["key"] == "domain":
                domain = get_or_compute("domain", lambda: run_domain_experiment(analytics_params))
                if domain is not None:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Nominal success", fmt_pct(domain["nominal"].get("success_rate")))
                    c2.metric("Randomized success", fmt_pct(domain["domain"].get("success_rate")))
                    c3.metric("Worst-case env", fmt_pct(domain["worst_case"]))
                    c4.metric("Generalization gap", fmt_pct(domain["gap"]))

                    vals = [
                        float(np.nan_to_num(domain["nominal"].get("success_rate", 0.0))),
                        float(np.nan_to_num(domain["domain"].get("success_rate", 0.0))),
                        float(np.nan_to_num(domain["worst_case"])),
                    ]
                    fig, ax = plt.subplots(figsize=(5.5, 2.9))
                    ax.bar(["Nominal", "Randomized", "Worst env"], vals, color=["#2a7fbf", "#e08b2c", "#c44e52"])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success rate")
                    for i, v in enumerate(vals):
                        ax.text(i, min(v + 0.03, 0.97), fmt_pct(v), ha="center", fontsize=9)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(domain["group_success"])

            if cfg["key"] == "monitor":
                monitor_data = get_or_compute("monitor_data", lambda: run_monitor_series(analytics_params))
                if monitor_data is not None:
                    monitor_series, monitor_metrics = monitor_data

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Detection delay", fmt_num(monitor_metrics.get("detection_delay"), 2) + " s")
                    c2.metric("Fallback engaged", "Yes" if monitor_metrics.get("fallback_ever") else "No")
                    c3.metric("False switch", "Yes" if monitor_metrics.get("false_switch") else "No")
                    c4.metric("Task success", fmt_pct(monitor_metrics.get("success")))

                    if monitor_series is not None and len(monitor_series) > 0:
                        fig, ax = plt.subplots(figsize=(7.2, 3.1))
                        ax.plot(monitor_series["t"], monitor_series["error"], color="#888888", label="instant error")
                        ax.plot(monitor_series["t"], monitor_series["ewma"], color="#c44e52", label="EWMA residual")
                        ax.axhline(float(params.get("monitor_threshold", 0.22)), ls="--", color="black", label="threshold")

                        ymax = max(0.05, float(monitor_series["ewma"].max()) * 1.15)
                        ax.fill_between(
                            monitor_series["t"],
                            0,
                            ymax,
                            where=monitor_series["fallback"],
                            color="#ffcc80",
                            alpha=0.35,
                            label="fallback active",
                        )

                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Error (m)")
                        ax.set_ylim(0, ymax)
                        ax.legend(fontsize=8)
                        st.pyplot(fig)
                        plt.close(fig)

                        st.dataframe(monitor_series.tail(12))

            if cfg["key"] == "multi":
                multi_df = get_or_compute("multi_df", lambda: run_multi_experiment(analytics_params))
                if multi_df is not None:
                    st.metric("Current agent count", int(params.get("agent_count", 3)))
                    st.dataframe(multi_df)

                    fig, ax = plt.subplots(figsize=(6.2, 2.9))
                    x = np.arange(len(multi_df))
                    width = 0.35
                    ax.bar(
                        x - width / 2,
                        multi_df["Success"].fillna(0),
                        width,
                        label="Success",
                        color="#2a7fbf",
                    )
                    ax.bar(
                        x + width / 2,
                        multi_df["Collision"].fillna(0),
                        width,
                        label="Collision",
                        color="#c44e52",
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(multi_df["Agents"].astype(str))
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Agent count")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

            if cfg["key"] == "curriculum":
                curriculum = get_or_compute("curriculum", lambda: run_curriculum_experiment(analytics_params))
                if curriculum is not None:
                    c1, c2 = st.columns(2)
                    c1.metric("Next recommended level", curriculum["next_level"])
                    c2.metric("Mastery threshold", fmt_pct(curriculum["mastery_threshold"]))

                    st.dataframe(curriculum["table"])

                    tbl = curriculum["table"]
                    fig, ax = plt.subplots(figsize=(6.2, 2.9))
                    ax.plot(tbl["Level"], tbl["Success"].fillna(0), marker="o", color="#2a7fbf")
                    ax.axhline(curriculum["mastery_threshold"], ls="--", color="#4daf4a", label="mastery")
                    ax.set_xlabel("Curriculum level")
                    ax.set_ylabel("Success rate")
                    ax.set_ylim(0, 1)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

            if cfg["key"] == "regression":
                regression = get_or_compute("regression", lambda: run_regression_experiment(analytics_params))
                if regression is not None:
                    st.dataframe(regression["table"])

                    g1, g2, g3 = st.columns(3)
                    g1.metric("Success gate", "PASS" if regression["gate"]["success_gate"] else "FAIL")
                    g2.metric("Collision gate", "PASS" if regression["gate"]["collision_gate"] else "FAIL")
                    g3.metric("Violation gate", "PASS" if regression["gate"]["violation_gate"] else "FAIL")

                    if regression["release"]:
                        st.success("Release gate: PASS — candidate can be promoted.")
                    else:
                        st.error("Release gate: BLOCKED — candidate regresses safety or performance.")

                    st.caption(
                        f"Success drop: {fmt_num(regression['success_drop'], 3)} | "
                        f"Collision increase: {fmt_num(regression['collision_increase'], 3)} | "
                        f"Violation increase: {fmt_num(regression['violation_increase'], 3)}"
                    )


if __name__ == "__main__":
    main()