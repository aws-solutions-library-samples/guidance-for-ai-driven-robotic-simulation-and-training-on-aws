#!/usr/bin/env python3
"""Incremental USD layer opening diagnostic for Isaac Sim.

Purpose: identify which subLayer causes a native crash.

Process:
 1. Opens SimulationApp (headless configurable).
 2. Enumerates sublayers of a target USD (TARGET_STAGE) without fully composing them first (string parse fallback if needed).
 3. Iteratively builds a temporary in-memory stage stacking sublayers one by one; after each addition runs N frames.
 4. If a crash happens between layer k and k+1 you know the last added layer is suspect.

Env Vars:
  TARGET_STAGE=/path/to/root.usd (default pushT.usd)
  HEADLESS=1
  FRAMES_PER_LAYER=90
  MAX_LAYERS=0 (0 = no cap)
  SAFE_RENDER_MODE=1

Run:
  python diagnostics/incremental_layer_open.py
"""
import os, time, sys, traceback

TARGET_STAGE = os.getenv("TARGET_STAGE", "/home/beable/ur5_simulation/pushT.usd")
FRAMES_PER_LAYER = int(os.getenv("FRAMES_PER_LAYER","90"))
MAX_LAYERS = int(os.getenv("MAX_LAYERS","0"))
HEADLESS = os.getenv("HEADLESS","1").lower() in ("1","true","yes")

try:
    from isaacsim import SimulationApp
except Exception as e:
    print(f"[FATAL] SimulationApp import failed: {e}")
    sys.exit(2)

simulation_app = SimulationApp({"headless": HEADLESS})

if os.getenv("SAFE_RENDER_MODE","1").lower() in ("1","true","yes"):
    try:
        import carb.settings
        s = carb.settings.get_settings()
        for k,v in [
            ("/rtx/enabled", False),
            ("/ngx/enabled", False),
            ("/app/renderer/resolution/width", 640),
            ("/app/renderer/resolution/height", 360),
            ("/app/renderer/vsync", False),
            ("/app/hydraEngine/hydraEngineDelegate", "Storm"),
        ]:
            try: s.set(k,v)
            except Exception: pass
        print("[SAFE_RENDER] Applied low settings.")
    except Exception:
        pass

print(f"[INCR] Target root stage: {TARGET_STAGE}")

from pxr import Sdf, Usd

def get_sublayers(path):
    try:
        layer = Sdf.Layer.FindOrOpen(path)
        if not layer:
            print(f"[INCR] Could not open layer: {path}")
            return []
        return list(layer.subLayerPaths)
    except Exception as e:
        print(f"[INCR] sublayer parse error: {e}")
        return []

sub_layers = get_sublayers(TARGET_STAGE)
print(f"[INCR] Found {len(sub_layers)} sublayers in root layer.")
for i,p in enumerate(sub_layers):
    print(f"  [{i}] {p}")

if not sub_layers:
    print("[INCR] No sublayers; testing direct open.")
    try:
        import omni.usd
        ctx = omni.usd.get_context(); ctx.open_stage(TARGET_STAGE)
        for f in range(FRAMES_PER_LAYER):
            simulation_app.update(); time.sleep(0.01)
        print("[INCR] Direct stage stable.")
    except Exception as e:
        print(f"[INCR] Direct stage open error: {e}")
    simulation_app.close(); sys.exit(0)

limit = len(sub_layers) if MAX_LAYERS <= 0 else min(MAX_LAYERS, len(sub_layers))

print(f"[INCR] Incrementally composing first {limit} layers.")

temp_stage = Usd.Stage.CreateInMemory()
root_layer = temp_stage.GetRootLayer()
added = []
for idx in range(limit):
    sl = sub_layers[idx]
    print(f"[INCR] Adding layer {idx}: {sl}")
    try:
        # Append to subLayerPaths and save
        root_layer.subLayerPaths.append(sl)
        added.append(sl)
    except Exception as e:
        print(f"[INCR] Failed to append layer {sl}: {e}")
        break
    # Run a few frames
    for f in range(FRAMES_PER_LAYER):
        simulation_app.update(); time.sleep(0.01)
    print(f"[INCR] Layer {idx} stable for {FRAMES_PER_LAYER} frames.")

print("[INCR] Completed incremental layering without detected crash inside loop.")
try:
    simulation_app.close()
except Exception:
    traceback.print_exc()
print("[INCR] Closed.")
