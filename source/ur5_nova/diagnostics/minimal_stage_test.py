#!/usr/bin/env python3
"""Minimal Isaac Sim stage load diagnostic.

Run this with Isaac Sim's Python interpreter (recommended) or your current one:
  python diagnostics/minimal_stage_test.py

Environment variables you can set:
  HEADLESS=1                -> Headless mode
  TEST_STAGE=/path/to.usd   -> Override stage path (optional)
  FRAMES=240                -> Frames to update before exit
  OPEN_STAGE=1              -> 1 to open stage (default), 0 to skip

This script purposefully avoids ROS, policy loading, finetuning, extra extensions.
If this crashes, problem is very low-level (install, driver, or USD asset).
If this passes, higher-level logic in main script is the culprit.
"""

import os, time, sys, traceback

HEADLESS = os.getenv("HEADLESS","0").lower() in ("1","true","yes")
STAGE_PATH = os.getenv("TEST_STAGE", "/home/beable/ur5_simulation/pushT.usd")
FRAMES = int(os.getenv("FRAMES","240"))
OPEN_STAGE = os.getenv("OPEN_STAGE","1").lower() in ("1","true","yes")

try:
    from isaacsim import SimulationApp
except Exception as e:
    print(f"[FATAL] Cannot import SimulationApp: {e}")
    sys.exit(2)

print(f"[DIAG] Starting SimulationApp headless={HEADLESS}")
simulation_app = SimulationApp({"headless": HEADLESS})

try:
    import omni.kit.app
    app = omni.kit.app.get_app()
    ver = getattr(app, 'get_version', lambda: 'unknown')()
    print(f"[DIAG] Isaac Sim version: {ver}")
except Exception:
    print("[DIAG] Could not query app version.")

if OPEN_STAGE:
    try:
        import omni.usd
        ctx = omni.usd.get_context()
        print(f"[DIAG] Opening stage: {STAGE_PATH}")
        ctx.open_stage(STAGE_PATH)
    except Exception as e:
        print(f"[DIAG] Stage open error: {e}")
else:
    print("[DIAG] Skipping stage open (OPEN_STAGE=0).")

print(f"[DIAG] Updating {FRAMES} frames...")
for i in range(FRAMES):
    simulation_app.update()
    if i % 60 == 0:
        print(f"[DIAG] Frame {i}")

print("[DIAG] Completed frames without crash.")

try:
    simulation_app.close()
except Exception:
    traceback.print_exc()
print("[DIAG] Closed cleanly.")
