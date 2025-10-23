import time
#!/usr/bin/env python3
"""
Simplified Scene Loader + Extension Enabler

This script intentionally removes ALL RL / finetune / policy logic.
It only:
  1. Starts Isaac Sim (optionally headless)
  2. (Optionally) enables all extensions (or filtered)
  3. Loads the given USD stage
  4. Idles at a target FPS

Environment Variables:
  SCENE_USD_PATH               Path to USD (default /home/beable/ur5_simulation/pushT.usd)
  HEADLESS=1                   Headless mode
  IDLE_FPS=20                  Idle update rate
  AUTO_ENABLE_ALL_EXTENSIONS=1 Enable all extensions
  AUTO_ENABLE_EXT_PREFIX=omni.isaac.  Only enable those starting with prefix
  AUTO_ENABLE_EXT_EXCLUDE=a,b  Comma list to skip
  SAFE_RENDER_MODE=1           Lower render load (Storm delegate etc.)
"""

import os, time, threading
from pathlib import Path


#os.environ['HOME'] + "/ur5_push_T-main/pushT.usd"

USD_PATH = os.getenv("SCENE_USD_PATH", os.environ['HOME'] + "/ur5_push_T-main/pushT.usd")
HEADLESS = os.getenv("HEADLESS", "0").lower() in ("1","true","yes")
IDLE_FPS = float(os.getenv("IDLE_FPS", "25"))
AUTO_ENABLE_ALL_EXT = os.getenv("AUTO_ENABLE_ALL_EXTENSIONS","1").lower() in ("1","true","yes")
SAFE_RENDER = os.getenv("SAFE_RENDER_MODE","1").lower() in ("1","true","yes")
EXT_PREFIX = os.getenv("AUTO_ENABLE_EXT_PREFIX", "")
EXT_EXCLUDE = {x.strip() for x in os.getenv("AUTO_ENABLE_EXT_EXCLUDE","" ).split(',') if x.strip()}
EXT_ALLOWLIST = {x.strip() for x in os.getenv("EXTENSION_ALLOWLIST","" ).split(',') if x.strip()}
EXT_BLOCKLIST = {x.strip() for x in os.getenv("EXTENSION_BLOCKLIST","" ).split(',') if x.strip()}
DISABLE_RTX = os.getenv("DISABLE_RTX","0").lower() in ("1","true","yes")

os.environ.setdefault("OMNI_FETCH_ASSETS","1")

try:
    from isaacsim import SimulationApp  # noqa: E402
except Exception as e:
    raise SystemExit(f"SimulationApp import failed: {e}")

simulation_app = SimulationApp({"headless": HEADLESS})

CRITICAL_ROS_EXT_NAMES = [
    "isaacsim.ros2.bridge",      # New preferred name per deprecation warning
    "omni.isaac.ros2_bridge",    # Legacy name
]

DISABLE_FORCE_ROS_BRIDGE = os.getenv("DISABLE_FORCE_ROS_BRIDGE","0").lower() in ("1","true","yes")
ROS_BRIDGE_AUTO_GRAPH = os.getenv("ROS_BRIDGE_AUTO_GRAPH","1").lower() in ("1","true","yes")
ROS_BRIDGE_VERBOSE_DIAG = os.getenv("ROS_BRIDGE_VERBOSE_DIAG","1").lower() in ("1","true","yes")

def _env_dump_for_ros():
    if not ROS_BRIDGE_VERBOSE_DIAG:
        return
    keys = [
        'PYTHONPATH','OMNI_EXTENSIONS_PATH','OMNI_KIT_PRIORITIZED_EXTENSIONS','LD_LIBRARY_PATH',
        'ROS_DOMAIN_ID','RMW_IMPLEMENTATION'
    ]
    print('[ROS2_BRIDGE][ENV] ----')
    for k in keys:
        v = os.getenv(k)
        if v:
            print(f'  {k}={v}')
    print('[ROS2_BRIDGE][ENV] ----')

def _dump_all_ros_extensions():
    if not ROS_BRIDGE_VERBOSE_DIAG:
        return
    try:
        import omni.kit.app
        app = omni.kit.app.get_app(); mgr = app.get_extension_manager()
        all_exts = mgr.get_extensions()
        print('[ROS2_BRIDGE][EXT] Listing extensions containing "ros" substring:')
        for ext_id, meta in sorted(all_exts.items()):
            if 'ros' in ext_id.lower():
                enabled = False
                for m in ("is_extension_enabled","is_enabled"):
                    if hasattr(mgr,m):
                        try:
                            if getattr(mgr,m)(ext_id):
                                enabled=True; break
                        except Exception: pass
                path = meta.get('path') or meta.get('location') or '?'
                print(f'   - {ext_id} enabled={enabled} version={meta.get("version","?")} path={path}')
    except Exception as e:
        print(f'[ROS2_BRIDGE][EXT] Dump failed: {e}')

def _force_ros2_bridge(verbose=True):
    if DISABLE_FORCE_ROS_BRIDGE:
        if verbose:
            print("[ROS2_BRIDGE] Force enable skipped (DISABLE_FORCE_ROS_BRIDGE=1)")
        return False
    try:
        import omni.kit.app
        app_obj = omni.kit.app.get_app()
        mgr = app_obj.get_extension_manager() if hasattr(app_obj,'get_extension_manager') else None
        if not mgr:
            return
        ext_list = []
        try:
            if hasattr(mgr,'get_extensions'):
                ext_list = list(getattr(mgr,'get_extensions')().keys())
            else:
                # fallback union method
                for attr in ("get_enabled_extension_ids","get_disabled_extension_ids"):
                    if hasattr(mgr,attr):
                        try: ext_list.extend(getattr(mgr,attr)())
                        except Exception: pass
            ext_list = list(set(ext_list))
        except Exception:
            pass

        for target in CRITICAL_ROS_EXT_NAMES:
            present = (target in ext_list) if ext_list else True
            if not present:
                if verbose: print(f"[ROS2_BRIDGE] {target} not yet discovered")
                continue
            already=False
            for m in ("is_extension_enabled","is_enabled"):
                if hasattr(mgr,m):
                    try:
                        if getattr(mgr,m)(target):
                            already=True; break
                    except Exception: continue
            if already:
                if verbose: print(f"[ROS2_BRIDGE] {target} already enabled")
                return True
            for m in ("set_extension_enabled_immediate","set_extension_enabled"):
                if hasattr(mgr,m):
                    try:
                        getattr(mgr,m)(target, True)
                        if verbose: print(f"[ROS2_BRIDGE] Enabled {target}")
                        return True
                    except Exception: continue
            if verbose: print(f"[ROS2_BRIDGE] Could not enable {target}")
        return False
    except Exception as e:
        if verbose:
            print(f"[ROS2_BRIDGE] Force enable failed: {e}")
    return False

def _wait_for_ros2_node_types(timeout=6.0, poll=0.2, required=None, verbose=True):
    """Wait until required ros2 bridge OmniGraph node types are registered.

    Env override:
      ROS_BRIDGE_REQUIRED_NODES=comma,list (full type names or short names without prefix)
      ROS_BRIDGE_NODE_WAIT_TIMEOUT=seconds
    """
    if DISABLE_FORCE_ROS_BRIDGE:
        return False
    try:
        import omni.graph.core as og
    except Exception as e:
        if verbose:
            print(f"[ROS2_BRIDGE] OmniGraph core import failed (cannot wait nodes): {e}")
        return False

    # Build required list
    env_req = os.getenv("ROS_BRIDGE_REQUIRED_NODES", "")
    if required is None:
        if env_req.strip():
            raw = [x.strip() for x in env_req.split(',') if x.strip()]
        else:
            # Default commonly used nodes
            raw = [
                "ROS2Context",
                "ROS2PublishTransformTree",
                "ROS2CameraInfoHelper",
                "ROS2SubscribeJointState",
                "ROS2CameraHelper",
            ]
    else:
        raw = list(required)
    # Normalize to full names
    pref_new = "isaacsim.ros2.bridge."
    full_required = []
    for name in raw:
        if name.startswith(pref_new):
            full_required.append(name)
        elif name.startswith("omni.isaac.ros2_bridge."):
            # legacy style fully qualified
            full_required.append(name)
        else:
            full_required.append(pref_new + name)

    timeout = float(os.getenv("ROS_BRIDGE_NODE_WAIT_TIMEOUT", str(timeout)))
    deadline = time.time() + timeout
    missing_prev = set(full_required)
    # Some nodes may be registered under legacy prefix; include mapping attempts
    legacy_map = {r: r.replace("isaacsim.ros2.bridge.", "omni.isaac.ros2_bridge.") for r in full_required}

    def _is_registered(tn):
        try:
            nt = og.get_node_type_registry().get_node_type(tn)
            return nt is not None
        except Exception:
            return False

    while time.time() < deadline:
        missing = []
        for full in full_required:
            if _is_registered(full):
                continue
            # check legacy alias
            legacy = legacy_map.get(full)
            if legacy and _is_registered(legacy):
                continue
            missing.append(full)
        if not missing:
            if verbose:
                print(f"[ROS2_BRIDGE] All required node types registered: {len(full_required)}")
            return True
        # Print only if changed to reduce spam
        s_missing = set(missing)
        if verbose and s_missing != missing_prev:
            short = [m.split('.')[-1] for m in missing]
            print(f"[ROS2_BRIDGE] Waiting node types ({len(missing)} missing): {', '.join(short)}")
            missing_prev = s_missing
        simulation_app.update(); time.sleep(poll)
    if verbose:
        print("[ROS2_BRIDGE] Node type registration timeout; still missing:")
        for m in missing_prev:
            print("  -", m)
    return False

def _dump_ros2_node_types(prefix_filters=("ros2","ROS2","isaacsim.ros2","omni.isaac.ros2")):
    """Diagnostic: list currently registered OmniGraph node types containing any of the prefixes."""
    try:
        import omni.graph.core as og
        names = []
        reg = og.get_node_type_registry()
        # get_registered_node_type_names may differ across versions; try common APIs
        for attr in ("get_node_type_names", "get_registered_node_type_names", "get_all_node_type_names"):
            if hasattr(reg, attr):
                try:
                    names = list(getattr(reg, attr)())
                    break
                except Exception:
                    continue
        if not names:
            print("[ROS2_BRIDGE][DIAG] Could not enumerate node type names.")
            return
        filt = []
        for n in names:
            low = n.lower()
            if any(p.lower() in low for p in prefix_filters):
                filt.append(n)
        print(f"[ROS2_BRIDGE][DIAG] Registered ROS-related node types ({len(filt)}):")
        for n in sorted(filt):
            print("   -", n)
    except Exception as e:
        print(f"[ROS2_BRIDGE][DIAG] Dump failed: {e}")

def _fallback_enable_ros2_extensions():
    """Attempt enabling legacy / supplemental ros2 bridge extensions if provided.

    Controlled by env:
      ROS_BRIDGE_FALLBACK_ENABLE=1 -> activate fallback
      ROS_BRIDGE_FALLBACK_LIST=comma,list (default legacy guesses)
    """
    if os.getenv("ROS_BRIDGE_FALLBACK_ENABLE","0").lower() not in ("1","true","yes"):
        return False
    fallback_raw = os.getenv("ROS_BRIDGE_FALLBACK_LIST",
                              "omni.isaac.ros2_bridge,isaacsim.ros2.bridge.graph,omni.isaac.ros2_bridge.graph")
    cand = [c.strip() for c in fallback_raw.split(',') if c.strip()]
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        mgr = app.get_extension_manager() if hasattr(app,'get_extension_manager') else None
        if not mgr:
            print("[ROS2_BRIDGE][FB] No extension manager.")
            return False
        enabled_any = False
        for name in cand:
            already = False
            for m in ("is_extension_enabled","is_enabled"):
                if hasattr(mgr,m):
                    try:
                        if getattr(mgr,m)(name):
                            already = True; break
                    except Exception:
                        continue
            if already:
                continue
            for m in ("set_extension_enabled_immediate","set_extension_enabled"):
                if hasattr(mgr,m):
                    try:
                        getattr(mgr,m)(name, True)
                        print(f"[ROS2_BRIDGE][FB] Enabled fallback extension: {name}")
                        enabled_any = True
                        break
                    except Exception as e:
                        print(f"[ROS2_BRIDGE][FB] Enable failed {name}: {e}")
        return enabled_any
    except Exception as e:
        print(f"[ROS2_BRIDGE][FB] Fallback process error: {e}")
        return False

def _post_enable_ros_bridge_check():
    """After initial enable attempt & wait, try fallback strategy if nodes missing."""
    if os.getenv("ROS_BRIDGE_WAIT_NODES","1").lower() not in ("1","true","yes"):
        return
    # First attempt already executed outside; we just verify presence
    success = _wait_for_ros2_node_types(timeout=float(os.getenv("ROS_BRIDGE_NODE_WAIT_TIMEOUT","6")),
                                        poll=0.3, verbose=False)
    if success:
        print("[ROS2_BRIDGE] Node types present after initial wait.")
        _dump_ros2_node_types()
        return
    print("[ROS2_BRIDGE] Initial node wait failed.")
    # Try auto graph extension enable if allowed
    if ROS_BRIDGE_AUTO_GRAPH:
        try:
            import omni.kit.app
            app = omni.kit.app.get_app(); mgr = app.get_extension_manager()
            graph_like = [e for e in mgr.get_extensions().keys() if 'ros2.bridge' in e and 'graph' in e]
            enabled_any = False
            for ext_id in graph_like:
                already = False
                for m in ("is_extension_enabled","is_enabled"):
                    if hasattr(mgr,m):
                        try:
                            if getattr(mgr,m)(ext_id):
                                already=True; break
                        except Exception: pass
                if already:
                    continue
                for m in ("set_extension_enabled_immediate","set_extension_enabled"):
                    if hasattr(mgr,m):
                        try:
                            getattr(mgr,m)(ext_id, True)
                            print(f"[ROS2_BRIDGE][AUTO_GRAPH] Enabled graph extension: {ext_id}")
                            enabled_any = True
                            break
                        except Exception as e:
                            print(f"[ROS2_BRIDGE][AUTO_GRAPH] Failed enable {ext_id}: {e}")
            if enabled_any:
                # Warm frames and re-check
                for _ in range(60):
                    simulation_app.update(); time.sleep(0.01)
                if _wait_for_ros2_node_types(timeout=5, poll=0.4, verbose=True):
                    print("[ROS2_BRIDGE] Node types appeared after auto graph enable.")
                    _dump_ros2_node_types(); return
        except Exception as e:
            print(f"[ROS2_BRIDGE][AUTO_GRAPH] Error during auto graph attempt: {e}")
    print("[ROS2_BRIDGE] Attempting fallback enable of legacy/supplemental extensions...")
    if _fallback_enable_ros2_extensions():
        # Warm a few frames
        for _ in range(40):
            simulation_app.update(); time.sleep(0.01)
        fb_timeout = float(os.getenv("ROS_BRIDGE_FALLBACK_TIMEOUT","10"))
        if _wait_for_ros2_node_types(timeout=fb_timeout, poll=0.4, verbose=True):
            print("[ROS2_BRIDGE] Fallback succeeded; node types registered.")
            _dump_ros2_node_types()
            return
    print("[ROS2_BRIDGE] Fallback did NOT register required node types. Dumping what exists:")
    _dump_ros2_node_types()
    print("[ROS2_BRIDGE] SUGGESTIONS: \n - Verify extension versions match Isaac Sim build\n - Clear Kit cache (~/.local/share/ov/Kit/*/cache) and restart\n - Ensure no PYTHONPATH hiding bridge ogn specs\n - Try enabling legacy 'omni.isaac.ros2_bridge' manually.")

def _list_ros2_extensions():
    try:
        import omni.kit.app
        app = omni.kit.app.get_app(); mgr = app.get_extension_manager()
        print("[ROS2_BRIDGE][DIAG] ros2-related extensions present:")
        for ext_id, meta in mgr.get_extensions().items():
            if 'ros2' in ext_id.lower():
                enabled = False
                for m in ("is_extension_enabled","is_enabled"):
                    if hasattr(mgr,m):
                        try:
                            if getattr(mgr,m)(ext_id):
                                enabled = True
                                break
                        except Exception: pass
                print(f"   - {ext_id} enabled={enabled} version={meta.get('version','?')}")
    except Exception as e:
        print(f"[ROS2_BRIDGE][DIAG] Could not list ros2 extensions: {e}")

def _scan_ogn_specs_for_bridge():
    """Attempt to locate .ogn spec files shipped with ros2 bridge extensions."""
    try:
        import omni.kit.app, os, glob
        app = omni.kit.app.get_app(); mgr = app.get_extension_manager()
        roots = []
        for ext_id, meta in mgr.get_extensions().items():
            if 'ros2' in ext_id.lower():
                folder = meta.get('path') or meta.get('location') or ''
                if folder and os.path.isdir(folder):
                    roots.append(folder)
        printed = False
        for r in roots:
            ogn_paths = glob.glob(os.path.join(r, '**', '*.ogn'), recursive=True)
            if ogn_paths:
                if not printed:
                    print("[ROS2_BRIDGE][DIAG] Discovered .ogn spec files:")
                    printed = True
                for p in ogn_paths[:50]:  # limit
                    print("   -", p)
                if len(ogn_paths) > 50:
                    print(f"   ... (+{len(ogn_paths)-50} more)")
        if not printed:
            print("[ROS2_BRIDGE][DIAG] No .ogn spec files found under ros2 extension paths (unexpected).")
    except Exception as e:
        print(f"[ROS2_BRIDGE][DIAG] OGN scan failed: {e}")

def _force_discovery_round():
    """If node types missing, try enabling any extra graph-related ros2 extensions and forcing registry refresh."""
    if os.getenv("ROS_BRIDGE_FORCE_DISCOVERY","0").lower() not in ("1","true","yes"):
        return
    print("[ROS2_BRIDGE][DISCOVERY] Forced discovery round starting...")
    _list_ros2_extensions()
    _scan_ogn_specs_for_bridge()
    # Try enabling potential graph sub-extensions
    candidates = [
        'isaacsim.ros2.bridge.graph',
        'omni.isaac.ros2_bridge.graph',
        'isaacsim.ros2.bridge.nodes'
    ]
    try:
        import omni.kit.app
        app = omni.kit.app.get_app(); mgr = app.get_extension_manager()
        for name in candidates:
            if name not in mgr.get_extensions():
                continue
            already = False
            for m in ("is_extension_enabled","is_enabled"):
                if hasattr(mgr,m):
                    try:
                        if getattr(mgr,m)(name):
                            already = True; break
                    except Exception: pass
            if already:
                continue
            for m in ("set_extension_enabled_immediate","set_extension_enabled"):
                if hasattr(mgr,m):
                    try:
                        getattr(mgr,m)(name, True)
                        print(f"[ROS2_BRIDGE][DISCOVERY] Enabled supplementary extension: {name}")
                        break
                    except Exception as e:
                        print(f"[ROS2_BRIDGE][DISCOVERY] Failed enabling {name}: {e}")
        # After enabling supplementary nodes, give frames for registration
        for _ in range(60):
            simulation_app.update(); time.sleep(0.01)
        _post_enable_ros_bridge_check()
    except Exception as e:
        print(f"[ROS2_BRIDGE][DISCOVERY] Discovery round error: {e}")

if SAFE_RENDER or DISABLE_RTX:
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
            ("/app/asyncRendering/enabled", False),
            ("/app/renderer/multiGpu/enabled", False),
            ("/rtx/rendermode", 0),  # basic
        ]:
            try: s.set(k,v)
            except Exception: pass
        print("[SAFE_RENDER] Low render settings applied (RTX disabled)." if SAFE_RENDER else "[SAFE_RENDER] RTX forced OFF.")
    except Exception:
        pass

def enable_extensions():
    # Skip mass enabling if in minimal diagnostic mode to avoid GPU spike
    if os.getenv('MINIMAL_TEST','0').lower() in ('1','true','yes'):
        print('[EXT] Skipping auto-enable (MINIMAL_TEST=1).')
        return
    if not AUTO_ENABLE_ALL_EXT:
        print('[EXT] AUTO_ENABLE_ALL_EXTENSIONS disabled.')
        return
    try:
        import omni.kit.app
        app_obj = omni.kit.app.get_app()
        mgr = None
        if app_obj and hasattr(app_obj,'get_extension_manager'):
            try: mgr = app_obj.get_extension_manager()
            except Exception: mgr=None
        if mgr is None:
            try:
                import omni.ext
                mgr = getattr(omni.ext,'get_extension_manager',lambda:None)()
            except Exception:
                mgr=None
        if mgr is None:
            print('[EXT] Extension manager not available.')
            return
        try:
            ext_dict = mgr.get_extensions(); names=list(ext_dict.keys())
        except Exception:
            names=[]
            for attr in ("get_enabled_extension_ids","get_disabled_extension_ids"):
                if hasattr(mgr,attr):
                    try: names.extend(getattr(mgr,attr)())
                    except Exception: pass
            names=list(set(names))
        force_first=["omni.isaac.ros2_bridge"]
        ordered=[]
        for n in force_first + names:
            if n not in ordered:
                ordered.append(n)
        def _is_enabled(n):
            for m in ("is_extension_enabled","is_enabled"):
                if hasattr(mgr,m):
                    try: return bool(getattr(mgr,m)(n))
                    except Exception: continue
            return False
        def _enable(n):
            for m in ("set_extension_enabled_immediate","set_extension_enabled"):
                if hasattr(mgr,m):
                    try:
                        getattr(mgr,m)(n,True);return True
                    except Exception: continue
            return False
        new_cnt=0; skipped=0
        for name in ordered:
            if EXT_PREFIX and not name.startswith(EXT_PREFIX):
                continue
            if name in EXT_EXCLUDE:
                skipped+=1; continue
            if EXT_ALLOWLIST and name not in EXT_ALLOWLIST:
                skipped+=1; continue
            if EXT_BLOCKLIST and name in EXT_BLOCKLIST:
                skipped+=1; continue
            if not _is_enabled(name):
                if _enable(name):
                    new_cnt+=1
                else:
                    skipped+=1
        print(f"[EXT] enabled_new={new_cnt} skipped={skipped} allowlist={'ON' if EXT_ALLOWLIST else 'OFF'} blocklist={'ON' if EXT_BLOCKLIST else 'OFF'}")
    except Exception as e:
        print(f"[EXT] enable failed: {e}")

ROS_BRIDGE_ENABLE_STRATEGY = os.getenv("ROS_BRIDGE_ENABLE_STRATEGY","post_stage")
ROS_BRIDGE_WARMUP_FRAMES = int(os.getenv("ROS_BRIDGE_WARMUP_FRAMES","30"))
ROS_BRIDGE_RETRY_SEC = float(os.getenv("ROS_BRIDGE_RETRY_SEC","3"))
ROS_BRIDGE_WITH_STAGE_DELAY_FRAMES = int(os.getenv("ROS_BRIDGE_WITH_STAGE_DELAY_FRAMES","5"))
ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES = int(os.getenv("ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES","10"))  # only used if strategy=pre_stage

# Early extension enabling (but not forcing ros bridge unless strategy says so)
enable_extensions()
_env_dump_for_ros()
_dump_all_ros_extensions()

if ROS_BRIDGE_ENABLE_STRATEGY in ("early","pre_stage") and not DISABLE_FORCE_ROS_BRIDGE:
    if ROS_BRIDGE_ENABLE_STRATEGY == "pre_stage":
        print(f"[ROS2_BRIDGE] Strategy=pre_stage: warming {ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES} frames, then enabling BEFORE stage load.")
        for i in range(max(0, ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES)):
            simulation_app.update(); time.sleep(0.01)
    else:
        print("[ROS2_BRIDGE] Strategy=early: attempting enable immediately BEFORE stage load.")
    deadline = time.time() + ROS_BRIDGE_RETRY_SEC
    tries = 0
    while time.time() < deadline:
        if _force_ros2_bridge(verbose=(tries==0)):
            if os.getenv("ROS_BRIDGE_WAIT_NODES","1").lower() in ("1","true","yes"):
                _wait_for_ros2_node_types()
            break
        # Give a couple of frames so internal registries & extension machinery settle
        simulation_app.update(); time.sleep(0.2); tries += 1
else:
    if ROS_BRIDGE_ENABLE_STRATEGY == "with_stage":
        print("[ROS2_BRIDGE] Strategy=with_stage: will attempt enable during stage load loop after delay frames.")
    elif ROS_BRIDGE_ENABLE_STRATEGY == "pre_stage":
        # pre_stage handled above (only executes if not DISABLE_FORCE_ROS_BRIDGE)
        pass
    else:
        print(f"[ROS2_BRIDGE] Strategy={ROS_BRIDGE_ENABLE_STRATEGY}: will defer forcing until after stage load.")

# ----------------------------------------------------------

import omni.usd
usd_ctx = omni.usd.get_context()
SKIP_STAGE = os.getenv("SKIP_STAGE","0").lower() in ("1","true","yes")
if not SKIP_STAGE:
    print(f"[SCENE] Opening USD: {USD_PATH}")
    usd_ctx.open_stage(USD_PATH)
else:
    print("[SCENE] SKIP_STAGE=1 -> Stage will not be opened here.")

if not SKIP_STAGE:
    start=time.time()
    _with_stage_frame = 0
    _ros_bridge_attempted_midload = False
    _ros_bridge_midload_ok = False
    while True:
        st=usd_ctx.get_stage()
        if st:
            try:
                if hasattr(usd_ctx,'is_stage_loading') and not usd_ctx.is_stage_loading():
                    break
                if time.time()-start>5:
                    print("[SCENE] Stage load wait timeout (5s) -> continuing")
                    break
            except Exception:
                break
        # Attempt mid-load enable if strategy = with_stage
        if ROS_BRIDGE_ENABLE_STRATEGY == "with_stage" and not DISABLE_FORCE_ROS_BRIDGE and not _ros_bridge_attempted_midload:
            if _with_stage_frame >= ROS_BRIDGE_WITH_STAGE_DELAY_FRAMES:
                _ros_bridge_attempted_midload = True
                print(f"[ROS2_BRIDGE] (with_stage) Enabling ROS bridge during load at frame {_with_stage_frame}...")
                _ros_bridge_midload_ok = bool(_force_ros2_bridge(verbose=True))
                if _ros_bridge_midload_ok and os.getenv("ROS_BRIDGE_WAIT_NODES","1").lower() in ("1","true","yes"):
                    _wait_for_ros2_node_types(timeout=float(os.getenv("ROS_BRIDGE_NODE_WAIT_TIMEOUT","4")), poll=0.25)
            else:
                _with_stage_frame += 1
        simulation_app.update(); time.sleep(0.02)
    print("[SCENE] Stage ready.")
    if ROS_BRIDGE_ENABLE_STRATEGY == "with_stage" and not DISABLE_FORCE_ROS_BRIDGE:
        if not _ros_bridge_midload_ok:
            print("[ROS2_BRIDGE] (with_stage) Mid-load enable failed or skipped; retrying post-stage.")
        else:
            print("[ROS2_BRIDGE] (with_stage) Bridge already enabled during load; post-stage retry skipped.")
    if ROS_BRIDGE_ENABLE_STRATEGY != "early" and not DISABLE_FORCE_ROS_BRIDGE and (ROS_BRIDGE_ENABLE_STRATEGY != "with_stage" or not _ros_bridge_midload_ok):
        # Warmup a few frames first to reduce race with renderer / physx init
        for _ in range(max(0, ROS_BRIDGE_WARMUP_FRAMES)):
            simulation_app.update()
            time.sleep(0.005)
        deadline = time.time() + ROS_BRIDGE_RETRY_SEC
        attempt = 0
        while time.time() < deadline:
            ok_bridge = _force_ros2_bridge(verbose=(attempt==0))
            if ok_bridge:
                if os.getenv("ROS_BRIDGE_WAIT_NODES","1").lower() in ("1","true","yes"):
                    _wait_for_ros2_node_types()
                break
            simulation_app.update(); time.sleep(0.3); attempt += 1
        else:
            print("[ROS2_BRIDGE] Could not enable bridge within retry window.")
    # List sublayers to aid diagnostics
    try:
        st = usd_ctx.get_stage()
        if st:
            root = st.GetRootLayer()
            subs = list(root.subLayerPaths)
            print(f"[SCENE] Root layer: {root.identifier}")
            if subs:
                print("[SCENE] Sublayers (order):")
                for i,p in enumerate(subs):
                    print(f"  [{i}] {p}")
            else:
                print("[SCENE] No sublayers detected.")
    except Exception as e:
        print(f"[SCENE] Could not list sublayers: {e}")
else:
    print("[SCENE] Stage load loop skipped (SKIP_STAGE=1).")

# ------------------------------------------------------------------
# Lightweight HTTP Reset Interface (optional)
# Env Vars:
#   RESET_HTTP_ENABLE=1           -> Enable server (default 1)
#   RESET_HTTP_PORT=8099          -> Port
#   RESET_ALLOW_REMOTE=0          -> 1 -> bind 0.0.0.0 else 127.0.0.1
#   RESET_INCLUDE_ROBOT=1         -> Also call reset_robot() if available
#   RESET_RELOAD_STAGE=0          -> If 1, attempt stage.Reload() (safe-ish)
#   RESET_CLEAR_RL_BUFFERS=1      -> Clear replay buffers if present
#   RESET_EXTRA_WARMUP_FRAMES=10  -> Frames to step after reset
# Endpoint Examples:
#   curl -s http://127.0.0.1:8099/health
#   curl -X POST http://127.0.0.1:8099/reset
#   curl -X POST http://127.0.0.1:8099/reset?robot=0&reload=1
# ------------------------------------------------------------------
import urllib.parse as _urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket

RESET_HTTP_ENABLE = os.getenv("RESET_HTTP_ENABLE","1").lower() in ("1","true","yes")
RESET_HTTP_PORT = int(os.getenv("RESET_HTTP_PORT","8099"))
RESET_ALLOW_REMOTE = os.getenv("RESET_ALLOW_REMOTE","0").lower() in ("1","true","yes")
RESET_DEFAULT_INCLUDE_ROBOT = os.getenv("RESET_INCLUDE_ROBOT","1").lower() in ("1","true","yes")
RESET_DEFAULT_RELOAD_STAGE = os.getenv("RESET_RELOAD_STAGE","0").lower() in ("1","true","yes")
RESET_DEFAULT_CLEAR_BUFFERS = os.getenv("RESET_CLEAR_RL_BUFFERS","1").lower() in ("1","true","yes")
RESET_EXTRA_WARMUP_FRAMES = int(os.getenv("RESET_EXTRA_WARMUP_FRAMES","10"))
RESET_RECREATE_ROS_NODES = os.getenv("RESET_RECREATE_ROS_NODES","1").lower() in ("1","true","yes")
RESET_RELOAD_USE_OPEN = os.getenv("RESET_RELOAD_USE_OPEN","1").lower() in ("1","true","yes")

# Runtime control flags for safe reset / node gating
ros_runtime_active = True   # Disabled during reset to prevent callbacks using invalid resources
executor = None             # Will hold rclpy executor (assigned later when created)
get_end_effector_pose = None
reward_node = None
joy_publisher = None

_pending_reset_requests = []  # list of dicts
_reset_lock = threading.Lock()

def _perform_simulation_reset(include_robot=True, reload_stage=False, clear_buffers=True):
    info = {
        'include_robot': include_robot,
        'reload_stage': reload_stage,
        'clear_buffers': clear_buffers,
        'ok': True,
        'errors': []
    }
    mem_diag = os.getenv("RESET_MEM_DIAG","0").lower() in ("1","true","yes")
    def _mem_snap(tag):
        if not mem_diag:
            return
        try:
            import gc, resource
            gc.collect()
            usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            node_counts = {}
            for obj in gc.get_objects():
                tn = type(obj).__name__
                if tn in ("Node","Action_Publisher","RewardPublisher","Get_End_Effector_Pose"):
                    node_counts[tn] = node_counts.get(tn,0)+1
            print(f"[RESET][MEM] {tag}: maxrss_kb={usage_kb} nodes={node_counts}")
        except Exception as e:
            print(f"[RESET][MEM] snap_err:{e}")
    _mem_snap('before')
    try:
        # Pause timeline if playing
        try:
            import omni.timeline
            tl_mod = omni.timeline
            tl_if = None
            if hasattr(tl_mod,'get_timeline_interface'):
                tl_if = tl_mod.get_timeline_interface()
            elif hasattr(tl_mod,'get_timeline'):
                tl_if = tl_mod.get_timeline()
            if tl_if and hasattr(tl_if,'is_playing') and tl_if.is_playing() and hasattr(tl_if,'pause'):
                tl_if.pause()
                info['timeline_paused'] = True
        except Exception as e:
            info['errors'].append(f'timeline_pause:{e}')
        # Give a couple frames to settle
        for _ in range(3):
            simulation_app.update(); time.sleep(0.01)
        # Reload stage if requested
        if reload_stage:
            try:
                import omni.usd
                uc = omni.usd.get_context()
                # Disable runtime publishing / callbacks during reload
                global ros_runtime_active, executor, get_end_effector_pose, reward_node, joy_publisher
                ros_runtime_active = False
                # Remove existing ROS nodes from executor if any
                existing_nodes = []
                for _n in (get_end_effector_pose, reward_node, joy_publisher):
                    if _n is not None:
                        existing_nodes.append(_n)
                if executor is not None:
                    for n in existing_nodes:
                        try:
                            executor.remove_node(n)
                        except Exception:
                            pass
                for n in existing_nodes:
                    try:
                        n.destroy_node()
                    except Exception:
                        pass
                # Stage open (preferred) or legacy reload
                if RESET_RELOAD_USE_OPEN:
                    uc.open_stage(USD_PATH)
                else:
                    st_old = uc.get_stage()
                    if st_old is not None:
                        try: st_old.Reload()
                        except Exception as e: info['errors'].append(f'stage_reload_call:{e}')
                # Wait for stage load completion
                load_start = time.time()
                while True:
                    st_new = uc.get_stage()
                    busy = False
                    if st_new is not None:
                        try:
                            if hasattr(uc,'is_stage_loading'):
                                busy = uc.is_stage_loading()
                            else:
                                from omni.usd import StageLoadingStatus
                                status = uc.get_stage_loading_status()
                                busy = status not in (StageLoadingStatus.COMPLETE, StageLoadingStatus.FAILED)
                        except Exception:
                            busy = False
                    if st_new is not None and not busy:
                        break
                    if time.time() - load_start > 25.0:
                        info['errors'].append('stage_reload_timeout')
                        break
                    simulation_app.update(); time.sleep(0.03)
                info['stage_reloaded'] = True
                # Stabilization frames
                for _ in range(12):
                    simulation_app.update(); time.sleep(0.01)
                # Recreate ROS nodes if requested
                recreated = False
                if RESET_RECREATE_ROS_NODES and 'Get_End_Effector_Pose' in globals() and Get_End_Effector_Pose is not None:
                    try:
                        # Ensure class definitions exist (idempotent)
                        if get_end_effector_pose is None or reward_node is None or joy_publisher is None:
                            pass
                        # Recreate instances
                        from rclpy.executors import MultiThreadedExecutor as _MTE
                        if executor is None:
                            try:
                                import rclpy
                                if not rclpy.utilities.ok():
                                    rclpy.init(args=None)
                                executor = _MTE()
                            except Exception as e:
                                info['errors'].append(f'exec_create:{e}')
                        get_end_effector_pose = Get_End_Effector_Pose()
                        reward_node = RewardPublisher()
                        joy_publisher = Action_Publisher()
                        if executor is not None:
                            try: executor.add_node(get_end_effector_pose)
                            except Exception: pass
                            try: executor.add_node(reward_node)
                            except Exception: pass
                            try: executor.add_node(joy_publisher)
                            except Exception: pass
                        recreated = True
                        info['ros_nodes_recreated'] = True
                    except Exception as e:
                        info['errors'].append(f'ros_recreate:{e}')
                # Re-enable runtime publishing
                ros_runtime_active = True
                # Extra warmup after recreation
                for _ in range(8 if recreated else 4):
                    simulation_app.update(); time.sleep(0.01)
                # Restart timeline if paused
                try:
                    if tl_if and hasattr(tl_if,'play'):
                        tl_if.play(); info['timeline_started'] = True
                except Exception as e:
                    info['errors'].append(f'timeline_play:{e}')
            except Exception as e:
                info['errors'].append(f'stage_reload:{e}')
        # Clear RL buffers
        if clear_buffers:
            for _name in ['rb_states','rb_images','rb_actions','rb_rewards','rb_dones']:
                if _name in globals():
                    try:
                        lst = globals()[_name]
                        if isinstance(lst, list):
                            lst.clear()
                    except Exception as e:
                        info['errors'].append(f'clear_{_name}:{e}')
            if 'updates_done' in globals():
                try: globals()['updates_done'] = 0
                except Exception: pass
            info['buffers_cleared'] = True
        # Reset robot joints (ROS publisher)
        if include_robot and 'reset_robot' in globals():
            try:
                reset_robot()
                info['robot_reset_called'] = True
            except Exception as e:
                info['errors'].append(f'reset_robot:{e}')
        # Post-reset warmup frames
        for _ in range(max(0, RESET_EXTRA_WARMUP_FRAMES)):
            simulation_app.update(); time.sleep(0.005)
        _mem_snap('after')
        return info
    except Exception as e:
        info['ok'] = False
        info['errors'].append(str(e))
        _mem_snap('after_error')
        return info

def _queue_reset_request(params: dict):
    with _reset_lock:
        _pending_reset_requests.append(params)

def _pop_next_reset():
    with _reset_lock:
        if _pending_reset_requests:
            return _pending_reset_requests.pop(0)
    return None

class _ResetHTTPHandler(BaseHTTPRequestHandler):
    server_version = "SimResetHTTP/0.1"
    def _send(self, code=200, body=b"OK", ctype="text/plain"):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
    def log_message(self, fmt, *args):  # silence
        return
    def do_GET(self):
        parsed = _urlparse.urlparse(self.path)
        if parsed.path == '/health':
            self._send(200, b"healthy")
            return
        if parsed.path == '/reset':
            qs = _urlparse.parse_qs(parsed.query)
            _queue_reset_request({
                'include_robot': (qs.get('robot',[str(int(RESET_DEFAULT_INCLUDE_ROBOT))])[0] in ('1','true','yes')),
                'reload_stage': (qs.get('reload',[str(int(RESET_DEFAULT_RELOAD_STAGE))])[0] in ('1','true','yes')),
                'clear_buffers': (qs.get('clear_buffers',[str(int(RESET_DEFAULT_CLEAR_BUFFERS))])[0] in ('1','true','yes')),
                'ts': time.time(),
                'source': 'http-get'
            })
            self._send(200, b"queued")
            return
        self._send(404, b"not found", ctype='text/plain')
    def do_POST(self):
        parsed = _urlparse.urlparse(self.path)
        if parsed.path == '/reset':
            length = int(self.headers.get('Content-Length','0'))
            body = b''
            if length:
                try: body = self.rfile.read(length)
                except Exception: body = b''
            payload = {}
            try:
                if body:
                    import json as _json
                    payload = _json.loads(body.decode('utf-8')) if body.strip() else {}
            except Exception:
                payload = {}
            _queue_reset_request({
                'include_robot': bool(payload.get('include_robot', RESET_DEFAULT_INCLUDE_ROBOT)),
                'reload_stage': bool(payload.get('reload_stage', RESET_DEFAULT_RELOAD_STAGE)),
                'clear_buffers': bool(payload.get('clear_buffers', RESET_DEFAULT_CLEAR_BUFFERS)),
                'ts': time.time(),
                'source': 'http-post'
            })
            self._send(200, b"queued")
            return
        self._send(404, b"not found", ctype='text/plain')

_reset_http_server = None
def _start_reset_http_server():
    global _reset_http_server
    host = '0.0.0.0' if RESET_ALLOW_REMOTE else '127.0.0.1'
    try:
        _reset_http_server = HTTPServer((host, RESET_HTTP_PORT), _ResetHTTPHandler)
    except OSError as e:
        print(f"[RESET][HTTP] Could not bind {host}:{RESET_HTTP_PORT} -> {e}")
        return
    def _serve():
        print(f"[RESET][HTTP] Listening on http://{host}:{RESET_HTTP_PORT} (remote={'ON' if RESET_ALLOW_REMOTE else 'OFF'})")
        while True:
            try:
                _reset_http_server.handle_request()
            except Exception as e:
                print(f"[RESET][HTTP] Server error: {e}")
                time.sleep(0.5)
    threading.Thread(target=_serve, daemon=True).start()

def _process_reset_requests():
    # Called each frame; perform at most one reset per frame to avoid stacking
    req = _pop_next_reset()
    if not req:
        return
    print(f"[RESET] Processing reset request: include_robot={req['include_robot']} reload_stage={req['reload_stage']} clear_buffers={req['clear_buffers']}")
    result = _perform_simulation_reset(include_robot=req['include_robot'], reload_stage=req['reload_stage'], clear_buffers=req['clear_buffers'])
    if result.get('ok'):
        print(f"[RESET] Done. Flags: robot={result.get('robot_reset_called',False)} stageReload={result.get('stage_reloaded',False)} buffersCleared={result.get('buffers_cleared',False)}")
    else:
        print(f"[RESET] FAILED: {result.get('errors')}")

if RESET_HTTP_ENABLE:
    _start_reset_http_server()
else:
    print("[RESET] HTTP interface disabled (RESET_HTTP_ENABLE=0).")

dt = 1.0 / max(1.0, IDLE_FPS)
try:
    while True:
        simulation_app.update();
        _process_reset_requests()
        time.sleep(dt)
except KeyboardInterrupt:
    pass
finally:
    try: simulation_app.close()
    except Exception: pass
    print("[EXIT] Closed.")

# ----------------------------------------------------------
_auto_enable_flag = any(os.getenv(v, "0").lower() in ("1","true","yes") for v in [
    "AUTO_ENABLE_ALL_EXTENSIONS","TO_ENABLE_ALL_EXTENSIONS","ENABLE_ALL_EXTENSIONS"
])
if _auto_enable_flag:
    try:
        import omni.kit.app
        app_obj = omni.kit.app.get_app()
        # Extension manager access may vary depending on Isaac Sim version.
        ext_mgr = None
        if app_obj:
            # New API attempt
            if hasattr(app_obj, 'get_extension_manager'):
                try:
                    ext_mgr = app_obj.get_extension_manager()
                except Exception:
                    ext_mgr = None
        if ext_mgr is None:
          
            try:
                import omni.ext
                ext_mgr = getattr(omni.ext, 'get_extension_manager', lambda: None)()
            except Exception:
                ext_mgr = None
        if ext_mgr is None:
            print("\033[31m[AUTO_ENABLE] Extension manager not accessible. API may have changed in this version.\033[0m")
            raise RuntimeError("No extension manager")

        # Extensions dict veya list alma
        try:
            all_exts_dict = ext_mgr.get_extensions()  # name->meta (most versions)
            all_exts = list(all_exts_dict.keys())
        except Exception:
            # Alternative: enabled + disabled combination estimated APIs
            all_exts = []
            for attr in ("get_enabled_extension_ids","get_disabled_extension_ids"):
                if hasattr(ext_mgr, attr):
                    try:
                        all_exts.extend(getattr(ext_mgr, attr)())
                    except Exception:
                        pass
            all_exts = list(set(all_exts))

        exclude_raw = os.getenv("AUTO_ENABLE_EXT_EXCLUDE","")
        exclude = {x.strip() for x in exclude_raw.split(',') if x.strip()}
        prefix = os.getenv("AUTO_ENABLE_EXT_PREFIX","").strip()
        dump_path = os.getenv("AUTO_ENABLE_EXT_DUMP","" ).strip()

        force_first = ["omni.isaac.ros2_bridge"]
        # Additional security/simplification modes
        SAFE_RENDER_MODE = os.getenv("SAFE_RENDER_MODE","0").lower() in ("1","true","yes")
        if SAFE_RENDER_MODE:
            # Reduce render load (try to disable some key settings)
            try:
                import carb.settings
                _s = carb.settings.get_settings()
                for k,v in [
                    ("/rtx/enabled", False),
                    ("/ngx/enabled", False),
                    ("/app/renderer/resolution/width", 640),
                    ("/app/renderer/resolution/height", 360),
                    ("/app/renderer/vsync", False),
                    ("/app/hydraEngine/hydraEngineDelegate", "Storm"),  # Basic instead of RTX
                ]:
                    try: _s.set(k,v)
                    except Exception: pass
                
            except Exception:
                pass

        # Piece by piece instead of bulk enable (reduce GPU spike)
        batch_enable = os.getenv("BATCH_ENABLE_EXT","0").lower() in ("1","true","yes")
        batch_size = int(os.getenv("EXT_ENABLE_BATCH_SIZE","20")) if batch_enable else None
        batch_sleep = float(os.getenv("EXT_ENABLE_BATCH_SLEEP","0.5")) if batch_enable else 0.0

        diag_ext = os.getenv("DIAG_EXT_ENABLE","0").lower() in ("1","true","yes")
        def _gpu_mem_snapshot():
            if not diag_ext:
                return None
            try:
                import subprocess
                out = subprocess.check_output([
                    'nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'
                ], stderr=subprocess.DEVNULL, timeout=1).decode().strip().splitlines()
                return [int(x) for x in out]
            except Exception:
                return None
        ordered = []
        for name in force_first + all_exts:
            if name not in ordered:
                ordered.append(name)

        newly_enabled, already_enabled, skipped = [], [], []

        # Helpers for different API variants
        def _is_enabled(name):
            for m in ("is_extension_enabled","is_enabled"):
                if hasattr(ext_mgr, m):
                    try:
                        return bool(getattr(ext_mgr, m)(name))
                    except Exception:
                        continue
            return False
        def _enable(name):
            # Try order: set_extension_enabled_immediate, set_extension_enabled
            for m in ("set_extension_enabled_immediate","set_extension_enabled"):
                if hasattr(ext_mgr, m):
                    try:
                        getattr(ext_mgr, m)(name, True)
                        return True
                    except Exception:
                        continue
            return False

        current_batch_count = 0
        for ext_name in ordered:
            if prefix and not ext_name.startswith(prefix):
                skipped.append((ext_name, "prefix"))
                continue
            if ext_name in exclude:
                skipped.append((ext_name, "excluded"))
                continue
            try:
                if not _is_enabled(ext_name):
                    pre_gpu = _gpu_mem_snapshot()
                    if _enable(ext_name):
                        newly_enabled.append(ext_name)
                        if diag_ext:
                            post_gpu = _gpu_mem_snapshot()
                            if pre_gpu and post_gpu:
                                print(f"\033[36m[EXT_DIAG] {ext_name} mem_before={pre_gpu} mem_after={post_gpu}\033[0m")
                    else:
                        skipped.append((ext_name, "enable_fail"))
                else:
                    already_enabled.append(ext_name)
            except Exception:
                skipped.append((ext_name, "error"))

            if batch_enable:
                current_batch_count += 1
                if current_batch_count >= batch_size:
                    current_batch_count = 0
                    # Small wait and a few updates to give GPU a breather
               
                    for _ in range(3):
                        simulation_app.update(); time.sleep(0.01)
                    time.sleep(batch_sleep)

        if newly_enabled:
            print("\033[36m[AUTO_ENABLE] First 10: \033[0m" + ", ".join(newly_enabled[:10]))
        if skipped and len(skipped) < 20:
            print("\033[33m[AUTO_ENABLE] Skipped: \033[0m" + ", ".join(f"{n}:{r}" for n,r in skipped))
        if dump_path:
            try:
                import json
                with open(dump_path,'w') as f:
                    json.dump({
                        'newly_enabled': newly_enabled,
                        'already_enabled': already_enabled,
                        'skipped': skipped,
                        'exclude': list(exclude),
                        'prefix': prefix,
                    }, f, indent=2)
                print(f"\033[36m[AUTO_ENABLE] Dump: {dump_path}\033[0m")
            except Exception as e:
                print(f"\033[31m[AUTO_ENABLE] Dump cannot be written: {e}\033[0m")
        # stabilize a few frames
        warm = min(25, 5 + len(newly_enabled)//4)
        for _ in range(warm):
            simulation_app.update(); time.sleep(0.01)
    except Exception as e:
        print(f"\033[31mAUTO_ENABLE_ALL_EXTENSIONS failed: {e}\033[0m")

# ----------------------------------------------------------

import atexit, datetime, traceback, subprocess, json

DIAG_ENABLED = os.getenv("DIAGNOSTIC", "0").lower() in ("1", "true", "yes")
if DIAG_ENABLED:
   

    def _diag_now():
        return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def _read_proc_mem():
    data = {}
    try:
        with open('/proc/self/status','r') as f:
            for line in f:
                if line.startswith(('VmRSS','VmSize','VmData','VmSwap')):
                    k,v = line.split(':',1)
                    data[k.strip()] = v.strip()
    except Exception:
        pass
    return data

def _gpu_query():
    if not (os.getenv('DIAG_GPU','0').lower() in ('1','true','yes')):
        return None
    try:
        out = subprocess.check_output([
            'nvidia-smi','--query-gpu=timestamp,name,index,memory.total,memory.used,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'
        ], stderr=subprocess.DEVNULL, timeout=2).decode().strip().splitlines()
        gpus = []
        for line in out:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                gpus.append({
                    'name': parts[1],
                    'index': int(parts[2]),
                    'mem_total_mb': int(parts[3]),
                    'mem_used_mb': int(parts[4]),
                    'util_gpu_pct': int(parts[5]),
                    'util_mem_pct': int(parts[6])
                })
        return gpus
    except Exception:
        return None

_diag_log_path = os.getenv('DIAG_LOG_PATH', '/tmp/isaac_diag.log')
_diag_interval = float(os.getenv('DIAG_INTERVAL_SEC','10') or 10)
_diag_oom_threshold_gb = None
try:
    if 'DIAG_OOM_THRESHOLD_GB' in os.environ:
        _diag_oom_threshold_gb = float(os.getenv('DIAG_OOM_THRESHOLD_GB',''))
except Exception:
    _diag_oom_threshold_gb = None

_diag_last_time = 0.0
_diag_started = time.time()

def _diag_snap(tag, extra=None):
    if not DIAG_ENABLED:
        return
    snap = {
        'ts': _diag_now(),
        'uptime_s': round(time.time() - _diag_started,2),
        'tag': tag,
        'proc_mem': _read_proc_mem(),
    }
    gpu = _gpu_query()
    if gpu is not None:
        snap['gpus'] = gpu
    if extra:
        snap.update(extra)
    # Memory threshold warning
    if _diag_oom_threshold_gb is not None:
        try:
            rss_kb = 0
            vmrss = snap['proc_mem'].get('VmRSS')
            if vmrss:
                rss_kb = int(vmrss.split()[0])
            rss_gb = rss_kb / 1024 / 1024
            if rss_gb >= _diag_oom_threshold_gb:
                snap['oom_warning'] = f"RSS {rss_gb:.2f} GB >= threshold {_diag_oom_threshold_gb} GB"
        except Exception:
            pass
    line = json.dumps(snap, ensure_ascii=False)
    try:
        with open(_diag_log_path,'a') as f:
            f.write(line+'\n')
    except Exception:
        print(f"\033[31m[DIAG] Could not write log: {_diag_log_path}\033[0m")
    if snap.get('oom_warning'):
        print(f"\033[31m[DIAG] {snap['oom_warning']}\033[0m")

def _diag_periodic():
    global _diag_last_time
    if not DIAG_ENABLED:
        return
    now = time.time()
    if now - _diag_last_time >= _diag_interval:
        _diag_last_time = now
        _diag_snap('periodic')

def _diag_thread():
    while DIAG_ENABLED:
        _diag_periodic()
        time.sleep(1)

if DIAG_ENABLED:
    threading.Thread(target=_diag_thread, daemon=True).start()
    _diag_snap('startup_pre_stage')

# Now safe to import pxr / omni
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics  # noqa: E402
import omni  # noqa: E402


# Open the pushT.usd stage
stage_opened = False
try:
    usd_ctx = omni.usd.get_context()
    usd_path = os.environ['HOME'] + "/ur5_push_T-main/pushT.usd"

    if usd_ctx.get_stage() is None:
        usd_ctx.open_stage(usd_path)
        def _stage_loading(uc):
            """Robust check for stage loading across Isaac Sim versions."""
            try:
                return uc.is_stage_loading()
            except AttributeError:
                try:
                    from omni.usd import StageLoadingStatus  # noqa: E402
                    status = uc.get_stage_loading_status()
                    return status not in (StageLoadingStatus.COMPLETE, StageLoadingStatus.FAILED)
                except Exception:
                    return False
        max_wait_s = 30.0
        poll_dt = 0.05
        waited = 0.0
        while True:
            stage = usd_ctx.get_stage()
            if stage is not None and not _stage_loading(usd_ctx):
                break
            simulation_app.update()
            time.sleep(poll_dt)
            waited += poll_dt
            if waited >= max_wait_s:
                print("\033[33mStage load timeout reached; continuing anyway.\033[0m")
                break
        for _ in range(5):
            simulation_app.update()
        if usd_ctx.get_stage() is None:
            print(f"\033[31mFailed to open Isaac Sim stage (None returned): {usd_path}\033[0m")
        else:
            print(f"\033[32mIsaac Sim stage opened: {usd_path}\033[0m")
            stage_opened = True
            _diag_snap('stage_opened')
    else:
        print("\033[36mStage already open from earlier block; skipping second open.\033[0m")
        stage_opened = True
except Exception as e:
    print(f"\033[31mFailed to (re)open Isaac Sim stage: {e}\033[0m")

# ----------------------------------------------------------
# SCENE_ONLY mode: Only open scene and wait
# Environment variable: SCENE_ONLY=1
# ----------------------------------------------------------
SCENE_ONLY = os.getenv("SCENE_ONLY", "0").lower() in ("1", "true", "yes")
if SCENE_ONLY:
    print("\033[36mSCENE_ONLY mode active. RL / ROS / Policy will not be loaded. Optimizing for lower GPU load...\033[0m")
    # Fine tuning with environment variables
    try:
        scene_idle_fps = float(os.getenv("SCENE_IDLE_FPS", "10"))  # default 10 FPS
    except ValueError:
        scene_idle_fps = 10.0
    scene_static = os.getenv("SCENE_STATIC", "0").lower() in ("1","true","yes")
    try:
        scene_autoclose_sec = float(os.getenv("SCENE_AUTOCLOSE_SEC", "0"))  # 0 => disabled
    except ValueError:
        scene_autoclose_sec = 0.0
    warmup_frames = int(os.getenv("SCENE_WARMUP_FRAMES", "60"))
    disable_render = os.getenv("SCENE_DISABLE_RENDER", "0").lower() in ("1","true","yes")

    if disable_render:
        print("\033[33mSCENE_DISABLE_RENDER=1: Trying to disable render.\033[0m")
        try:
            import carb.settings as _cs
            _s = _cs.get_settings()
            # Attempt to disable some common settings (ignore if error)
            for key, val in [
                ("/app/renderer/enabled", False),
                ("/rtx/enabled", False),
                ("/ngx/enabled", False),
                ("/omni/kit/renderer/clearColor/enable", False),
            ]:
                try:
                    _s.set(key, val)
                except Exception:
                    pass
        except Exception as e:
            print(f"\033[31mRender disable failed: {e}\033[0m")

    idle_dt = 1.0 / max(0.5, scene_idle_fps)
    start_time = time.time()
    frames = 0
    last_diag = 0.0
    print(f"\033[36m[SCENE_ONLY] idle_fps={scene_idle_fps} static={scene_static} warmup_frames={warmup_frames} autoclose={scene_autoclose_sec}s\033[0m")
    if scene_static:
        print("\033[36m[SCENE_ONLY] STATIC mode: update frequency will be reduced after warmup.\033[0m")
    try:
        while True:
            now = time.time()
            # Auto close
            if scene_autoclose_sec > 0 and (now - start_time) >= scene_autoclose_sec:
                print(f"\033[32m[SCENE_ONLY] Auto close time reached ({scene_autoclose_sec}s).\033[0m")
                break
            # First warmup_frames update at full speed
            if frames < warmup_frames:
                simulation_app.update()
                frames += 1
                continue
            if scene_static:
                # Static mode: Reduce GPU load. Still update occasionally.
                if frames % 120 == 0:  # ~ every 12s @10fps
                    simulation_app.update()
                else:
                    time.sleep(0.5)
                frames += 1
            else:
                simulation_app.update()
                frames += 1
                time.sleep(idle_dt)
            # Light periodic log
            if now - last_diag > 30:
                last_diag = now
                print(f"\033[36m[SCENE_ONLY] frames={frames} uptime={now-start_time:.1f}s\033[0m")
    except KeyboardInterrupt:
        print("\033[33m[SCENE_ONLY] Stopped by user.\033[0m")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    raise SystemExit(0)

# ----------------------------------------------------------
# Staged Startup
# 1. USD scene loaded (above) and a few frames stabilized
# 2. Physics / timeline PLAY is executed
# 3. RL (policy action generation) is activated with delay
# Environment Variables:
#   SIM_AUTO_PLAY=1 (default 1) => timeline.play()
#   RL_AUTOSTART=1 (default 1) => RL activation is planned
#   RL_START_DELAY_SEC=N (default 5) => RL wait after timeline PLAY
# ----------------------------------------------------------
SIM_AUTO_PLAY = os.getenv("SIM_AUTO_PLAY", "1").lower() in ("1", "true", "yes")
RL_AUTOSTART = os.getenv("RL_AUTOSTART", "1").lower() in ("1", "true", "yes")
RL_WAIT_FOR_PLAY = os.getenv("RL_WAIT_FOR_PLAY", "1").lower() in ("1", "true", "yes")
HEAVY_INIT_WAIT_FOR_PLAY = os.getenv("HEAVY_INIT_WAIT_FOR_PLAY", "1").lower() in ("1", "true", "yes")
FORCE_HEAVY_INIT = os.getenv("FORCE_HEAVY_INIT", "0").lower() in ("1", "true", "yes")
TIMELINE_DEBUG = os.getenv("TIMELINE_DEBUG", "0").lower() in ("1", "true", "yes")
try:
    RL_START_DELAY_SEC = float(os.getenv("RL_START_DELAY_SEC", "5"))
except ValueError:
    RL_START_DELAY_SEC = 5.0

# RL activity flag (action publishing + finetune trigger) – initially off
rl_active = False

def _activate_rl():
    """Activate RL loop only if policy & heavy imports are ready."""
    global rl_active, policy, bridge
    if rl_active:
        return
    if bridge is None:
        print("\033[33mRL could not be activated: heavy imports not yet completed.\033[0m")
        return
    if policy is None:
        print("\033[33mRL could not be activated: policy not yet loaded.\033[0m")
        return
    rl_active = True
    print("\033[36mRL action publishing ACTIVATED.\033[0m")
    _diag_snap('rl_activated')

def _schedule_rl_activation():
    # If manual PLAY of timeline is expected, defer here.
    if RL_WAIT_FOR_PLAY:
        print("\033[36mRL_WAIT_FOR_PLAY=1: RL activation will wait until user presses PLAY.\033[0m")
        return
    if not RL_AUTOSTART:
        print("\033[33mRL_AUTOSTART=0: RL action publishing disabled (enable manually).\033[0m")
        return
    delay = max(0.0, RL_START_DELAY_SEC)
    if delay == 0:
        _activate_rl()
        return
    import threading
    def _delayed():
        print(f"\033[36mRL will activate in {delay:.1f} seconds...\033[0m")
        target = time.time() + delay
        while time.time() < target:
            time.sleep(0.2)
        _activate_rl()
    threading.Thread(target=_delayed, daemon=True).start()

def _play_timeline_if_requested():
    if RL_WAIT_FOR_PLAY:
        # Manual PLAY expected, automatic timeline startup.
        print("\033[33mSIM timeline will not start automatically (RL_WAIT_FOR_PLAY=1).\033[0m")
        return
    if not SIM_AUTO_PLAY:
        print("\033[33mSIM_AUTO_PLAY=0: timeline PLAY will not be executed.\033[0m")
        return
    # Multi API compatibility
    try:
        import omni.timeline as _otl
        # First try modern interface
        if hasattr(_otl, 'get_timeline_interface'):
            tl_if = _otl.get_timeline_interface()
            if tl_if and not tl_if.is_playing():
                tl_if.play()
                print("\033[32mTimeline PLAY (interface) started.\033[0m")
                _diag_snap('timeline_play_started', {'method':'interface'})
                return
        # Legacy get_timeline fallback
        tl = getattr(_otl, 'get_timeline', lambda: None)()
        if tl and hasattr(tl, 'is_playing') and not tl.is_playing():
            tl.play()
            print("\033[32mTimeline PLAY (legacy) started.\033[0m")
            _diag_snap('timeline_play_started', {'method':'legacy'})
            return
    except Exception as e:
        print(f"\033[31mTimeline play error (omni.timeline): {e}\033[0m")

_last_play_state_log = 0.0

def _is_timeline_playing():
    """Detect timeline PLAY status. Don't spam errors if omni.kit.timeline is missing."""
    # Modern interface
    try:
        import omni.timeline as _otli
        if hasattr(_otli, 'get_timeline_interface'):
            tl_if = _otli.get_timeline_interface()
            if tl_if and tl_if.is_playing():
                return True
        # Legacy get_timeline
        tl = getattr(_otli, 'get_timeline', lambda: None)()
        if tl and hasattr(tl, 'is_playing') and tl.is_playing():
            return True
    except ModuleNotFoundError:
        if TIMELINE_DEBUG:
            print("\033[33mTimeline module not found (omni.timeline).\033[0m")
    except Exception as e:
        if TIMELINE_DEBUG:
            print(f"\033[33mTimeline access error: {e}\033[0m")
    # Heuristic: frame increment
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        global _last_frame_index
        current_frame = app.get_time().frame
        if '_last_frame_index' in globals():
            if current_frame is not None and _last_frame_index is not None and current_frame > _last_frame_index:
                return True
        _last_frame_index = current_frame
    except Exception:
        pass
    # Heuristik 2: settings play state (varsa)
    try:
        import carb.settings
        s = carb.settings.get_settings()
        # In some versions: /app/player/play returns bool
        if s and s.get_as_bool('/app/player/play'):
            return True
    except Exception:
        pass
    return False

DEFER_HEAVY_INIT = os.getenv("DEFER_HEAVY_INIT", "1").lower() in ("1", "true", "yes")
# EAGER_ALL=1 => Heavy import + policy + ROS definition targeted as soon as scene opens (without waiting for PLAY).
EAGER_ALL = os.getenv("EAGER_ALL", "0").lower() in ("1", "true", "yes")
# EAGER_IMPORTS_ONLY=1 => Load scene + heavy imports (+ optional policy); do not activate timeline / ROS / RL.
EAGER_IMPORTS_ONLY = os.getenv("EAGER_IMPORTS_ONLY", "0").lower() in ("1","true","yes")
EAGER_IMPORTS_ONLY_LOAD_POLICY = os.getenv("EAGER_IMPORTS_ONLY_LOAD_POLICY", "0").lower() in ("1","true","yes")

if stage_opened and (EAGER_ALL or not DEFER_HEAVY_INIT):
    # Heavy imports will be done in main section; here just give timeline / RL planning a chance.
    _play_timeline_if_requested()
    _schedule_rl_activation()
elif not stage_opened:
    print("\033[33mRL / timeline will not start without scene being opened.\033[0m")
else:
    print("\033[36mDEFER_HEAVY_INIT=1: Heavy (ML/ROS) imports will be done after scene loading is complete.\033[0m")

bridge = None  # init later after heavy imports

# Global stop flag for graceful shutdown
stop_event = threading.Event()

tool_pose_xy = [0.0, 0.0]  # tool(end effector) pose
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 240  # Reduced
vid_W = 320  # Reduced
wrist_camera_image = None
top_camera_image = None

###############################################
# ML / Policy imports AFTER stage is ready    #
###############################################

# Reduce potential incompat with Isaac-bundled typing_extensions by allowing external site-packages first.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")  # Avoid dynamo traces inside sim loop (stability)

###############################################
# Optional torchvision stub to avoid import   #
# issues with typing_extensions inside Isaac  #
###############################################
if os.getenv("LEROBOT_STUB_TORCHVISION", "0").lower() in ("1", "true"):
    import sys, types, torch, importlib.machinery  # noqa: E402

    # Root package stub
    tv = types.ModuleType("torchvision")
    tv.__path__ = []  # mark as namespace/package
    tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, is_package=True)

    # Common submodules expected by availability checks
    # Pre-create basic submodules
    for _sub in ["datasets", "io", "models", "ops", "utils"]:
        if _sub == "models":
            # models will get extra nested modules
            m_models = types.ModuleType("torchvision.models")
            m_models.__spec__ = importlib.machinery.ModuleSpec("torchvision.models", loader=None, is_package=True)
            sys.modules['torchvision.models'] = m_models
            setattr(tv, 'models', m_models)
        else:
            m = types.ModuleType(f"torchvision.{_sub}")
            m.__spec__ = importlib.machinery.ModuleSpec(f"torchvision.{_sub}", loader=None)
            setattr(tv, _sub, m)
            sys.modules[f"torchvision.{_sub}"] = m

    # models._utils with IntermediateLayerGetter
    models_utils = types.ModuleType("torchvision.models._utils")
    models_utils.__spec__ = importlib.machinery.ModuleSpec("torchvision.models._utils", loader=None)

    import torch.nn as _nn
    from collections import OrderedDict as _OD

    class IntermediateLayerGetter(_nn.Module):  # minimal version
        def __init__(self, model: _nn.Module, return_layers: dict):
            super().__init__()
            if not set(return_layers).issubset(name for name, _ in model.named_children()):
                # Fallback: allow nested names, but we won't validate fully
                pass
            self.model = model
            self.return_layers = dict(return_layers)
        def forward(self, x):
            out = _OD()
            for name, module in self.model.named_children():
                x = module(x)
                if name in self.return_layers:
                    out_name = self.return_layers[name]
                    out[out_name] = x
            return out

    models_utils.IntermediateLayerGetter = IntermediateLayerGetter
    sys.modules['torchvision.models._utils'] = models_utils
    # attach under models
    sys.modules['torchvision.models']._utils = models_utils  # type: ignore

    # ops.misc submodule with required lightweight layers
    import torch.nn as _nn2, torch.nn.functional as _F2  # noqa: E402
    ops_root = sys.modules.get('torchvision.ops')
    if ops_root is None:
        ops_root = types.ModuleType('torchvision.ops')
        ops_root.__spec__ = importlib.machinery.ModuleSpec('torchvision.ops', loader=None, is_package=True)
        sys.modules['torchvision.ops'] = ops_root

    misc_mod = types.ModuleType('torchvision.ops.misc')
    misc_mod.__spec__ = importlib.machinery.ModuleSpec('torchvision.ops.misc', loader=None)

    class FrozenBatchNorm2d(_nn2.Module):
        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.register_buffer('weight', torch.ones(num_features))
            self.register_buffer('bias', torch.zeros(num_features))
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        def forward(self, x):
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            rm = self.running_mean.reshape(1, -1, 1, 1)
            rv = self.running_var.reshape(1, -1, 1, 1)
            scale = w / (rv + self.eps).sqrt()
            bias = b - rm * scale
            return x * scale + bias

    class Conv2dNormActivation(_nn2.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, norm_layer=None, activation_layer=_nn2.ReLU, bias=None):
            if padding is None:
                padding = (kernel_size - 1) // 2
            layers = []
            layers.append(_nn2.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=(bias if bias is not None else norm_layer is None)))
            if norm_layer is not None:
                if norm_layer is FrozenBatchNorm2d:
                    layers.append(FrozenBatchNorm2d(out_channels))
                else:
                    layers.append(norm_layer(out_channels))
            if activation_layer is not None:
                layers.append(activation_layer())
            super().__init__(*layers)

    class Permute(_nn2.Module):
        def __init__(self, *dims):
            super().__init__()
            self.dims = dims
        def forward(self, x):
            return x.permute(self.dims)

    misc_mod.FrozenBatchNorm2d = FrozenBatchNorm2d
    misc_mod.Conv2dNormActivation = Conv2dNormActivation
    misc_mod.Permute = Permute
    sys.modules['torchvision.ops.misc'] = misc_mod
    setattr(ops_root, 'misc', misc_mod)

    # Optional minimal ResNet18 fallback (only if explicitly allowed)
    if os.getenv("ALLOW_RESNET18_FALLBACK", "0").lower() in ("1", "true"):
        import torch.nn as _nnr
        from torch import Tensor as _Tensor

        class BasicBlock(_nnr.Module):
            expansion = 1
            def __init__(self, in_planes, planes, stride=1, downsample=None):
                super().__init__()
                self.conv1 = _nnr.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = _nnr.BatchNorm2d(planes)
                self.relu = _nnr.ReLU(inplace=True)
                self.conv2 = _nnr.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = _nnr.BatchNorm2d(planes)
                self.downsample = downsample
            def forward(self, x: _Tensor) -> _Tensor:
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                out = self.relu(out)
                return out

        class ResNet(_nnr.Module):
            def __init__(self, block, layers, num_classes=1000):
                super().__init__()
                self.inplanes = 64
                self.conv1 = _nnr.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = _nnr.BatchNorm2d(64)
                self.relu = _nnr.ReLU(inplace=True)
                self.maxpool = _nnr.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.avgpool = _nnr.AdaptiveAvgPool2d((1, 1))
                self.fc = _nnr.Linear(512 * block.expansion, num_classes)
            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = _nnr.Sequential(
                        _nnr.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                        _nnr.BatchNorm2d(planes * block.expansion),
                    )
                layers = [block(self.inplanes, planes, stride, downsample)]
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))
                return _nnr.Sequential(*layers)
            def forward(self, x: _Tensor) -> _Tensor:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = _nnr.Flatten()(x)
                x = self.fc(x)
                return x

        def resnet18(pretrained=False, num_classes=1000, **kwargs):  # signature mimic
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            if pretrained:
                print("\033[33mWarning: Fallback resnet18 cannot load real pretrained weights.\033[0m")
            return model

        # Attach only if real attr is missing
        import types as _types
        models_root = sys.modules.get('torchvision.models')
        if models_root and not hasattr(models_root, 'resnet18'):
            setattr(models_root, 'resnet18', resnet18)
            print("\033[33mAttached fallback ResNet18 to torchvision.models (ALLOW_RESNET18_FALLBACK=1).\033[0m")

    # transforms submodule with minimal functionality
    transforms_mod = types.ModuleType("torchvision.transforms")
    transforms_mod.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)

    class _Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms)
        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class _ToTensor:
        def __call__(self, pic):
            import numpy as _np
            if isinstance(pic, torch.Tensor):
                return pic
            arr = _np.array(pic, copy=False)
            if arr.ndim == 2:  # grayscale
                t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
            else:
                t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return t

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean)[:, None, None]
            self.std = torch.tensor(std)[:, None, None]
        def __call__(self, t):
            return (t - self.mean) / (self.std + 1e-6)

    class _Resize:
        def __init__(self, size):
            self.size = size if isinstance(size, (tuple, list)) else (size, size)
        def __call__(self, t):
            import torch.nn.functional as F
            return F.interpolate(t.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)

    transforms_mod.Compose = _Compose
    transforms_mod.ToTensor = _ToTensor
    transforms_mod.Normalize = _Normalize
    transforms_mod.Resize = _Resize
    tv.transforms = transforms_mod

    # v2 submodule (torchvision.transforms.v2) stub
    import enum as _enum
    v2_mod = types.ModuleType("torchvision.transforms.v2")
    v2_mod.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.v2", loader=None)

    class InterpolationMode(_enum.Enum):
        NEAREST = 0
        BILINEAR = 2
    v2_mod.InterpolationMode = InterpolationMode

    # Base Transform class (minimal API)
    class Transform:
        def __call__(self, x):
            return x
    v2_mod.Transform = Transform

    # Wrap existing helpers as subclasses for compatibility
    class V2Compose(_Compose, Transform):
        pass
    class V2ToTensor(_ToTensor, Transform):
        pass
    class V2Normalize(_Normalize, Transform):
        pass
    class V2Resize(_Resize, Transform):
        pass

    v2_mod.Compose = V2Compose
    v2_mod.ToTensor = V2ToTensor
    v2_mod.Normalize = V2Normalize
    v2_mod.Resize = V2Resize

    # functional namespace
    func_mod = types.ModuleType("torchvision.transforms.v2.functional")
    func_mod.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.v2.functional", loader=None)
    import torch.nn.functional as _Fv2
    def resize(img, size, interpolation=InterpolationMode.BILINEAR):
        return _Fv2.interpolate(img.unsqueeze(0), size=size if isinstance(size, (tuple, list)) else (size, size), mode="bilinear", align_corners=False).squeeze(0)
    def to_tensor(img):
        return _ToTensor()(img)
    def normalize(t, mean, std):
        return _Normalize(mean, std)(t)
    func_mod.resize = resize
    func_mod.to_tensor = to_tensor
    func_mod.normalize = normalize
    v2_mod.functional = func_mod
    sys.modules['torchvision.transforms.v2.functional'] = func_mod

    sys.modules['torchvision.transforms.v2'] = v2_mod
    transforms_mod.v2 = v2_mod

    # Register modules
    sys.modules['torchvision'] = tv
    sys.modules['torchvision.transforms'] = transforms_mod

    print("\033[33mUsing enhanced torchvision stub (with IntermediateLayerGetter + ops.misc + transforms.v2) (LEROBOT_STUB_TORCHVISION=1).\033[0m")

policy = None
device = "cuda"
pretrained_policy_path = Path("~/lerobot/outputs/train/2025-09-26/11-37-52_act/checkpoints/012000/pretrained_model").expanduser()

# ----------------------------------------------------------
# Finetune confirmation (ask user before enabling updates)
# ----------------------------------------------------------
finetune_enabled = False
finetune_ready_at = float('inf')
_finetune_delay_notice_done = False
optimizer = None

# Finetune hyperparameters
FINETUNE_EVERY_STEPS = 100
FINETUNE_EPOCHS = 2
MINI_BATCH = 32
GAMMA = 0.99
SAVE_EVERY_UPDATES = 6
MAX_EPISODE_STEPS = 600

updates_done = 0
checkpoint_dir = Path("~/lerobot/outputs/rl_finetune").expanduser()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
LATEST_CKPT_NAME = "last_checkpoint.pt"

# Replay buffers
rb_states = []
rb_images = []
rb_actions = []
rb_rewards = []
rb_dones = []
start_buffer_index = 0

def reset_robot():
    node = Node('robot_reset')
    pub = node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
    
    traj_msg = JointTrajectory()
    traj_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point.time_from_start = Duration(sec=3)
    
    traj_msg.points = [point]
    pub.publish(traj_msg)
    print("\033[32mRobot moving to home position...\033[0m")

def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - float(d))
        returns.append(G)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

def save_checkpoint(tag: str):
    try:
        out = checkpoint_dir / LATEST_CKPT_NAME
        torch.save({
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tag": tag,
        }, out)
        print(f"\033[35mCheckpoint saved: {out} (tag={tag})\033[0m")
    except Exception as e:
        print(f"\033[31mFailed to save checkpoint: {e}\033[0m")

def load_checkpoint_by_tag(tag: str):
    try:
        path = checkpoint_dir / LATEST_CKPT_NAME if tag in ("latest", "last") else checkpoint_dir / f"policy_{tag}.pt"
        data = torch.load(path, map_location=device)
        policy.load_state_dict(data["state_dict"])
        optimizer.load_state_dict(data.get("optimizer", optimizer.state_dict()))
        policy.to(device)
        policy.train()
        print(f"\033[35mCheckpoint loaded: {path}\033[0m")
        return True
    except Exception as e:
        print(f"\033[31mFailed to load checkpoint '{tag}': {e}\033[0m")
        return False

def load_latest_checkpoint_if_exists():
    latest_path = checkpoint_dir / LATEST_CKPT_NAME
    if latest_path.exists():
        try:
            data = torch.load(latest_path, map_location=device)
            policy.load_state_dict(data["state_dict"])
            if "optimizer" in data:
                optimizer.load_state_dict(data["optimizer"])
            policy.to(device)
            policy.train()
            print(f"\033[35mResumed from latest checkpoint: {latest_path}\033[0m")
            return True
        except Exception as e:
            print(f"\033[31mFailed to load fixed latest checkpoint: {e}\033[0m")
    try:
        if not checkpoint_dir.exists():
            return False
        candidates = sorted(checkpoint_dir.glob("policy_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            try:
                data = torch.load(cand, map_location=device)
                policy.load_state_dict(data["state_dict"])
                if "optimizer" in data:
                    optimizer.load_state_dict(data["optimizer"])
                policy.to(device)
                policy.train()
                print(f"\033[35mResumed from checkpoint: {cand}\033[0m")
                return True
            except Exception as e:
                print(f"\033[31mFailed to load candidate {cand.name}: {e}\033[0m")
        return False
    except Exception as e:
        print(f"\033[31mFailed to scan checkpoints: {e}\033[0m")
        return False

def finetune_step():
    global _finetune_delay_notice_done
    if not finetune_enabled:
        return
    # Wait if startup delay is active
    if time.time() < finetune_ready_at:
        if not _finetune_delay_notice_done:
            remaining = int(finetune_ready_at - time.time())
            print(f"\033[33mFinetune will start in {remaining} seconds...\033[0m")
            _finetune_delay_notice_done = True
        return
    if len(rb_rewards) < 2:
        return
    states = torch.stack(rb_states).to(device)
    images = torch.stack(rb_images).to(device)
    actions = torch.stack(rb_actions).to(device)
    dones_t = torch.tensor(rb_dones, dtype=torch.float32, device=device)
    returns = compute_returns(rb_rewards, rb_dones, GAMMA)

    with torch.enable_grad():
        dataset_size = states.shape[0]
        idx = torch.randperm(dataset_size, device=device)
        for _ in range(FINETUNE_EPOCHS):
            for start in range(0, dataset_size, MINI_BATCH):
                batch_idx = idx[start:start+MINI_BATCH]
                batch_states = states[batch_idx]
                batch_images = images[batch_idx]
                target_actions = actions[batch_idx]
                adv = returns[batch_idx].unsqueeze(-1)

                batch_obs = {
                    "observation.state": batch_states,
                    "observation.image": batch_images,
                }

                with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                    out = policy(batch_obs)
                    pred_actions = out["action"] if isinstance(out, dict) and "action" in out else None
                    if pred_actions is None:
                        return
                    loss = ((pred_actions - target_actions)**2 * adv.abs()).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

    rb_states.clear()
    rb_images.clear()
    rb_actions.clear()
    rb_rewards.clear()
    rb_dones.clear()
    global start_buffer_index
    start_buffer_index = 0

    global updates_done
    updates_done += 1
    if updates_done % SAVE_EVERY_UPDATES == 0:
        save_checkpoint(f"update{updates_done}")

"""ROS2 node classes are defined after heavy imports + (optional) policy loading.

Initially we use None instead of placeholder; when _define_ros_and_policy_classes
is called, real Node subclasses are created. This way the script can be loaded
without importing rclpy at the top of the file."""
Get_End_Effector_Pose = None  # type: ignore
RewardPublisher = None  # type: ignore
Action_Publisher = None  # type: ignore

def _heavy_imports_only():
    """Perform heavy library imports; don't load policy weights yet."""
    global torch, optim, nn, rclpy, Node, Joy, Image, std_msgs, TFMessage, R, np, cv2, CvBridge, JointTrajectory, JointTrajectoryPoint, Duration
    global bridge, wrist_camera_image, top_camera_image
    if bridge is not None:
        return
    print("\033[36mStarting heavy imports (torch, rclpy, cv2) ...\033[0m")
    _diag_snap('heavy_import_start')
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    import torch.optim as optim  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import rclpy  # noqa: F401
    from rclpy.node import Node  # noqa: F401
    from sensor_msgs.msg import Joy, Image  # noqa: F401
    import std_msgs.msg as std_msgs  # noqa: F401
    from tf2_msgs.msg import TFMessage  # noqa: F401
    from scipy.spatial.transform import Rotation as R  # noqa: F401
    import cv2  # noqa: F401
    from cv_bridge import CvBridge  # noqa: F401
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: F401
    from builtin_interfaces.msg import Duration  # noqa: F401
    bridge = CvBridge()
    wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
    top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
    _diag_snap('heavy_import_complete')

def _load_policy_and_optimizer():
    global policy, finetune_enabled, finetune_ready_at, optimizer
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
    print("\033[36mLoading policy weights...\033[0m")
    _diag_snap('policy_load_start')
    try:
        _policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        _policy.to(device)
        _policy.train()
        policy = _policy
        print(f"\033[32mPolicy loaded: {pretrained_policy_path}\033[0m")
    except Exception as _e:
        raise RuntimeError(f"Failed to load ACTPolicy from {pretrained_policy_path}: {_e}")
    finetune_enabled = os.getenv("FINETUNE_ENABLE", "0").lower() in ("1", "true", "yes")
    if not finetune_enabled and os.getenv("AUTO_ACCEPT_FINETUNE"):
        print("\033[33mAUTO_ACCEPT_FINETUNE ignored; finetune only enabled with FINETUNE_ENABLE=1.\033[0m")
    print(f"\033[35mFinetune status (env FINETUNE_ENABLE): {'ON' if finetune_enabled else 'OFF'}\033[0m")
    finetune_ready_at = time.time() + 10 if finetune_enabled else float('inf')
    import torch.optim as _optim_local
    optimizer = _optim_local.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=1e-5)
    _diag_snap('policy_loaded')

def _define_ros_and_policy_classes():
    """Define real ROS2 Node classes (one-time)."""
    global Get_End_Effector_Pose, RewardPublisher, Action_Publisher
    if Get_End_Effector_Pose is not None and RewardPublisher is not None and Action_Publisher is not None:
        return True  # already defined
    if 'Node' not in globals():  # heavy imports not finished
        print("\033[33mNode class not available yet; ROS classes could not be defined.\033[0m")
        return False

    class Get_End_Effector_Pose(Node):
        def __init__(self):
            super().__init__('get_modelstate')
            self.subscription = self.create_subscription(
                TFMessage,
                '/isaac_tf',
                self.listener_callback,
                10)
            self.euler_angles = np.array([0.0, 0.0, 0.0], float)

        def listener_callback(self, data):
            global tool_pose_xy, tbar_pose_xyw
            tool_pose = data.transforms[0].transform.translation
            tool_pose_xy[0] = tool_pose.y
            tool_pose_xy[1] = tool_pose.x
            tbar_translation = data.transforms[1].transform.translation
            tbar_rotation = data.transforms[1].transform.rotation
            tbar_pose_xyw[0] = tbar_translation.y
            tbar_pose_xyw[1] = tbar_translation.x
            self.euler_angles[:] = R.from_quat([tbar_rotation.x, tbar_rotation.y, tbar_rotation.z, tbar_rotation.w]).as_euler('xyz', degrees=False)
            tbar_pose_xyw[2] = self.euler_angles[2]

    class RewardPublisher(Node):
        def __init__(self):
            super().__init__('rl_reward_publisher')
            self.pub = self.create_publisher(std_msgs.Float32, '/rl/reward', 10)
            self.done_pub = self.create_publisher(std_msgs.Bool, '/rl/done', 10)
        def publish(self, reward: float, done: bool):
            msg = std_msgs.Float32(data=float(reward))
            self.pub.publish(msg)
            self.done_pub.publish(std_msgs.Bool(data=bool(done)))

    class Action_Publisher(Node):
        def __init__(self):
            super().__init__('Joy_Publisher')
            self.declare_parameter('hz', 10)
            self.declare_parameter('success_threshold', 0.90)
            self.declare_parameter('finetune_every_steps', FINETUNE_EVERY_STEPS)
            self.declare_parameter('max_episode_steps', MAX_EPISODE_STEPS)
            self.declare_parameter('accuracy_floor', 0.17)
            self.declare_parameter('peak_min', 0.50)
            self.declare_parameter('drop_threshold', 0.26)
            self.declare_parameter('floor_warmup_steps', 30)
            self.declare_parameter('reward_bonus_threshold', 0.82)
            self.declare_parameter('reward_bonus_value', 0.5)
            self.declare_parameter('low_acc_penalty_threshold', 0.30)
            self.declare_parameter('low_acc_penalty_value', -0.1)
            self.declare_parameter('drop_stop_penalty_value', -0.5)
            self.declare_parameter('success_reward', 10.0)
            self.declare_parameter('timeout_penalty', -1.0)

            self.hz = int(self.get_parameter('hz').get_parameter_value().integer_value)
            self.success_threshold = float(self.get_parameter('success_threshold').get_parameter_value().double_value)
            self.finetune_every_steps = int(self.get_parameter('finetune_every_steps').get_parameter_value().integer_value)
            self.max_episode_steps = int(self.get_parameter('max_episode_steps').get_parameter_value().integer_value)
            self.accuracy_floor = float(self.get_parameter('accuracy_floor').get_parameter_value().double_value)
            self.peak_min = float(self.get_parameter('peak_min').get_parameter_value().double_value)
            self.drop_threshold = float(self.get_parameter('drop_threshold').get_parameter_value().double_value)
            self.floor_warmup_steps = int(self.get_parameter('floor_warmup_steps').get_parameter_value().integer_value)
            self.reward_bonus_threshold = float(self.get_parameter('reward_bonus_threshold').get_parameter_value().double_value)
            self.reward_bonus_value = float(self.get_parameter('reward_bonus_value').get_parameter_value().double_value)
            self.low_acc_penalty_threshold = float(self.get_parameter('low_acc_penalty_threshold').get_parameter_value().double_value)
            self.low_acc_penalty_value = float(self.get_parameter('low_acc_penalty_value').get_parameter_value().double_value)
            self.drop_stop_penalty_value = float(self.get_parameter('drop_stop_penalty_value').get_parameter_value().double_value)
            self.success_reward = float(self.get_parameter('success_reward').get_parameter_value().double_value)
            self.timeout_penalty = float(self.get_parameter('timeout_penalty').get_parameter_value().double_value)

            self.pub_joy = self.create_publisher(Joy, '/joy', 10)
            self.joy_commands = Joy()
            self.joy_commands.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            self.joy_commands.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.timer = self.create_timer(1/self.hz, self.timer_callback)

            self.initial_image = cv2.imread(os.environ['HOME'] + "/ur5_push_T-main/images/stand_top_plane.png")
            self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
            self.tool_radius = 10
            self.scale = 1.639344
            self.C_W = 182
            self.C_H = 152
            self.OBL1 = int(150/self.scale)
            self.OBL2 = int(120/self.scale)
            self.OBW = int(30/self.scale)
            self.radius = int(10/self.scale)
            self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)
            self.T_image = cv2.imread(os.environ['HOME'] + "/ur5_push_T-main/images/stand_top_plane_filled.png")
            self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
            thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
            self.blue_region = cv2.bitwise_not(img_th)
            self.blue_region_sum = cv2.countNonZero(self.blue_region)

            self.episode_reward = 0.0
            self.step_count = 0
            self.episode_index = 0
            self.best_accuracy = 0.0
            global start_buffer_index
            start_buffer_index = 0

        def _save_and_stop(self, reason: str):
            tag = f"stop_{reason}_ep{self.episode_index}_step{self.step_count}"
            save_checkpoint(tag)
            print(f"\033[33mStopping script due to: {reason}\033[0m")
            reset_robot()
            time.sleep(3.0)
            self.reset_simulation()
            stop_event.set()

        def timer_callback(self):
            global tool_pose_xy, tbar_pose_xyw, start_buffer_index
            if not rl_active:
                if RL_WAIT_FOR_PLAY and _is_timeline_playing():
                    if RL_AUTOSTART:
                        print("\033[36mTimeline PLAY detected. Starting RL activation...\033[0m")
                        if RL_START_DELAY_SEC <= 0:
                            _activate_rl()
                        else:
                            import threading
                            delay = RL_START_DELAY_SEC
                            def _delayed_rl():
                                print(f"\033[36mRL will activate in {delay:.1f} seconds (after PLAY).\033[0m")
                                target = time.time() + delay
                                while time.time() < target:
                                    time.sleep(0.2)
                                _activate_rl()
                            threading.Thread(target=_delayed_rl, daemon=True).start()
                    else:
                        print("\033[33mTimeline PLAY detected but RL_AUTOSTART=0, waiting for manual activation.\033[0m")
                else:
                    now = time.time()
                    global _last_play_state_log
                    if RL_WAIT_FOR_PLAY and (now - _last_play_state_log) > 5.0 and stage_opened:
                        print("\033[33mRL will start when PLAY is pressed... (RL_WAIT_FOR_PLAY=1)\033[0m")
                        _last_play_state_log = now
                return
            self.joy_commands.header.frame_id = "joy"
            self.joy_commands.header.stamp = self.get_clock().now().to_msg()

            base_image = copy.copy(self.initial_image)
            self.Tbar_region[:] = 0

            x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
            y = int((tool_pose_xy[1]*1000 - 320)/self.scale)
            cv2.circle(base_image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)

            x1 = tbar_pose_xyw[0]
            y1 = tbar_pose_xyw[1]
            th1 = -tbar_pose_xyw[2] - pi/2
            dx1 = -self.OBW/2*cos(th1 - pi/2)
            dy1 = -self.OBW/2*sin(th1 - pi/2)
            self.tbar1_ob = [
                [int(cos(th1)*self.OBL1/2 - sin(th1)*self.OBW/2 + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2 + cos(th1)*self.OBW/2 + dy1 + (1000*y1-320)/self.scale)],
                [int(cos(th1)*self.OBL1/2 - sin(th1)*(-self.OBW/2) + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2 + cos(th1)*(-self.OBW/2) + dy1 + (1000*y1-320)/self.scale)],
                [int(cos(th1)*(-self.OBL1/2) - sin(th1)*(-self.OBW/2) + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*(-self.OBW/2) + dy1 + (1000*y1-320)/self.scale)],
                [int(cos(th1)*(-self.OBL1/2) - sin(th1)*self.OBW/2 + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*self.OBW/2 + dy1 + (1000*y1-320)/self.scale)]
            ]
            pts1_ob = np.array(self.tbar1_ob, np.int32)
            cv2.fillPoly(base_image, [pts1_ob], (0, 0, 180))
            cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)

            th2 = -tbar_pose_xyw[2] - pi
            dx2 = self.OBL2/2*cos(th2)
            dy2 = self.OBL2/2*sin(th2)
            self.tbar2_ob = [
                [int(cos(th2)*self.OBL2/2 - sin(th2)*self.OBW/2 + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2 + cos(th2)*self.OBW/2 + dy2 + (1000*y1-320)/self.scale)],
                [int(cos(th2)*self.OBL2/2 - sin(th2)*(-self.OBW/2) + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2 + cos(th2)*(-self.OBW/2) + dy2 + (1000*y1-320)/self.scale)],
                [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2) + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2) + dy2 + (1000*y1-320)/self.scale)],
                [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2 + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2 + dy2 + (1000*y1-320)/self.scale)]
            ]
            pts2_ob = np.array(self.tbar2_ob, np.int32)
            cv2.fillPoly(base_image, [pts2_ob], (0, 0, 180))
            cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)

            cv2.circle(base_image, center=(int(self.C_W + 1000*x1/self.scale), int((1000*y1-320)/self.scale)), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)

            img_msg = bridge.cv2_to_imgmsg(base_image)
            self.pub_img.publish(img_msg)

            common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
            common_part_sum = cv2.countNonZero(common_part)
            accuracy = common_part_sum/self.blue_region_sum if self.blue_region_sum > 0 else 0.0
            if self.step_count % 10 == 0:
                print(f"\033[32mstep {self.step_count} | ep {self.episode_index} | accuracy(reward): {accuracy:.3f}\033[0m")

            if self.step_count >= self.floor_warmup_steps and accuracy < self.accuracy_floor:
                self._save_and_stop(reason=f"floor_{self.accuracy_floor:.2f}")
                return

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
            if self.best_accuracy >= self.peak_min and (self.best_accuracy - accuracy) >= self.drop_threshold:
                penalty = -abs(self.drop_stop_penalty_value)
                reward_node.publish(penalty, True)
                self.episode_reward += penalty
                self._save_and_stop(reason=f"drop_{self.drop_threshold:.2f}_from_{self.best_accuracy:.3f}_to_{accuracy:.3f}")
                return

            state_t = torch.from_numpy(np.array(tool_pose_xy)).to(torch.float32)
            image_t = torch.from_numpy(base_image).to(torch.float32) / 255
            image_t = image_t.permute(2, 0, 1)
            state_t = state_t.to(device).unsqueeze(0)
            image_t = image_t.to(device).unsqueeze(0)
            obs_t = {
                "observation.state": state_t,
                "observation.image": image_t,
            }

            with torch.no_grad():
                action_t = policy.select_action(obs_t)
            numpy_action = action_t.squeeze(0).to("cpu").numpy()
            numpy_action = np.clip(numpy_action, -1.0, 1.0)

            reward_value = accuracy + (self.reward_bonus_value if accuracy >= self.reward_bonus_threshold else 0.0)
            reward_value = min(1.0, reward_value)
            if accuracy < self.low_acc_penalty_threshold:
                reward_value -= abs(self.low_acc_penalty_value)
            if accuracy >= self.success_threshold:
                reward_value += self.success_reward

            self.joy_commands.axes[0] = float(numpy_action[0])
            self.joy_commands.axes[1] = float(numpy_action[1])
            self.pub_joy.publish(self.joy_commands)

            rb_states.append(state_t.squeeze(0).detach().to("cpu"))
            rb_images.append(image_t.squeeze(0).detach().to("cpu"))
            rb_actions.append(action_t.squeeze(0).detach().to("cpu"))
            rb_rewards.append(float(reward_value))
            rb_dones.append(bool(accuracy >= self.success_threshold))

            reward_node.publish(reward_value, bool(accuracy >= self.success_threshold))

            self.episode_reward += float(reward_value)
            self.step_count += 1

            if self.step_count % self.finetune_every_steps == 0:
                if finetune_enabled:
                    try:
                        print("\033[36mStarting finetune step...\033[0m")
                        finetune_step()
                        print("\033[36mFinetune step done.\033[0m")
                    except Exception as e:
                        print(f"\033[31mFinetune failed: {e}\033[0m")
                else:
                    if self.step_count == self.finetune_every_steps:
                        print("\033[33mFinetune disabled. You can enable it with FINETUNE_ENABLE=1.\033[0m")

            done = (accuracy >= self.success_threshold) or (self.step_count >= self.max_episode_steps)
            if done:
                status = 'SUCCESS' if accuracy >= self.success_threshold else 'TIMEOUT'
                if status == 'SUCCESS':
                    success_reward = self.success_reward
                    rb_rewards.append(success_reward)
                    self.episode_reward += success_reward
                    reward_node.publish(success_reward, True)
                    print(f"\033[32mSUCCESS! Final reward: {success_reward}\033[0m")
                if status == 'TIMEOUT':
                    timeout_penalty = self.timeout_penalty
                    rb_rewards.append(timeout_penalty)
                    self.episode_reward += timeout_penalty
                    reward_node.publish(timeout_penalty, True)
                    print(f"\033[31mTIMEOUT! Penalty: {timeout_penalty}\033[0m")

                self.episode_reward = 0.0
                self.step_count = 0
                self.episode_index += 1
                self.best_accuracy = 0.0
                start_buffer_index = len(rb_rewards)
                if status == 'SUCCESS':
                    save_checkpoint(f"ep{self.episode_index}_success")
                reset_robot()
                time.sleep(3.0)
                self.reset_simulation()

        def reset_simulation(self):
            global tool_pose_xy, tbar_pose_xyw, rb_states, rb_images, rb_actions, rb_rewards, rb_dones, start_buffer_index
            try:
                stage = Usd.Stage.Open("~/ur5_push_T-main/pushT.usd", load=Usd.Stage.LoadAll)
                if not stage:
                    print("\033[31mError: USD scene could not be opened.\033[0m")
                    return
                main_prim = stage.GetPrimAtPath("/Main")
                if main_prim.IsValid():
                    xform = UsdGeom.Xformable(main_prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                    xform.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                    print("\033[32m/Main prim position reset.\033[0m")
                    for prim in stage.Traverse():
                        if prim.GetParent() == main_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
                            rigid_body_api.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
                            rigid_body_api.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
                            print(f"\033[32m{prim.GetPath()} physics properties reset.\033[0m")
                else:
                    print("\033[31mError: /Main prim not found.\033[0m")
                    return
                root_prim = stage.GetPrimAtPath("/")
                if root_prim.HasProperty("physics:rigidBodyEnabled"):
                    root_prim.RemoveProperty("physics:rigidBodyEnabled")
                    print("\033[32mInvalid physics:rigidBodyEnabled removed from root prim.\033[0m")
                for prim_path in ["/World/simple_table/visuals", "/World/simple_table/collisions"]:
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI):
                        collision_api = UsdPhysics.CollisionAPI(prim)
                        collision_api.CreateCollisionEnabledAttr(True)
                        collision_api.CreateApproximationAttr("convexHull")
                        print(f"\033[32mCollision approximation set to 'convexHull' for {prim_path}.\033[0m")
                table_prim = stage.GetPrimAtPath("/World/simple_table")
                if table_prim.IsValid() and table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    table_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    print("\033[32mRigidBodyAPI removed from /World/simple_table prim (made static).\033[0m")
                root_layer = stage.GetRootLayer()
                if not root_layer.Export(root_layer.identifier):
                    print("\033[31mError: USD scene could not be saved: {}\033[0m".format(root_layer.identifier))
                    return
                print("\033[32mSimulation prims reset: /Main.\033[0m")
                main_prim = stage.GetPrimAtPath("/Main")
                if main_prim.IsValid():
                    xform = UsdGeom.Xformable(main_prim)
                    transform = xform.GetLocalTransformation()
                    translation = transform.ExtractTranslation()
                    rotation = transform.ExtractRotation().GetQuat()
                    print(f"\033[32m/Main position: {translation}, orientation: {rotation}\033[0m")
                else:
                    print("\033[31m/Main prim invalid, position could not be verified.\033[0m")
                print("\033[33mPlease reload the scene in Isaac Sim interface (File > Reload or Stop/Play)!\033[0m")
            except Exception as e:
                print(f"\033[31mUSD reset error: {e}\033[0m")
            tool_pose_xy = [0.0, 0.0]
            tbar_pose_xyw = [0.0, 0.0, 0.0]
            rb_states.clear()
            rb_images.clear()
            rb_actions.clear()
            rb_rewards.clear()
            rb_dones.clear()
            start_buffer_index = 0
            time.sleep(3.0)

    # Assign to globals (finalize)
    Get_End_Effector_Pose.__name__ = 'Get_End_Effector_Pose'
    RewardPublisher.__name__ = 'RewardPublisher'
    Action_Publisher.__name__ = 'Action_Publisher'
    globals()['Get_End_Effector_Pose'] = Get_End_Effector_Pose
    globals()['RewardPublisher'] = RewardPublisher
    globals()['Action_Publisher'] = Action_Publisher
    print("\033[36mROS2 Node classes defined.\033[0m")
    return True

if __name__ == '__main__':
    # Ultra-minimal diagnostic branch to isolate crashes before heavy logic.
    if os.getenv('MINIMAL_TEST','0').lower() in ('1','true','yes'):
        frames = int(os.getenv('MINIMAL_FRAMES','300'))
        print(f"\033[36m[MINIMAL_TEST] Running {frames} frames (no ROS, no policy, no finetune).\033[0m")
        for i in range(frames):
            simulation_app.update()
            time.sleep(0.01)
            if i % 60 == 0:
                print(f"[MINIMAL_TEST] frame {i}")
        try:
            simulation_app.close()
        except Exception:
            pass
        print("\033[32m[MINIMAL_TEST] Completed without internal logic. Exiting.\033[0m")
        raise SystemExit(0)
    POLICY_LOAD_AFTER_IMPORT = os.getenv('POLICY_LOAD_AFTER_IMPORT','1').lower() in ('1','true','yes')
    try:
        POLICY_LOAD_DELAY_SEC = float(os.getenv('POLICY_LOAD_DELAY_SEC','0'))
    except ValueError:
        POLICY_LOAD_DELAY_SEC = 0.0
    # EAGER_IMPORTS_ONLY mode (early exit): only heavy import + optional policy
    if 'EAGER_IMPORTS_ONLY' in globals() and EAGER_IMPORTS_ONLY:
        if not stage_opened:
            print("\033[31mEAGER_IMPORTS_ONLY: Scene could not be opened. Exiting.\033[0m")
            raise SystemExit(5)
        print("\033[36m[EAGER_IMPORTS_ONLY] Loading heavy imports...\033[0m")
        if bridge is None:
            _heavy_imports_only()
        if EAGER_IMPORTS_ONLY_LOAD_POLICY:
            if policy is None:
                try:
                    _load_policy_and_optimizer()
                except Exception as e:
                    print(f"\033[31mPolicy could not be loaded: {e}\033[0m")
        else:
            print("\033[33m[EAGER_IMPORTS_ONLY] Policy not loaded.\033[0m")
        print("\033[32m[EAGER_IMPORTS_ONLY] Preparation complete. RL/ROS not activated. Exit with CTRL+C.\033[0m")
        try:
            while True:
                simulation_app.update()
                time.sleep(1/30)
        except KeyboardInterrupt:
            pass
        finally:
            try: simulation_app.close()
            except Exception: pass
        raise SystemExit(0)
    if DEFER_HEAVY_INIT and not EAGER_ALL:
        if not stage_opened:
            print("\033[31mScene not loaded; cannot perform heavy imports.\033[0m")
            raise SystemExit(1)
        if HEAVY_INIT_WAIT_FOR_PLAY and not FORCE_HEAVY_INIT:
            print("\033[36mHEAVY_INIT_WAIT_FOR_PLAY=1: Waiting for PLAY (first import, then policy).\033[0m")
            _last_msg = 0.0
            _first_detect = True
            wait_start = time.time()
            max_wait_frames = int(os.getenv('HEAVY_INIT_MAX_WAIT_FRAMES', '0'))
            frame_counter = 0
            while not _is_timeline_playing():
                simulation_app.update()
                now = time.time()
                if TIMELINE_DEBUG and _first_detect:
                    print("\033[36m[TimelineDebug] Initial check performed; not PLAY.\033[0m")
                    _first_detect = False
                if now - _last_msg > 5.0:
                    elapsed = now - wait_start
                    print(f"\033[33mWaiting for PLAY... (elapsed: {elapsed:.1f}s)\033[0m")
                    _last_msg = now
                frame_counter += 1
                max_wait = float(os.getenv('HEAVY_INIT_MAX_WAIT_SEC','0'))
                if max_wait > 0 and (now - wait_start) >= max_wait:
                    print(f"\033[33mMaximum wait ({max_wait}s) exceeded, continuing.\033[0m")
                    break
                if max_wait_frames > 0 and frame_counter >= max_wait_frames:
                    print(f"\033[33mMaximum frame wait ({max_wait_frames}) exceeded, continuing.\033[0m")
                    break
                time.sleep(1/60)
            else:
                print("\033[32mPLAY detected.\033[0m")
        elif FORCE_HEAVY_INIT and HEAVY_INIT_WAIT_FOR_PLAY:
            print("\033[33mFORCE_HEAVY_INIT=1: Continuing without waiting for PLAY detection.\033[0m")
        # Stage 1: heavy imports
        _heavy_imports_only()
        # Opsiyonel gecikme
        if POLICY_LOAD_AFTER_IMPORT and POLICY_LOAD_DELAY_SEC > 0:
            print(f"\033[36mPolicy loading delayed by {POLICY_LOAD_DELAY_SEC:.1f} seconds...\033[0m")
            target = time.time() + POLICY_LOAD_DELAY_SEC
            while time.time() < target:
                simulation_app.update()
                time.sleep(0.05)
        # Stage 2: policy + optimizer
        if POLICY_LOAD_AFTER_IMPORT:
            _load_policy_and_optimizer()
        else:
            print("\033[33mPOLICY_LOAD_AFTER_IMPORT=0: Policy loading skipped (you can call manually).\033[0m")
        _define_ros_and_policy_classes()
        if not _is_timeline_playing():
            _play_timeline_if_requested()
        _schedule_rl_activation()
    else:
        if bridge is None:
            if EAGER_ALL:
                print("\033[36mEAGER_ALL=1: Starting heavy import + policy loading without waiting for PLAY.\033[0m")
            _heavy_imports_only()
            _load_policy_and_optimizer()
            _define_ros_and_policy_classes()
    import rclpy
    rclpy.init(args=None)
    fresh_start_env = os.getenv("RL_FINETUNE_FRESH_START", "0").strip()
    fresh_start = fresh_start_env in ("1", "true", "True")
    if not fresh_start:
        resumed = load_latest_checkpoint_if_exists()
        if not resumed:
            save_checkpoint("startup")
    else:
        print("\033[32mFresh start requested; ignoring existing checkpoints.\033[0m")
        save_checkpoint("startup")

    if Get_End_Effector_Pose is None or RewardPublisher is None or Action_Publisher is None:
        ok = _define_ros_and_policy_classes()
        if not ok:
            print("\033[31mROS Node classes could not be defined; exiting.\033[0m")
            raise SystemExit(2)
    get_end_effector_pose = Get_End_Effector_Pose()
    reward_node = RewardPublisher()
    joy_publisher = Action_Publisher()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_end_effector_pose)
    executor.add_node(joy_publisher)
    executor.add_node(reward_node)

    try:
        while rclpy.ok() and not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
           
    except KeyboardInterrupt:
        pass
    finally:
        for node in [get_end_effector_pose, joy_publisher, reward_node]:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        override_path = "/tmp/tbar_override.usda"
        if os.path.exists(override_path):
            try:
                os.remove(override_path)
                print(f"\033[32mOverride layer deleted: {override_path}\033[0m")
            except Exception as e:
                print(f"\033[31mOverride layer deletion error: {e}\033[0m")
        try:
            simulation_app.close()
        except Exception:
            pass
        
        print("\033[32mSimulationApp closed.\033[0m")