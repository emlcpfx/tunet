"""
comfy_resolve.py — derive a comfy_spark preset's dependencies from a ComfyUI
workflow, so people can bring their own graph and we auto-grab the custom nodes
and model weights it needs.

This is the Phase-1 "resolver brain" for EZ-Comfy (see EZ_COMFY_TODO.md). It is
pure stdlib (urllib only) on purpose: it runs locally for fast iteration, can be
packed onto the Spark node, and can be shelled out from tunet-web — one
implementation, never reimplemented in TS.

  workflow JSON  ->  { node_packs[], models[], unresolved[], unknown_nodes[],
                       ambiguous_nodes[], notes[] }

The output mirrors the existing preset schema (node_packs + models), so
"resolve a workflow" == "draft a preset."

Resolution is layered, because real workflows only carry partial metadata
(verified against the shipped presets):

  NODES (active nodes only — bypassed/muted are dropped, matching comfy_run's
         UI->API converter):
    1. properties.aux_id        -> github.com/<aux_id>           (exact)
    2. properties.cnr_id        -> Comfy Registry / Manager map   (exact-ish)
    3. SEED_NODE_MAP            -> curated LTX/Wan-ecosystem map   (offline)
    4. ComfyUI-Manager extension-node-map.json (cached)           (long tail)
       - class mapping to >1 repo -> ambiguous (needs confirmation)
       - class mapping to 0 repos -> unknown
    ("comfy-core" / built-in classes need no pack and are dropped.)

  MODELS (model-extension widget values on active nodes):
    - dest folder inferred from the loader node type (checkpoints/loras/vae/
      text_encoders/...), preserving any subpath in the filename.
    - URL from properties.models (embedded) -> Manager model-list.json -> else
      it lands in `unresolved` (exactly the URLs a human had to supply).
    - weights pulled by preprocessor packs (controlnet_aux, depth-anything) are
      tagged auto_download so they aren't reported as required.
"""

import json
import os
import re
import sys
import time
import urllib.request

# ── Model folder inference ────────────────────────────────────────────────────
# ComfyUI's loader node -> models/<folder> (folder_paths categories). Custom
# loaders fall through to _folder_from_name() which keys off the type name.
LOADER_FOLDERS = {
    "CheckpointLoaderSimple": "checkpoints",
    "CheckpointLoader": "checkpoints",
    "ImageOnlyCheckpointLoader": "checkpoints",
    "unCLIPCheckpointLoader": "checkpoints",
    "LoraLoader": "loras",
    "LoraLoaderModelOnly": "loras",
    "LTXICLoRALoaderModelOnly": "loras",
    "VAELoader": "vae",
    "LTXVAudioVAELoader": "vae",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "TripleCLIPLoader": "text_encoders",
    "QuadrupleCLIPLoader": "text_encoders",
    "LTXAVTextEncoderLoader": "text_encoders",
    "UNETLoader": "diffusion_models",
    "ControlNetLoader": "controlnet",
    "UpscaleModelLoader": "upscale_models",
    "LatentUpscaleModelLoader": "latent_upscale_models",
    "CLIPVisionLoader": "clip_vision",
    "StyleModelLoader": "style_models",
    "GLIGENLoader": "gligen",
    # SeedVR2 video upscaler keeps its DiT + VAE weights together under models/SEEDVR2
    # (the node's own cache dir), so both loaders map there rather than vae/unknown.
    "SeedVR2LoadDiTModel": "SEEDVR2",
    "SeedVR2LoadVAEModel": "SEEDVR2",
}

# Loaders whose weights are normally auto-fetched by the node pack itself, not
# something the user must supply (see README: yolox/dwpose/depth anything).
PREPROCESSOR_LOADERS = {
    "DWPreprocessor", "CannyEdgePreprocessor", "OpenposePreprocessor",
    "LineArtPreprocessor", "MiDaS-DepthMapPreprocessor",
    "LoadVideoDepthAnythingModel", "VideoDepthAnythingProcess",
    "DepthAnythingV2Preprocessor",
}

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".onnx",
              ".bin", ".gguf", ".sft")

# Frontend/UI-only classes that carry no provenance and need no node pack.
CORE_EXTRA = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}

# When one physical file is referenced by several loaders (e.g. an LTX .safetensors
# read as checkpoint AND vae AND text encoder), it's downloaded once — keep the
# entry whose folder ranks earliest here.
FOLDER_PRIORITY = ["checkpoints", "diffusion_models", "unet", "loras", "controlnet",
                   "vae", "text_encoders", "clip_vision", "style_models",
                   "upscale_models", "latent_upscale_models", "gligen", "unknown"]

# ── Curated seed: class -> github repo for the LTX/Wan ecosystem these presets
# target. Covers custom nodes that ship NO provenance (cnr_id/aux_id are null in
# the exports). The Manager map (fetched) covers everything else. ──────────────
SEED_NODE_MAP = {
    # Lightricks/ComfyUI-LTXVideo (non-core LTX nodes)
    "LTXAddVideoICLoRAGuide": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "LTXICLoRALoaderModelOnly": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "LTXVImgToVideoConditionOnly": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "LTXVTiledVAEDecode": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "LTXFloatToInt": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "LTXVHDRDecodePostprocess": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "GemmaAPITextEncode": "https://github.com/Lightricks/ComfyUI-LTXVideo",
    # Fannovel16/comfyui_controlnet_aux (preprocessors)
    "DWPreprocessor": "https://github.com/Fannovel16/comfyui_controlnet_aux",
    "CannyEdgePreprocessor": "https://github.com/Fannovel16/comfyui_controlnet_aux",
    "OpenposePreprocessor": "https://github.com/Fannovel16/comfyui_controlnet_aux",
    "LineArtPreprocessor": "https://github.com/Fannovel16/comfyui_controlnet_aux",
    # cubiq/ComfyUI_essentials
    "SimpleMath+": "https://github.com/cubiq/ComfyUI_essentials",
    "GetImageSize+": "https://github.com/cubiq/ComfyUI_essentials",
    # numz/ComfyUI-SeedVR2_VideoUpscaler (diffusion video super-resolution).
    # Graphs exported with the ainvfx fork carry an aux_id (which wins); these seed
    # entries resolve a BYO SeedVR2 graph that ships no provenance.
    "SeedVR2VideoUpscaler": "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
    "SeedVR2LoadDiTModel": "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
    "SeedVR2LoadVAEModel": "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
    "SeedVR2TorchCompileSettings": "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
}

# Some packs register only an ENUM OPTION (a scheduler/sampler name) into a CORE
# node's combo — no custom node class appears in the graph, so class-based
# resolution misses them. A widget value here implies its pack. (Found the hard
# way: ltx_faceswap's BasicScheduler uses 'bong_tangent' from RES4LYF, which was
# absent from node_packs → the render output failed prompt validation on Spark.)
# Extensible; only add values distinctive to one pack.
WIDGET_VALUE_PACKS = {
    "bong_tangent": "https://github.com/ClownsharkBatwing/RES4LYF",
    "beta57":       "https://github.com/ClownsharkBatwing/RES4LYF",
}

# UI-only convenience nodes (no backend class, never in API format) — need no pack.
UI_ONLY_NODES = {
    "Fast Groups Muter (rgthree)", "Fast Groups Bypasser (rgthree)",
    "Label (rgthree)", "Reroute (rgthree)", "Note Plus (rgthree)",
}

# Canonical authors — when a class maps to several repos (official + forks/mirrors)
# in the Manager map, prefer the repo owned by one of these over the rest.
CANONICAL_OWNERS = {
    "lightricks", "kijai", "kosinkadink", "rgthree", "cubiq",
    "fannovel16", "ltdrdata", "evanspearman", "clownsharkbatwing", "yuvraj108c",
    "alisson-anjos", "city96",
}

# Manager map points built-in classes at the core ComfyUI repo — those need no
# pack. (Bonus: on the fetch path this authoritatively identifies core nodes that
# carry no provenance in the export, e.g. ImageToMask.)
CORE_REPOS = {"https://github.com/comfyanonymous/comfyui"}


def _prefer_canonical(repos):
    """From a set of repo URLs, return the single canonical-owner repo if exactly
    one matches; else None (genuinely ambiguous)."""
    hits = [r for r in repos if _norm_repo(r).lower().split("github.com/")[-1].split("/")[0] in CANONICAL_OWNERS]
    return hits[0] if len(hits) == 1 else None


# ── ComfyUI-Manager databases (fetched + cached). Overridable via env. ─────────
EXT_NODE_MAP_URL = os.environ.get(
    "COMFY_EXT_NODE_MAP_URL",
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json")
MODEL_LIST_URL = os.environ.get(
    "COMFY_MODEL_LIST_URL",
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json")
CACHE_TTL = 24 * 3600


# ── Workflow shape helpers ────────────────────────────────────────────────────

def active_nodes(graph):
    """Yield (class_type, properties, string_widget_values) for every ACTIVE node,
    normalizing UI-graph and API formats. The caller derives model filenames and
    enum-implied packs from the string widget values.

    UI-graph: {"nodes": [{type, mode, properties, widgets_values}, ...]}
              mode 0 = active; 2 = muted, 4 = bypassed (dropped, like the
              UI->API converter does).
    API:      {"<id>": {class_type, inputs}, ...}  (no provenance, no modes)
    """
    if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
        for n in graph["nodes"]:
            if n.get("mode") in (2, 4):
                continue
            ct = n.get("type")
            if not ct:
                continue
            yield ct, (n.get("properties") or {}), _str_widgets(n.get("widgets_values"))
    elif isinstance(graph, dict):
        for n in graph.values():
            if not isinstance(n, dict) or "class_type" not in n:
                continue
            ins = n.get("inputs") or {}
            yield n["class_type"], {}, [v for v in ins.values() if isinstance(v, str)]


def _str_widgets(wv):
    if isinstance(wv, dict):
        vals = wv.values()
    elif isinstance(wv, list):
        vals = wv
    else:
        return []
    return [v for v in vals if isinstance(v, str)]


def _looks_like_model(v):
    return isinstance(v, str) and v.lower().endswith(MODEL_EXTS)


def _folder_from_name(type_name):
    """Best-effort model folder from a custom loader's type name."""
    t = type_name.lower()
    if "lora" in t:                        return "loras"
    if "vae" in t:                         return "vae"
    if "checkpoint" in t or "ckpt" in t:   return "checkpoints"
    if "controlnet" in t:                  return "controlnet"
    if "clipvision" in t or "clip_vision" in t: return "clip_vision"
    if "textencoder" in t or "text_encoder" in t or "cliploader" in t: return "text_encoders"
    if "upscale" in t:                     return "latent_upscale_models" if "latent" in t else "upscale_models"
    if "unet" in t or "diffusion" in t:    return "diffusion_models"
    return "unknown"


def _norm_repo(url):
    """Normalize a github repo URL for dedup (drop scheme/case/.git/trailing /)."""
    u = url.strip().rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    return u


# ── ComfyUI-Manager DB fetch (+ cache) ────────────────────────────────────────

def _cache_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dl", "resolver_cache")


def _fetch_json(url, cache_name, notes):
    """GET + cache a Manager DB. Falls back to a stale cache, then to nothing
    (the seed map + embedded metadata still work offline)."""
    cdir = _cache_dir()
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, cache_name)
    if os.path.isfile(cpath) and (time.time() - os.path.getmtime(cpath)) < CACHE_TTL:
        try:
            with open(cpath, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "comfy_resolve"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception as e:  # network/parse — degrade gracefully
        if os.path.isfile(cpath):
            notes.append(f"using stale cache for {cache_name} ({e})")
            try:
                with open(cpath, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass
        notes.append(f"could not fetch {cache_name} ({e}); long-tail nodes/models may be unresolved")
        return None


def _build_class_index(ext_map, notes):
    """Invert ComfyUI-Manager's extension-node-map.json into class -> {repos}.
    Format: { "<repo_url>": [ [class, class, ...], {meta} ], ... }."""
    idx = {}
    if not isinstance(ext_map, dict):
        return idx
    for repo, payload in ext_map.items():
        if not (isinstance(payload, list) and payload and isinstance(payload[0], list)):
            continue
        for cls in payload[0]:
            if isinstance(cls, str):
                idx.setdefault(cls, set()).add(_norm_repo(repo))
    return idx


def _build_model_index(model_list):
    """Index ComfyUI-Manager's model-list.json by filename -> url."""
    idx = {}
    for m in (model_list or {}).get("models", []) if isinstance(model_list, dict) else []:
        url = m.get("url")
        if not url:
            continue
        for key in (m.get("filename"), m.get("name")):
            if isinstance(key, str) and key:
                idx.setdefault(os.path.basename(key), url)
    return idx


# ── Core resolution ───────────────────────────────────────────────────────────

def resolve_workflow(graph, *, fetch=True):
    """Resolve a workflow's node packs + models. See module docstring."""
    notes = []
    class_index = model_index = {}
    if fetch:
        class_index = _build_class_index(_fetch_json(EXT_NODE_MAP_URL, "extension-node-map.json", notes), notes)
        model_index = _build_model_index(_fetch_json(MODEL_LIST_URL, "model-list.json", notes))

    packs = {}            # normalized repo url -> source ("aux_id"/"cnr_id"/"seed"/"manager")
    unknown_nodes = set()
    ambiguous = {}        # class -> [repo, ...]
    core_classes = set()
    raw_models = []       # one entry per (loader, filename); deduped below

    for ct, props, widgets in active_nodes(graph):
        # ── nodes (class-based) ──
        repo, source = _resolve_node(ct, props, class_index)
        if source == "core":
            core_classes.add(ct)
        elif source == "ambiguous":
            ambiguous[ct] = repo
        elif repo:
            packs.setdefault(_norm_repo(repo), source)
        else:
            unknown_nodes.add(ct)

        # ── packs implied by an enum widget value (scheduler/sampler name) ──
        for v in widgets:
            ev = WIDGET_VALUE_PACKS.get(v)
            if ev:
                if packs.setdefault(_norm_repo(ev), "enum") == "enum":
                    notes.append(f"pack inferred from widget value '{v}' on {ct} -> {ev}")

        # ── models ──
        files = [v for v in widgets if _looks_like_model(v)]
        folder = LOADER_FOLDERS.get(ct) or _folder_from_name(ct)
        embedded = _embedded_models(props)
        for fn in files:
            base = os.path.basename(fn)
            raw_models.append({
                "filename": fn, "base": base, "loader": ct, "folder": folder,
                "url": embedded.get(base) or model_index.get(base),
                "auto_download": ct in PREPROCESSOR_LOADERS,
                "source": "embedded" if embedded.get(base) else ("manager" if model_index.get(base) else None),
            })

    # Dedup by basename: one physical file may be referenced by several loaders
    # (e.g. an LTX .safetensors read as checkpoint AND vae AND text encoder).
    # Keep the highest-priority folder; record the others as a note.
    def _rank(rec):
        f = rec["folder"]
        return FOLDER_PRIORITY.index(f) if f in FOLDER_PRIORITY else len(FOLDER_PRIORITY)
    seen_models = {}
    for rec in raw_models:
        cur = seen_models.get(rec["base"])
        if cur is None:
            seen_models[rec["base"]] = rec
        else:
            if _rank(rec) < _rank(cur):
                rec["_also"] = cur.get("_also", set()) | {cur["folder"]}
                seen_models[rec["base"]] = rec
            elif rec["folder"] != cur["folder"]:
                cur.setdefault("_also", set()).add(rec["folder"])
    for rec in seen_models.values():
        rec["dest"] = f"models/{rec['folder']}/{rec['filename']}".replace("\\", "/")
        if rec.get("_also"):
            notes.append(f"{rec['base']} referenced by multiple loaders ({rec['folder']} + {', '.join(sorted(rec['_also']))}) — downloaded once to {rec['folder']}")

    models, unresolved = [], []
    for rec in seen_models.values():
        if rec["url"]:
            models.append({"url": rec["url"], "dest": rec["dest"],
                           "source": rec["source"], "auto_download": rec["auto_download"]})
        elif rec["auto_download"]:
            notes.append(f"{rec['filename']} (via {rec['loader']}) likely auto-downloaded by its node pack — not marked required")
        else:
            unresolved.append({"filename": rec["filename"], "dest": rec["dest"], "loader": rec["loader"]})

    return {
        "node_packs": sorted(packs.keys()),
        "node_pack_sources": packs,
        "models": models,
        "unresolved": unresolved,
        "unknown_nodes": sorted(unknown_nodes),
        "ambiguous_nodes": {k: sorted(v) for k, v in ambiguous.items()},
        "core_node_count": len(core_classes),
        "notes": notes,
    }


def _resolve_node(ct, props, class_index):
    """Return (repo_url|list, source). source in
    {core, aux_id, cnr_id, seed, manager, ambiguous, ''}."""
    cnr = props.get("cnr_id")
    if cnr == "comfy-core" or ct in CORE_EXTRA or ct in UI_ONLY_NODES:
        return None, "core"
    aux = props.get("aux_id")
    if aux:
        return f"https://github.com/{aux}", "aux_id"
    if ct in SEED_NODE_MAP:
        return SEED_NODE_MAP[ct], "seed"
    # Prefer a real github URL (cloneable by comfy_run) from the Manager map —
    # this also translates a cnr_id node to its github repo when the map knows it.
    repos = class_index.get(ct)
    if repos:
        if any(_norm_repo(r).lower() in CORE_REPOS for r in repos):
            return None, "core"   # core ComfyUI provides it — no pack, forks ignored
        if len(repos) == 1:
            return next(iter(repos)), "manager"
        pick = _prefer_canonical(repos)
        if pick:
            return pick, "manager"
        return sorted(repos), "ambiguous"
    if cnr:  # registry id, not in the map — record for a cnr/comfy-cli install
        return f"cnr:{cnr}", "cnr_id"
    return None, ""


def _embedded_models(props):
    """Pull {basename: url} from a node's properties.models (newer ComfyUI
    can embed {name,url,hash,directory})."""
    out = {}
    for m in props.get("models") or []:
        if isinstance(m, dict) and m.get("url") and m.get("name"):
            out[os.path.basename(m["name"])] = m["url"]
    return out


# ── Output: draft preset + human report ───────────────────────────────────────

def to_draft_preset(result, workflow_filename):
    """Shape the result into the existing preset schema, with REPLACE_ markers
    for unresolved weights (same convention as the shipped template presets)."""
    models = [{"url": m["url"], "dest": m["dest"]} for m in result["models"]]
    for u in result["unresolved"]:
        models.append({"url": f"REPLACE_WITH_URL_FOR/{u['filename']}", "dest": u["dest"]})
    preset = {
        "description": "DRAFT — auto-resolved by comfy_resolve.py. Fill REPLACE_ "
                       "URLs, then run --convert-only to validate before rendering.",
        "workflow": workflow_filename,
        "node_packs": result["node_packs"],
        "models": models,
    }
    if result["unknown_nodes"]:
        preset["_UNKNOWN_NODES"] = result["unknown_nodes"]
    if result["ambiguous_nodes"]:
        preset["_AMBIGUOUS_NODES"] = result["ambiguous_nodes"]
    return preset


def format_report(result):
    L = []
    L.append(f"node packs ({len(result['node_packs'])}):")
    for p in result["node_packs"]:
        L.append(f"  + {p}   [{result['node_pack_sources'][p]}]")
    if result["ambiguous_nodes"]:
        L.append("ambiguous nodes (class maps to >1 repo — pick one):")
        for cls, repos in result["ambiguous_nodes"].items():
            L.append(f"  ? {cls}: {', '.join(repos)}")
    if result["unknown_nodes"]:
        L.append("unknown nodes (no pack found — supply manually):")
        for cls in result["unknown_nodes"]:
            L.append(f"  ! {cls}")
    L.append(f"\nmodels resolved ({len(result['models'])}):")
    for m in result["models"]:
        tag = " (auto)" if m.get("auto_download") else ""
        L.append(f"  + {m['dest']}  <- {m['url']}  [{m['source']}]{tag}")
    if result["unresolved"]:
        L.append(f"\nmodels UNRESOLVED ({len(result['unresolved'])}) — paste a URL each:")
        for u in result["unresolved"]:
            L.append(f"  ? {u['dest']}  (loader {u['loader']})")
    L.append(f"\ncore (built-in) classes: {result['core_node_count']}")
    for n in result["notes"]:
        L.append(f"note: {n}")
    return "\n".join(L)


def load_graph(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python comfy_resolve.py workflow.json [--no-fetch]")
    res = resolve_workflow(load_graph(sys.argv[1]), fetch="--no-fetch" not in sys.argv)
    print(format_report(res))
