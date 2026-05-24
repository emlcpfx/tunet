"""One-off: clean the upstream **SeedVR2 Video Upscaler** example graph into the
flat UI-graph workflow comfy_run.py's UI->API converter can ingest directly.

Source (UI/graph export, drag-and-drop example):
  raw.githubusercontent.com/numz/ComfyUI-SeedVR2_VideoUpscaler/main/
    example_workflows/SeedVR2_HD_video_upscale.json   (saved here as seedvr2.src.json)

The example ships two *convenience* nodes bypassed (mode 4):
  • JoinImageWithAlpha — an optional RGBA passthrough between the frame source and
    the upscaler. Bypassed = the upscaler reads frames straight from
    GetVideoComponents; we make that the real wiring.
  • SeedVR2TorchCompileSettings — feeds optional torch_compile_args into the DiT
    and VAE loaders. Bypassed = no compile; those optional inputs just go unwired.
Plus two Note nodes (no schema). comfy_run's converter DROPS muted/bypassed nodes
rather than reconnecting through them (it has no bypass-passthrough logic — same
gap ltx_faceswap.derive.py works around), so leaving JoinImageWithAlpha bypassed
would dangle the upscaler's image link onto a node the converter removed and
/prompt would reject the graph. So we strip those nodes here and reconnect:
  - each dropped *passthrough* (an input whose type matches an output that has a
    link) is bypassed by rewiring its consumers to its upstream source, and
  - dropped nodes with no matching input (TorchCompileSettings) just have their
    consumer links dropped (the loaders' torch_compile_args is optional).

Result: LoadVideo -> GetVideoComponents -> SeedVR2VideoUpscaler -> CreateVideo ->
SaveVideo, with the DiT + VAE loaders feeding the upscaler. All scalar/combo
widgets, no subgraphs, no Set/Get — a clean flat graph for the converter.

Run:  python seedvr2.derive.py [SRC.json] [OUT_DIR]
      (defaults: seedvr2.src.json in this dir -> seedvr2.workflow.json here)
Kept in-repo so the graph can be regenerated if the upstream example moves.
"""
import json
import os
import sys

NOTE_TYPES = {"Note", "MarkdownNote"}
# The preset defaults to the 7B model (higher quality); bake that into the graph's
# DiT loader so the shipped default matches the preset and the resolver/selftest see
# the same weight the preset declares. The preset's `model` param can still pick 3B/fp8.
DEFAULT_DIT = "seedvr2_ema_7b_fp16.safetensors"


def clean(src_path):
    d = json.load(open(src_path, encoding="utf-8"))
    nodes, links = d["nodes"], d["links"]
    by_id = {n["id"]: n for n in nodes}
    link_by_id = {l[0]: l for l in links}

    # Drop muted/bypassed nodes (mode 2/4) and notes — the converter drops them
    # too, but we reconnect passthroughs first so nothing dangles.
    drop = {n["id"] for n in nodes if n.get("mode") in (2, 4) or n.get("type") in NOTE_TYPES}
    keep_nodes = [n for n in nodes if n["id"] not in drop]
    keep_ids = {n["id"] for n in keep_nodes}

    def out_type(nid, slot):
        outs = (by_id.get(nid) or {}).get("outputs") or []
        return outs[slot]["type"] if slot < len(outs) else None

    def passthrough_src(nid, slot):
        """A dropped node is a passthrough on `slot` if it has an INPUT of the same
        type carrying a link — return that link's (origin_node, origin_slot)."""
        t = out_type(nid, slot)
        for inp in (by_id[nid].get("inputs") or []):
            if inp.get("type") == t and inp.get("link") is not None:
                l = link_by_id.get(inp["link"])
                if l:
                    return (l[1], l[2])
        return None

    def resolve(nid, slot, seen=None):
        """Follow passthroughs through dropped nodes to the first KEPT source, or
        None if the chain dead-ends in a dropped non-passthrough (e.g. the torch
        compile settings node) — in which case the consumer link is simply dropped."""
        seen = seen or set()
        if nid in keep_ids:
            return (nid, slot)
        if nid in seen:
            return None
        seen.add(nid)
        src = passthrough_src(nid, slot)
        return resolve(src[0], src[1], seen) if src else None

    # Rebuild the links array: keep links into kept nodes, resolving the source
    # through any dropped passthroughs; renumber sequentially.
    new_links, lid = [], 0
    for l in links:
        _lid, sn, ss, tn, ts, typ = l[:6]
        if tn not in keep_ids:
            continue
        src = resolve(sn, ss)
        if src is None:
            continue
        lid += 1
        new_links.append([lid, src[0], src[1], tn, ts, typ])

    # Re-sync per-node inputs[].link / outputs[].links to the rebuilt array — the
    # converter traces edges via inputs[].link, so stale ids would prune the graph.
    keep = {n["id"]: n for n in keep_nodes}
    for n in keep_nodes:
        for s in (n.get("inputs") or []):
            s["link"] = None
        for o in (n.get("outputs") or []):
            o["links"] = []
    for lk in new_links:
        _lid, src, sslot, dst, dslot = lk[0], lk[1], lk[2], lk[3], lk[4]
        dn = keep.get(dst)
        if dn and dslot < len(dn.get("inputs") or []):
            dn["inputs"][dslot]["link"] = _lid
        sn_ = keep.get(src)
        if sn_ and sslot < len(sn_.get("outputs") or []):
            sn_["outputs"][sslot].setdefault("links", []).append(_lid)

    # Bake the 7B default into the DiT loader's model widget (slot 0).
    for n in keep_nodes:
        if n.get("type") == "SeedVR2LoadDiTModel" and isinstance(n.get("widgets_values"), list) and n["widgets_values"]:
            n["widgets_values"][0] = DEFAULT_DIT

    d["nodes"] = keep_nodes
    d["links"] = new_links
    d["last_node_id"] = max(n["id"] for n in keep_nodes)
    d["last_link_id"] = lid
    return d


def validate(d):
    by_id = {n["id"]: n for n in d["nodes"]}
    errs = []
    for l in d["links"]:
        lid, sn, ss, tn, ts = l[:5]
        if sn not in by_id:
            errs.append(f"link {lid}: source node {sn} missing")
        if tn not in by_id:
            errs.append(f"link {lid}: target node {tn} missing")
    linkids = {l[0] for l in d["links"]}
    for n in d["nodes"]:
        for s in (n.get("inputs") or []):
            lk = s.get("link")
            if lk is not None and lk not in linkids:
                errs.append(f"node {n['id']} {n.get('type')} input {s.get('name')!r} "
                            f"has stale link {lk}")
    return errs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "seedvr2.src.json")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else here
    d = clean(src)
    errs = validate(d)
    path = os.path.join(out_dir, "seedvr2.workflow.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    kinds = sorted({n.get("type") for n in d["nodes"]})
    print(f"wrote {path} ({len(d['nodes'])} nodes, {len(d['links'])} links)")
    print(f"node types: {', '.join(kinds)}")
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("structural validation: OK (all link endpoints resolve, no stale links)")


if __name__ == "__main__":
    main()
