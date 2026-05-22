"""One-off: flatten the official ComfyUI **WAN 2.2** subgraph template into a flat
UI-graph workflow comfy_run.py's UI->API converter can ingest (it is flat-graph
only — subgraphs aren't supported).

Source (UI/graph export, the subgraphed template):
  raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/03_video_wan2_2_14B_i2v_subgraphed.json

The template wraps the whole pipeline in ONE subgraph definition
(`definitions.subgraphs[0]`) instanced by a single outer node. Litegraph encodes
the subgraph boundary with virtual nodes: inner links from `origin_id == -10` are
the subgraph's *inputs*, and inner links to `target_id == -20` are its *outputs*.
Crucially, every promoted **widget** value (model names, prompts, width/height/
length) is ALSO stored on the inner node itself, so flattening = inline the inner
nodes, drop the `-10` widget-promotion links (values stay on the nodes), keep all
inner<->inner links, and rewire only the genuine *data* connections that crossed
the boundary (here: `start_image` IMAGE in, `VIDEO` out).

This is the WAN 2.2 14B two-expert (high/low-noise) i2v graph with the lightx2v
4-step Lightning LoRAs — same family as the WAN 2.1 VACE preset, all core nodes.

Run:  python wan22.derive.py SRC_SUBGRAPH_TEMPLATE.json OUT_DIR
Kept in-repo so the flat graph can be regenerated if the template moves.
"""
import json
import sys

SKIP_TYPES = {"Note", "MarkdownNote"}
# Boundary inputs of these types are real data wires to rewire across the boundary;
# everything else (STRING/INT/FLOAT/COMBO/BOOLEAN) is a promoted widget — its value
# already lives on the inner node, so the -10 link is just dropped.
DATA_TYPES = {"IMAGE", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING", "MASK",
              "VIDEO", "AUDIO", "CLIP_VISION", "CONTROL_NET"}


def flatten(template_path):
    d = json.load(open(template_path, encoding="utf-8"))
    subs = d.get("definitions", {}).get("subgraphs", [])
    if len(subs) != 1:
        sys.exit(f"expected exactly 1 subgraph definition, found {len(subs)}")
    sub = subs[0]
    nodes, links = d["nodes"], d["links"]
    inst = next(n for n in nodes if n.get("type") == sub["id"])

    base_nid = d.get("last_node_id") or max(n["id"] for n in nodes)
    base_lid = d.get("last_link_id") or max((l[0] for l in links), default=0)

    # outer nodes we keep (drop the subgraph instance + any notes)
    out_nodes = [n for n in nodes if n["id"] != inst["id"] and n.get("type") not in SKIP_TYPES]
    out_ids = {n["id"] for n in out_nodes}
    # outer links not touching the instance (its crossings are re-added below)
    new_links = [l for l in links if inst["id"] not in (l[1], l[3])]

    # instance input name -> outer (src_node, src_slot) via the outer link feeding it
    inst_in_src = {}
    for ii in inst.get("inputs", []):
        if ii.get("link") is not None:
            src = next((l for l in links if l[0] == ii["link"]), None)
            if src:
                inst_in_src[ii["name"]] = (src[1], src[2])
    # instance output index -> list of outer (dst_node, dst_slot)
    inst_out_dst = []
    for oi in inst.get("outputs", []):
        dsts = []
        for lk in (oi.get("links") or []):
            l = next((x for x in links if x[0] == lk), None)
            if l:
                dsts.append((l[3], l[4]))
        inst_out_dst.append(dsts)

    # inline inner nodes; renumber ONLY ids that collide with kept outer ids
    inner_keep = [n for n in sub["nodes"] if n.get("type") not in SKIP_TYPES]
    nid_map, nxt = {}, base_nid
    for n in inner_keep:
        if n["id"] in out_ids:
            nxt += 1
            nid_map[n["id"]] = nxt
        else:
            nid_map[n["id"]] = n["id"]
    for n in inner_keep:
        m = dict(n)
        m["id"] = nid_map[n["id"]]
        out_nodes.append(m)

    sub_inputs = sub["inputs"]      # order == origin_slot for -10 boundary inputs
    lid = base_lid

    def add(src_n, src_s, dst_n, dst_s, typ):
        nonlocal lid
        lid += 1
        new_links.append([lid, src_n, src_s, dst_n, dst_s, typ])

    for il in sub["links"]:
        oid, oslot = il["origin_id"], il["origin_slot"]
        tid, tslot, typ = il["target_id"], il["target_slot"], il.get("type")
        if il["target_id"] == -20:            # boundary OUTPUT: inner producer -> outer consumers
            if oid in nid_map:
                for dst_n, dst_s in inst_out_dst[oslot]:
                    add(nid_map[oid], oslot, dst_n, dst_s, typ)
            continue
        if tid not in nid_map:                 # target was a dropped note
            continue
        if oid == -10:                         # boundary INPUT
            bi = sub_inputs[oslot]
            if bi.get("type") in DATA_TYPES:   # real data wire -> rewire from outer source
                src = inst_in_src.get(bi["name"])
                if src:
                    add(src[0], src[1], nid_map[tid], tslot, bi.get("type"))
            # else: promoted widget — value already on the inner node, drop the link
            continue
        if oid not in nid_map:                 # origin was a dropped note
            continue
        add(nid_map[oid], oslot, nid_map[tid], tslot, typ)

    d["nodes"] = out_nodes
    d["links"] = new_links
    d["last_node_id"] = max(n["id"] for n in out_nodes)
    d["last_link_id"] = lid
    d.pop("definitions", None)
    return d, nid_map


def validate(d):
    """Structural check: every link endpoint resolves to a real node + slot."""
    by_id = {n["id"]: n for n in d["nodes"]}
    errs = []
    for l in d["links"]:
        lid, sn, ss, tn, ts, typ = l[:6]
        if sn not in by_id:
            errs.append(f"link {lid}: source node {sn} missing")
        if tn not in by_id:
            errs.append(f"link {lid}: target node {tn} missing")
    # no leftover subgraph instance / virtual ids
    if any(n["id"] < 0 for n in d["nodes"]):
        errs.append("a virtual (-10/-20) node leaked into the flat graph")
    return errs


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    d, _ = flatten(src)
    errs = validate(d)
    path = f"{out_dir}/wan22_i2v.workflow.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"wrote {path} ({len(d['nodes'])} nodes, {len(d['links'])} links)")
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("structural validation: OK (all link endpoints resolve, no virtual nodes)")


if __name__ == "__main__":
    main()
