"""One-off: derive the comfy_spark LTX-2.3 face-swap workflow from Alissonerdx's
V3 "head swap" drag-and-drop graph.

Source (UI/graph export), pinned in-repo as ltx_faceswap.src.json:
  huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video
    /resolve/main/workflows/workflow_ltx2_head_swap_drag_and_drop_v3.0.json

Why this script exists — the source graph can't be shipped as-is to Spark:

  1. KJNodes Set/Get virtual nodes. The V3 graph routes almost everything through
     ~68 SetNode/GetNode pairs (a frontend-only mechanism: a Get reads a Set's
     value by variable NAME, with no `links` edge between them). comfy_run.py's
     UI->API converter is a flat-graph converter — it has no Set/Get resolution,
     so it would emit GetNodes with dangling inputs and /prompt would fail. We
     FLATTEN them here: every consumer of a GetNode is repointed at the real node
     feeding the matching SetNode (transitively), then all Set/Get nodes are
     dropped. The result is a plain graph the converter handles.

  2. Ollama auto-prompt. The graph defaults to an "automatic" prompt produced by
     OllamaVideoDescriber talking to http://localhost:11434 — which doesn't exist
     on a Spark node. We rewire the final CLIPTextEncode's `text` straight from
     the "Positive Prompt" box (node 498), so the manual prompt is always used.
     The Ollama/ComfySwitch cluster does NOT auto-prune, though: it terminates in
     three PreviewAny OUTPUT nodes (426, 507, 572), and the converter keeps any
     node feeding an output — so the cluster stayed live and ComfySwitchNode's
     dangling inner-node refs threw a KeyError that rejected the WHOLE prompt
     (even though the real output 341 validated). So we MUTE the cluster's leaf
     previews + the two ComfySwitchNodes + OllamaVideoDescriber explicitly; their
     upstream primitives then prune away, leaving 341 as the only output.

  3. Custom-audio / comparison-preview nodes. The active audio rides the BODY clip
     (node 345's own audio output), so the "Load Custom Audio" node (276, default
     filename that won't be uploaded) and the two comparison VHS_VideoCombine
     preview nodes (365, 559) are muted — leaving node 341 SaveVideo "Result" as
     the single output.

  4. Bypassed LoRA pass-throughs. The model chain runs UNET(478) -> two BYPASSED
     alt-LoRA loaders (419, 537) -> the active head-swap LoRA (573). ComfyUI
     reconnects through a bypassed node; the flat converter just drops it, leaving
     573.model dangling. We resolve LoRA-loader bypass pass-throughs the same way
     as Set/Get. We also BYPASS the abliterated-gemma CLIP LoRA (424) — its only
     job is uncensored prompting and its weight isn't pinned, so by default the
     CLIP comes straight from the DualCLIPLoader (re-enable 424 + add its weight to
     the preset's `models` for uncensored prompts).

  5. Weight filenames. The head-swap LoRA loader (573) and the UNET loader (478)
     widgets are pointed at the filenames the preset's `models` list downloads to.

Run:  python ltx_faceswap.derive.py ltx_faceswap.src.json .
Kept in-repo so the graph can be regenerated if the upstream workflow moves.
"""
import json
import sys

# ── node-id constants (from the V3 source graph) ─────────────────────────────
N_FINAL_ENCODE = 481   # CLIPTextEncode whose .text feeds the positive conditioning
N_POSITIVE_BOX = 498   # PrimitiveStringMultiline "Positive Prompt"
N_HEADSWAP_LORA = 573  # LoraLoaderModelOnly (the V3 head-swap LoRA)
N_UNET = 478           # UNETLoader (LTX-2.3 distilled transformer)
MUTE = [276, 365, 559, 418,  # custom-audio loader, 2 comparison combiners, dead
                             # upscaler loader (418 has no consumers — the V3 graph
                             # ships it but wires no second pass; result is single-pass)
        426, 507, 572,       # vestigial Ollama-path PreviewAny outputs — keeping
                             # them live kept the whole Ollama/switch cluster, whose
                             # ComfySwitchNode dangling refs rejected the prompt
        493, 516, 586]       # the two ComfySwitchNodes + OllamaVideoDescriber
BYPASS = [424]          # abliterated-gemma CLIP LoRA (optional; weight not pinned)

# bypassed LoRA loaders pass their input straight through: output slot -> input name
PASSTHROUGH = {"LoraLoaderModelOnly": {0: "model"},
               "LoraLoader": {0: "model", 1: "clip"}}

# filenames the preset's `models` list saves the weights as (must match loaders)
HEADSWAP_LORA_NAME = "ltxv/ltx2/head_swap_v3_rank_adaptive_fro_098.safetensors"
UNET_NAME = "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors"

VIRTUAL_SETGET = {"SetNode", "GetNode"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def node(wf, nid):
    return next((n for n in wf["nodes"] if n["id"] == nid), None)


def find_input_link(wf, nid, input_name):
    """The link id currently feeding node `nid`'s input `input_name` (or None)."""
    n = node(wf, nid)
    for ins in (n.get("inputs") or []):
        if ins.get("name") == input_name:
            return ins.get("link")
    return None


def rewire_prompt_to_manual(wf):
    """Point the final CLIPTextEncode (481) `text` straight at the Positive Prompt
    box (498), bypassing the Ollama/ComfySwitch decision cluster entirely."""
    link_id = find_input_link(wf, N_FINAL_ENCODE, "text")
    if link_id is None:
        raise SystemExit(f"node {N_FINAL_ENCODE} has no `text` input link to rewire")
    for L in wf["links"]:
        if L[0] == link_id:
            L[1], L[2] = N_POSITIVE_BOX, 0   # source node, source slot
            break
    # keep the source node's outputs[].links honest (cosmetic, but tidy)
    out = node(wf, N_POSITIVE_BOX)["outputs"][0]
    out["links"] = sorted(set((out.get("links") or []) + [link_id]))


def flatten_setget(wf):
    """Resolve all KJNodes SetNode/GetNode pairs into direct links, then drop the
    Set/Get nodes. A consumer reading from a GetNode is repointed at the real
    (non-virtual) node feeding the matching SetNode, following Set->Get->Set chains
    transitively."""
    nodes_by_id = {n["id"]: n for n in wf["nodes"]}
    links_map = {L[0]: (L[1], L[2]) for L in wf["links"]}   # link id -> (src, slot)
    virtual = {n["id"]: n["type"] for n in wf["nodes"] if n["type"] in VIRTUAL_SETGET}

    # var name -> (immediate source node, slot) feeding each SetNode
    set_src = {}
    var_of = {}
    for n in wf["nodes"]:
        if n["type"] not in VIRTUAL_SETGET:
            continue
        var = (n.get("widgets_values") or [None])[0]
        var_of[n["id"]] = var
        if n["type"] == "SetNode":
            lid = (n.get("inputs") or [{}])[0].get("link")
            set_src[var] = links_map.get(lid)   # may be None if unconnected

    def resolve(src):
        """Follow a (node, slot) through any Set/Get hops to a real source."""
        seen = set()
        while src and src[0] in virtual:
            var = var_of.get(src[0])
            nxt = set_src.get(var)
            if not nxt or src[0] in seen:
                return None
            seen.add(src[0])
            src = nxt
        return src

    dropped_links = []
    for L in wf["links"]:
        if L[1] in virtual:                      # source is a Set/Get node
            real = resolve((L[1], L[2]))
            if real is None:
                dropped_links.append(L[0])
            else:
                L[1], L[2] = real[0], real[1]

    # drop links that still touch a virtual node as endpoint (set-input feeds,
    # unconsumed get outputs) and any we couldn't resolve
    wf["links"] = [L for L in wf["links"]
                   if L[0] not in dropped_links
                   and L[1] not in virtual and L[3] not in virtual]
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in virtual]
    return len(virtual)


def flatten_bypassed_loras(wf):
    """Reconnect through BYPASSED LoRA loaders: a consumer of a bypassed loader's
    output is repointed at the source feeding the matching input (transitively, so
    a chain of bypassed loaders collapses to the first live producer)."""
    nodes_by_id = {n["id"]: n for n in wf["nodes"]}
    links_map = {L[0]: (L[1], L[2]) for L in wf["links"]}

    def in_source(nid, input_name):
        for ins in (nodes_by_id[nid].get("inputs") or []):
            if ins.get("name") == input_name:
                return links_map.get(ins.get("link"))
        return None

    def resolve(node_id, slot, seen):
        n = nodes_by_id.get(node_id)
        if n and n.get("mode") == 4 and n["type"] in PASSTHROUGH \
                and slot in PASSTHROUGH[n["type"]] and node_id not in seen:
            src = in_source(node_id, PASSTHROUGH[n["type"]][slot])
            if src:
                return resolve(src[0], src[1], seen | {node_id})
        return (node_id, slot)

    rewired = 0
    for L in wf["links"]:
        real = resolve(L[1], L[2], set())
        if real != (L[1], L[2]):
            L[1], L[2] = real
            rewired += 1
    return rewired


def swap_guide_multi_to_batch(wf):
    """LTXVAddGuideMulti uses a ComfyUI **DynamicCombo** input (`num_guides`, a
    single structured value the frontend assembles from image_1/frame_idx_1/
    strength_1). The flat UI->API converter can't synthesize a DynamicCombo, so
    the node fails ('unexpected keyword argument'). Swap it for KJNodes'
    `LTXVAddGuidesFromBatch`, which takes plain inputs `images` (IMAGE batch) +
    `strength` — converter-friendly. The single V3 guide (node 360's
    persistent-template output, present on every frame) becomes the images batch.
    Both nodes return (positive, negative, latent), so downstream is unchanged."""
    swapped = 0
    for n in wf["nodes"]:
        if n.get("type") != "LTXVAddGuideMulti":
            continue
        wv = n.get("widgets_values") or []
        strength = wv[2] if len(wv) >= 3 else 1.0          # [count, frame_idx, strength]
        n["type"] = "LTXVAddGuidesFromBatch"
        n["widgets_values"] = [strength]
        kept = []
        for ins in (n.get("inputs") or []):
            nm = ins.get("name", "")
            if nm in ("positive", "negative", "vae", "latent"):
                kept.append(ins)
            elif nm.endswith("image_1") or ins.get("label") == "image_1":
                ins["name"] = "images"            # the guide image → images batch
                ins.pop("label", None)
                kept.append(ins)
            # drop any other num_guides.* sub-input sockets (frame_idx_*, strength_*)
        n["inputs"] = kept
        swapped += 1
    return swapped


def derive(src):
    wf = load(src)
    rewire_prompt_to_manual(wf)
    for nid in MUTE:
        n = node(wf, nid)
        if n:
            n["mode"] = 2                        # mute -> dropped by the converter
    for nid in BYPASS:
        n = node(wf, nid)
        if n:
            n["mode"] = 4                        # bypass -> pass-through resolved below
    node(wf, N_HEADSWAP_LORA)["widgets_values"][0] = HEADSWAP_LORA_NAME
    node(wf, N_UNET)["widgets_values"][0] = UNET_NAME
    n_virtual = flatten_setget(wf)               # Set/Get first: concrete links
    n_bypass = flatten_bypassed_loras(wf)        # then reconnect bypassed loaders
    n_guide = swap_guide_multi_to_batch(wf)      # DynamicCombo -> batch guide node
    print(f"flattened {n_virtual} Set/Get virtual node(s); "
          f"rewired {n_bypass} link(s) through bypassed LoRA loaders; "
          f"swapped {n_guide} LTXVAddGuideMulti -> LTXVAddGuidesFromBatch")
    return wf


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "ltx_faceswap.src.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    wf = derive(src)
    path = f"{out_dir}/ltx_faceswap.workflow.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False)
    print(f"wrote {path} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")


if __name__ == "__main__":
    main()
