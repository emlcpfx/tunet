"""One-off: derive the two comfy_spark VACE-inpaint workflow variants from the
official ComfyUI template. Source (UI/graph export):
  raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_wan_vace_inpainting.json

  static  : the template, reference_image disconnected (object removal, not
            replacement). Mask is a single image repeated across all frames.
  maskvid : same, but the mask is a VIDEO (per-frame masks): LoadVideo ->
            GetVideoComponents -> ImageToMask, feeding all mask consumers
            per-frame and dropping the single-frame repeat (111/129/130).

Both variants read the mask from the image's RED channel (white = region to
inpaint/remove), not the alpha channel the upstream template uses — so ordinary
white-on-black roto masks work, no alpha needed.

Run:  python wan_vace_inpaint.derive.py SRC_TEMPLATE.json OUT_DIR
Kept in-repo so the variants can be regenerated if the upstream template moves.
"""
import json
import sys


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def node(wf, nid):
    return next(n for n in wf["nodes"] if n["id"] == nid)


def set_input_link(wf, nid, input_name, link_id):
    for ins in node(wf, nid).get("inputs", []):
        if ins["name"] == input_name:
            ins["link"] = link_id
            return
    raise KeyError(f"node {nid} has no input {input_name!r}")


def drop_link(wf, link_id):
    wf["links"] = [l for l in wf["links"] if l[0] != link_id]


def add_link(wf, lid, from_node, from_slot, to_node, to_slot, typ):
    wf["links"].append([lid, from_node, from_slot, to_node, to_slot, typ])


def new_node(wf, nid, ntype, x, y, inputs, outputs, widgets):
    wf["nodes"].append({"id": nid, "type": ntype, "pos": [x, y], "size": [240, 80],
                        "flags": {}, "order": 0, "mode": 0, "inputs": inputs,
                        "outputs": outputs, "properties": {}, "widgets_values": widgets})


def disconnect_reference(wf):
    """Remove 162 -> 49.reference_image (link 261): pure removal, no reference."""
    set_input_link(wf, 49, "reference_image", None)
    drop_link(wf, 261)


def image_to_mask(wf, nid, src_node, src_slot, in_link, out_links, x, y):
    """Add an ImageToMask(red) reading an IMAGE and producing a per-pixel MASK."""
    add_link(wf, in_link, src_node, src_slot, nid, 0, "IMAGE")
    new_node(wf, nid, "ImageToMask", x, y,
             inputs=[{"name": "image", "type": "IMAGE", "link": in_link}],
             outputs=[{"name": "MASK", "type": "MASK", "links": out_links}],
             widgets=["red"])


def make_static(src):
    wf = load(src)
    disconnect_reference(wf)
    node(wf, 209)["widgets_values"] = ["input_video.mp4", "image"]   # source video
    node(wf, 215)["widgets_values"] = ["mask.png", "image"]          # mask image

    # Mask from the red channel of LoadImage(215).IMAGE (slot 0), not its alpha
    # (MASK, slot 1). Repoint the three single-frame consumers; control_masks
    # still flows through the repeat chain (111->129->130) to match frame count.
    px, py = node(wf, 215)["pos"]
    image_to_mask(wf, 250, 215, 0, 910, [911, 912, 913], px + 320, py)
    drop_link(wf, 335); add_link(wf, 911, 250, 0, 208, 2, "MASK")    # composite mask
    set_input_link(wf, 208, "mask", 911)
    drop_link(wf, 345); add_link(wf, 912, 250, 0, 111, 0, "MASK")    # -> repeat -> control_masks
    set_input_link(wf, 111, "mask", 912)
    drop_link(wf, 351); add_link(wf, 913, 250, 0, 219, 0, "MASK")    # gray-fill invert
    set_input_link(wf, 219, "mask", 913)
    return wf


def make_maskvid(src):
    wf = load(src)
    disconnect_reference(wf)
    node(wf, 209)["widgets_values"] = ["input_video.mp4", "image"]

    # 215: LoadImage -> LoadVideo (the mask, now a per-frame video).
    n215 = node(wf, 215)
    n215["type"] = "LoadVideo"
    n215["widgets_values"] = ["mask_video.mp4"]
    n215["outputs"] = [{"name": "VIDEO", "type": "VIDEO", "links": [900]}]
    n215.pop("inputs", None)
    px, py = n215["pos"]

    # LoadVideo -> GetVideoComponents -> ImageToMask(red) = per-frame masks.
    new_node(wf, 250, "GetVideoComponents", px + 320, py,
             inputs=[{"name": "video", "type": "VIDEO", "link": 900}],
             outputs=[{"name": "images", "type": "IMAGE", "links": [901]},
                      {"name": "audio", "type": "AUDIO", "links": None},
                      {"name": "fps", "type": "FLOAT", "links": None}],
             widgets=[])
    add_link(wf, 900, 215, 0, 250, 0, "VIDEO")
    image_to_mask(wf, 251, 250, 0, 901, [902, 903, 904], px + 640, py)

    # Per-frame mask straight to all three consumers (no repeat needed).
    drop_link(wf, 349); add_link(wf, 902, 251, 0, 49, 4, "MASK")     # control_masks
    set_input_link(wf, 49, "control_masks", 902)
    drop_link(wf, 335); add_link(wf, 903, 251, 0, 208, 2, "MASK")    # composite mask
    set_input_link(wf, 208, "mask", 903)
    drop_link(wf, 351); add_link(wf, 904, 251, 0, 219, 0, "MASK")    # gray-fill invert
    set_input_link(wf, 219, "mask", 904)
    # The single-frame repeat chain (215->111->129->130) is obsolete: delete it.
    for lid in (345, 201, 202, 346):
        drop_link(wf, lid)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in {111, 129, 130}]
    for o in node(wf, 211).get("outputs", []):  # 211 no longer feeds 129.amount
        if o.get("links"):
            o["links"] = [l for l in o["links"] if l != 346]
    return wf


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    for name, wf in [("wan_vace_inpaint.workflow.json", make_static(src)),
                     ("wan_vace_inpaint_maskvid.workflow.json", make_maskvid(src))]:
        path = f"{out_dir}/{name}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False)
        print(f"wrote {path} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")


if __name__ == "__main__":
    main()
