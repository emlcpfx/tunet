"""Derive the REFERENCE-TO-VIDEO VACE variant from the static inpaint workflow.

The static `wan_vace_inpaint.workflow.json` DISCONNECTS node 162 (reference
LoadImage) -> node 49 (WanVaceToVideo).reference_image for pure object REMOVAL.
Re-connecting it turns the same graph into Wan-VACE's "reference-to-video" /
all-in-one mode: the masked region is regenerated to contain the REFERENCE
image's subject (object/character replacement) instead of being cleaned away.

We derive from our existing FLAT workflow (which still has node 49) rather than
the upstream template, because upstream has since moved to a subgraph layout
(node 49 is gone, replaced by a subgraph) that our flat-only converter path and
this preset's node-id param map don't expect.

Run: python wan_vace_ref2video.derive.py wan_vace_inpaint.workflow.json wan_vace_ref2video.workflow.json
"""
import json
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    wf = json.load(open(src, encoding="utf-8"))

    def node(nid):
        return next(n for n in wf["nodes"] if n["id"] == nid)

    lid = max(l[0] for l in wf["links"]) + 1
    # 162.IMAGE (slot 0) -> 49.reference_image (slot 5)
    wf["links"].append([lid, 162, 0, 49, 5, "IMAGE"])
    for inp in node(49)["inputs"]:
        if inp["name"] == "reference_image":
            inp["link"] = lid
    node(162)["outputs"][0]["links"] = [lid]
    node(162)["widgets_values"] = ["reference.png", "image"]

    json.dump(wf, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {out} ({len(wf['nodes'])} nodes, {len(wf['links'])} links); "
          f"reference link id {lid}")


if __name__ == "__main__":
    main()
