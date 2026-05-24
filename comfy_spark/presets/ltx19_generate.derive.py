"""Fix the LTX-2 19B 'wLora' generate graph for Spark conversion.

The official `LTX-2_T2V_Distilled_wLora` ships two BYPASSED camera-LoRA slots
(LoraLoaderModelOnly 5230/5231) feeding the two-stage sampler subgraph's model
inputs (slots 0,1); the checkpoint 5228 also feeds slot 5 directly. comfy_run's
subgraph flattener can't resolve a BYPASSED node's passthrough across the subgraph
boundary, so the converted graph references the dropped 5230/5231 and `/prompt`
fails with `KeyError: '5230'` / `'5231'`.

Fix: delete 5230/5231 and repoint the subgraph's model inputs straight to the
checkpoint 5228 (which is exactly what a bypassed LoraLoaderModelOnly does — pass
the model through). Camera LoRAs still apply: comfy_spark's --lora stack splices
onto lora_chain anchor 5228 and repoints every consumer (all three subgraph model
inputs) to the end of the chain.

Run: python ltx19_generate.derive.py ltx19_generate.workflow.json ltx19_generate.workflow.json
"""
import json
import sys

CKPT, SLOTS = 5228, (5230, 5231)   # checkpoint; the two bypassed camera-LoRA loaders


def main():
    src, out = sys.argv[1], sys.argv[2]
    wf = json.load(open(src, encoding="utf-8"))
    links = wf["links"]   # [id, src_node, src_slot, dst_node, dst_slot, type]

    # Repoint links that ORIGINATE at a bypassed loader -> checkpoint MODEL out (slot 0).
    for l in links:
        if l[1] in SLOTS:
            l[1], l[2] = CKPT, 0
    # Drop any link still touching a bypassed loader (its model-in from the checkpoint).
    wf["links"] = [l for l in links if l[1] not in SLOTS and l[3] not in SLOTS]
    # Remove the bypassed loader nodes.
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in SLOTS]
    # Re-sync the checkpoint's MODEL output link list.
    ck = next(n for n in wf["nodes"] if n["id"] == CKPT)
    ck["outputs"][0]["links"] = [l[0] for l in wf["links"] if l[1] == CKPT]

    json.dump(wf, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {out} ({len(wf['nodes'])} nodes, {len(wf['links'])} links); "
          f"dropped {SLOTS}, rewired subgraph model inputs -> {CKPT}")


if __name__ == "__main__":
    main()
