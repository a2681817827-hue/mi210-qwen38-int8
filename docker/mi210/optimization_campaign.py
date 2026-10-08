"""Plan or execute serial MI210 split-K gates; never restart a serving process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SPLITS = (None, 8, 16, 32, 64)


def gate_commands(output_dir, kv_heads=4):
    if kv_heads not in (2, 4):
        raise ValueError("Use TP1 24:4 or experimental TP2 12:2 geometry")
    helpers = Path(__file__).resolve().parent
    for splits in SPLITS:
        label = "default" if splits is None else f"split{splits}"
        for depth in (33, 4097, 48081, 65537):
            for query in (1, 7, 14):
                output = output_dir / f"{label}-kv{depth}-q{query}.jsonl"
                argv = [sys.executable, str(helpers / "operator_sweep.py"),
                        "--context", str(depth), "--query", str(query),
                        "--kv-heads", str(kv_heads), "--split", "--segment-capacity", "64",
                        "--arm", "gluon_cdna2_decode_tile64", "--output", str(output)]
                if splits is not None:
                    argv += ["--target-splits", str(splits)]
                yield {"label": label, "gate": "operator_numerics", "argv": argv,
                       "output": str(output)}
        for query in (8, 10, 14):
            output = output_dir / f"{label}-graph-q{query}.jsonl"
            yield {"label": label, "gate": "graph_replay",
                   "argv": [sys.executable, str(helpers / "graph_split_gate.py"),
                            "--splits", str(splits or 0), "--max-query", str(query),
                            "--kv-heads", str(kv_heads),
                            "--output", str(output)], "output": str(output)}


def inspect_gate(job):
    rows = [json.loads(line) for line in Path(job["output"]).read_text().splitlines()
            if line.strip()]
    if len(rows) != 1 or rows[0].get("candidate_launches", 0) < 1:
        raise ValueError("Expected one gate result exercising the native CDNA2 core")
    row = rows[0]
    fields = ("numeric",) if job["gate"] == "operator_numerics" else (
        "ragged_numeric", "graph_replay")
    if any(row.get(field) != "PASS" for field in fields):
        raise ValueError("GPU gate failed")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Run inside an idle, GPU-enabled qualified MI210 environment")
    parser.add_argument("--kv-heads", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    source_paths = [root / "vllm/v1/attention/ops" / name for name in (
        "triton_unified_attention.py", "mi210_split_tuning.py",
        "gfx90a_g128_gluon_decode.py", "gfx90a_g128_gluon_prefill.py")]
    jobs = list(gate_commands(args.output_dir, args.kv_heads))
    report = {"status": "planned", "hardware_validated": False,
              "performance_claim": None, "quality": "NOT_RUN",
              "target_generation_tokens_per_second": 120,
              "attention_geometry": {"heads": args.kv_heads * 6, "kv_heads": args.kv_heads},
              "two_card_collectives_validated": False,
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_paths}, "jobs": jobs, "completed": []}
    path = args.output_dir / "campaign.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if not args.execute:
        print(json.dumps({"plan": str(path), "gpu_jobs": len(jobs),
                          "hardware_validated": False}))
        return
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Isolate exactly one idle MI210 for this serial campaign")
        if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
            raise RuntimeError("gfx90a is required")
        env = dict(os.environ)
        env.pop("VLLM_MI210_FLASH_SPLITS", None)
        report["status"] = "running"
        save()
        for job in jobs:
            subprocess.run(job["argv"], env=env, check=True)
            report["completed"].append({"label": job["label"], "gate": job["gate"],
                                        "output": job["output"], "result": inspect_gate(job)})
            save()
        report.update(status="GPU_GATES_PASS", hardware_validated=True,
                      operator_numerics="PASS", graph_replay="PASS")
        save()
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(json.dumps({"report": str(path), "status": report["status"],
                      "performance_claim": None, "quality": "NOT_RUN"}))


if __name__ == "__main__":
    main()
