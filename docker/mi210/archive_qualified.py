"""Archive a measured 48K candidate only after its generation threshold is met."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from summarize_depth import summarize

SETTINGS = [
    "VLLM_MI210_FLASH_SPLITS", "VLLM_UA_3D_MAXQ", "VLLM_UA_TILE",
    "VLLM_G128_GLUON", "VLLM_G128_GLUON_MMA", "VLLM_G128_DECODE_CDNA2",
    "VLLM_G128_DECODE_CDNA2_TILE", "VLLM_G128_PREFILL_GLUON",
    "VLLM_G128_PREFILL_GLUON_MMA", "VLLM_G128_PREFILL_CDNA2",
    "VLLM_G128_PREFILL_CDNA2_TILE", "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "VLLM_GFX908_INT8_LM_HEAD",
    "VLLM_GFX908_ACT_QUANT", "VLLM_DISABLED_KERNELS", "HIP_VISIBLE_DEVICES",
    "GPU_ARCHS", "AITER_JIT_DIR",
]


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True)


def qualified(inputs, target):
    if target < 90:
        raise ValueError("Archive target must be at least 90 token/s")
    report = summarize(inputs, target)
    if len(report["results"]) != 1:
        raise ValueError("Supply results for exactly one depth/model/image/arm")
    result = report["results"][0]
    if (result["actual_input_tokens"] < 48000 or result["samples"] < 5
            or result["output_tokens"] < 512 or not result["all_samples_target_met"]):
        raise ValueError("Need >=48K actual input, >=512 output, >=5 C1 samples, all meeting target")
    if not result["image_id"]:
        raise ValueError("Benchmark must record image_id; pass --image-id to benchmark.py")
    return report, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--target", type=float, choices=(90, 100, 120), required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-gate-report", type=Path,
                        help="JSON: operator_numerics, graph_replay, quality must each be PASS")
    args = parser.parse_args()
    report, result = qualified(args.inputs, args.target)
    if not args.execute:
        print(json.dumps({"qualified": result, "action": "plan only; no Docker export"}, indent=2))
        return
    if not args.gpu_gate_report:
        parser.error("Export requires --gpu-gate-report")
    gates = json.loads(args.gpu_gate_report.read_text())
    if any(gates.get(key) != "PASS" for key in ("operator_numerics", "graph_replay", "quality")):
        raise ValueError("GPU numeric, graph replay and quality gates must pass")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh archive directory")
    meta = json.loads(docker("inspect", args.container))[0]
    image_id = meta["Image"]
    if result["image_id"].removeprefix("sha256:") != image_id.removeprefix("sha256:"):
        raise ValueError("Measured image_id does not match selected container")
    env = dict(entry.split("=", 1) for entry in meta["Config"]["Env"] if "=" in entry)
    # Store only named operator settings. Never persist API keys or full inspect.
    selected_env = {key: env[key] for key in SETTINGS if key in env}
    command = list(meta["Config"].get("Cmd") or [])
    for index, value in enumerate(command):
        if value == "--api-key" and index + 1 < len(command):
            command[index + 1] = "${MI210_API_KEY}"
        elif value.startswith("--api-key="):
            command[index] = "--api-key=${MI210_API_KEY}"
    probe = (
        "import importlib.util,json,os; "
        "specs={n:importlib.util.find_spec(n) for n in ('vllm','aiter')}; "
        "paths={n:list(s.submodule_search_locations)[0] for n,s in specs.items() if s}; "
        "paths.update({n:p for n,p in {'aiter_jit':os.getenv('AITER_JIT_DIR',''),"
        "'triton_cache':'/root/.triton','vllm_cache':'/root/.cache/vllm'}.items() if p and os.path.isdir(p)}); "
        "print(json.dumps(paths))"
    )
    packages = json.loads(docker("exec", args.container,
                                 "/opt/int8-vllm/.venv/bin/python", "-c", probe).splitlines()[-1])
    if not all(name in packages for name in ("vllm", "aiter")):
        raise ValueError("Cannot locate both operator packages")
    args.output_dir.mkdir(parents=True)
    for index, path in enumerate(args.inputs):
        shutil.copy2(path, args.output_dir / f"benchmark-{index}.jsonl")
    shutil.copy2(args.gpu_gate_report, args.output_dir / "gpu-gates.json")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {"status": "exporting", "image_id": image_id,
                "container": args.container, "operator_env": selected_env,
                "entrypoint": meta["Config"].get("Entrypoint"), "command": command,
                "package_and_cache_paths": packages,
                "note": "Model weights and API key must be supplied separately; mounted runtime code/cache copied alongside image"}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["docker", "image", "save", "--output",
                    str(args.output_dir / "image.tar"), image_id], check=True)
    for name, path in packages.items():
        subprocess.run(["docker", "cp", f"{args.container}:{path}",
                        str(args.output_dir / name)], check=True)
    hashes = []
    for path in sorted(args.output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        hashes.append(f"{digest.hexdigest()}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"saved": str(args.output_dir), "image_id": image_id}))


if __name__ == "__main__":
    main()
