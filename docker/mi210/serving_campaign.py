"""Run serial host Docker comparisons and retain every arm's raw artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.request

from short_c1_plan import candidates
from metrics_delta import acceptance


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/opt/int8-vllm/.venv/bin/python"
TOOLS = "/tmp/mi210-tools"


def extra_arm_settings(recipes, control, existing):
    """Allow kernel/geometry combinations while keeping the model contract."""
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError('Expected a nonempty mapping of labels to settings')
    choices = {
        'SPEC_TOKENS': {str(n) for n in range(1, 33)},
        'MI210_QUERY_TILE': {'5', '10', '16'},
        'MI210_KV_TILE': {'32', '64'},
        'MI210_DRAFT_SPLITS': {'8', '16', '32', '64'},
        'MI210_FLASH_SPLITS': {'8', '16', '32', '64'},
        **{name: {'0', '1'} for name in (
            'MI210_INT8_LM_HEAD', 'MI210_TUNED_CK', 'MI210_GDN_TINY_BA',
            'MI210_GDN_ZVIEW', 'MI210_NATIVE_GDN', 'MI210_INT8_QK',
            'MI210_TARGET_WAVE_LAYOUT', 'MI210_DRAFT_GLUON',
            'MI210_REGISTER_QUANT_ALIAS', 'MI210_EXTENDED_CK',
            'MI210_QKPV_INTERLEAVE', 'MI210_REDUCTION_POLICY')},
        'MI210_NATIVE_GDN_RAW': {'0', '1'},
    }
    result = {}
    for label, values in recipes.items():
        if (not isinstance(label, str) or len(label) > 40
                or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', label)
                or label in existing):
            raise ValueError('Invalid or existing arm label')
        if not isinstance(values, dict) or not values:
            raise ValueError('Expected explicit kernel settings for ' + label)
        for name, value in values.items():
            if name not in choices or not isinstance(value, str) or value not in choices[name]:
                raise ValueError('Unsupported setting: ' + name)
        env = {**control, **values}
        if env.get('MI210_NATIVE_GDN') == '1' and env.get('MI210_NATIVE_GDN_RAW') == '1':
            raise ValueError('Select one GDN output contract')
        if (env.get('MI210_NATIVE_GDN') == '1' or env.get('MI210_NATIVE_GDN_RAW') == '1') and int(env['SPEC_TOKENS']) > 15:
            raise ValueError('Native GDN requires at most 16 verification tokens')
        result[label] = env
    return result


def docker_json(*argv):
    return json.loads(subprocess.check_output(["docker", *argv], text=True))


def summarize(directory):
    result = {}
    for workload, filename in (("short", "short/depth-short.jsonl"),
                               ("long", "long/depth-48000.jsonl"),
                               ("long120k", "generation120k/depth-120000.jsonl")):
        path = directory / filename
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        wall = [r["wall_output_tokens_per_second"] for r in rows]
        samples = [s for r in rows for s in r["samples"]]
        generation = [1 / s["tpot_estimate_seconds"] for s in samples
                      if s["tpot_estimate_seconds"] is not None]
        result[workload] = {"wall_tokens_per_second": wall,
                            "median_wall": statistics.median(wall),
                            "generation_estimates": generation,
                            "median_generation": statistics.median(generation) if generation else None,
                            "ttft": [s["ttft_seconds"] for s in samples],
                            "usage": [s["usage"] for s in samples],
                            "output_sha256": [hashlib.sha256(s["output"].encode()).hexdigest()
                                              for s in samples],
                            "timing_schema": "first_last_content_v2"}
        suffix = path.stem.removeprefix("depth-")
        before = path.parent / f"metrics-{suffix}-before.prom"
        after = path.parent / f"metrics-{suffix}-after.prom"
        result[workload]["acceptance"] = acceptance(before, after) if before.exists() and after.exists() else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", default="int8-vllm-mi210:single-candidates-20261005")
    parser.add_argument("--base", default="int8-vllm-mi210:long-context-cdna2")
    parser.add_argument("--arms", nargs="+", default=["original", *dict(candidates())])
    parser.add_argument("--arm-config", type=Path,
                        help='JSON mapping of additional kernel/parameter combinations')
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--reference-quality", type=Path)
    parser.add_argument("--execution", default="graph-c8", choices=("graph-c8", "graph-only-c8", "c8"))
    parser.add_argument("--extended-validation", action="store_true",
                        help="Also run concurrent answers and 120K retrieval/generation")
    parser.add_argument("--capture-fidelity", action="store_true",
                        help='Capture diverse greedy token distributions after timing')
    args = parser.parse_args()
    if args.output_dir.exists() or args.repetitions < 3:
        parser.error("Use a fresh output directory and at least three repetitions")
    settings = dict(candidates())
    settings["original"] = dict(settings["control-ns13"])
    control = settings["control-ns13"]
    for ns in (1, 2, 3):
        settings[f"mtp{ns}"] = {**control, "SPEC_METHOD": "mtp", "SPEC_TOKENS": str(ns)}
    settings["mtp1-no-prefix"] = {**settings["mtp1"], "PREFIX_CACHING": "0"}
    settings["query16-split64"] = {**control, "MI210_QUERY_TILE": "16", "MI210_FLASH_SPLITS": "64"}
    settings["head-split64"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_FLASH_SPLITS": "64"}
    settings["head-query16-split64"] = {**settings["head-split64"], "MI210_QUERY_TILE": "16"}
    settings["ck-control"] = {**control, "MI210_TUNED_CK": "0"}
    settings["ck-short"] = {**control, "MI210_TUNED_CK": "1"}
    settings["ck-head"] = {**control, "MI210_TUNED_CK": "1", "MI210_INT8_LM_HEAD": "1"}
    settings["ck-head-split64"] = {**settings["head-split64"], "MI210_TUNED_CK": "1"}
    settings["head-query16"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_QUERY_TILE": "16"}
    for ns in (7, 9, 11, 15, 17):
        settings[f"head-ns{ns}"] = {**control, "MI210_INT8_LM_HEAD": "1", "SPEC_TOKENS": str(ns)}
    settings["head-tiny-ba"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_GDN_TINY_BA": "1"}
    settings["head-zview"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_GDN_ZVIEW": "1"}
    settings["head-split32"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_FLASH_SPLITS": "32"}
    settings["head-split8"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_FLASH_SPLITS": "8"}
    settings["head-tiny-ba-zview"] = {**settings["head-tiny-ba"], "MI210_GDN_ZVIEW": "1"}
    settings["head-native-gdn"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_NATIVE_GDN": "1"}
    settings["head-native-gdn-tiny-ba-zview"] = {
        **settings["head-tiny-ba-zview"], "MI210_NATIVE_GDN": "1", "MI210_TUNED_CK": "1",
    }
    settings["head-ns15-combined"] = {
        **settings["head-native-gdn-tiny-ba-zview"], "SPEC_TOKENS": "15",
        "MI210_QUERY_TILE": "16", "MI210_FLASH_SPLITS": "32",
    }
    settings["head-ns17-flash"] = {
        **settings["head-ns17"], "MI210_UA_3D_MAXQ": "20",
    }
    settings["head-int8-qk"] = {**control, "MI210_INT8_LM_HEAD": "1", "MI210_INT8_QK": "1"}
    settings["all-combined-int8-qk"] = {
        **settings["head-native-gdn-tiny-ba-zview"], "MI210_INT8_QK": "1",
    }
    for ns in (1, 3):
        settings[f"mtp{ns}-graph-safe-native"] = {
            **settings["head-native-gdn"], "SPEC_METHOD": "mtp",
            "SPEC_TOKENS": str(ns), "PREFIX_CACHING": "0",
        }
    if args.arm_config:
        try:
            settings.update(extra_arm_settings(
                json.loads(args.arm_config.read_text()), control, settings))
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if any(label not in settings for label in args.arms):
        parser.error("Unknown arm")
    if subprocess.check_output(["docker", "ps", "-q"], text=True).strip():
        parser.error("Run on an idle Docker host; stop the previous GPU arm first")
    args.output_dir.mkdir(parents=True)
    report = {"status": "RUNNING", "quality_scope": "operator numerics plus 52 greedy probes and smoke",
              "full_fidelity": "NOT_RUN", "arms": []}
    if args.arm_config:
        report['arm_config_sha256'] = hashlib.sha256(args.arm_config.read_bytes()).hexdigest()
    report_path = args.output_dir / "campaign.json"
    reference = args.reference_quality
    key = os.environ.get("MI210_API_KEY", "<SET_YOUR_API_KEY>")

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def emit(message):
        print(json.dumps(message), flush=True)

    save()
    for label in args.arms:
        name = "mi210-" + args.output_dir.name + "-" + label
        image = args.base if label == "original" else args.image
        image_id = docker_json("image", "inspect", image)[0]["Id"]
        if subprocess.run(["docker", "inspect", name], capture_output=True).returncode == 0:
            raise RuntimeError(f"Container already exists: {name}")
        output = args.output_dir / label
        output.mkdir()
        # Commit snapshots can contain the previous run's /tmp artifacts.
        # Each new campaign needs its own remote namespace as well.
        remote = f'/tmp/mi210-measurements-{args.output_dir.name}-{label}'
        env = dict(os.environ)
        for variable in ("SPEC_TOKENS", "SPEC_METHOD", "PREFIX_CACHING", "MI210_TUNED_CK", "MI210_QUERY_TILE", "MI210_KV_TILE", "MI210_FLASH_SPLITS",
                         "MI210_INT8_LM_HEAD", "MI210_PROFILE", "MI210_GDN_TINY_BA", "MI210_GDN_ZVIEW",
                         "MI210_NATIVE_GDN", "MI210_NATIVE_GDN_RAW", "MI210_INT8_QK", "MI210_TARGET_WAVE_LAYOUT",
                         "MI210_DRAFT_GLUON", "MI210_DRAFT_SPLITS",
                         "MI210_REGISTER_QUANT_ALIAS", "MI210_EXTENDED_CK",
                         "MI210_QKPV_INTERLEAVE", "MI210_REDUCTION_POLICY"):
            env.pop(variable, None)
        env.update(settings[label], MI210_IMAGE=image_id, MI210_NAME=name,
                   MI210_HOST="127.0.0.1", PORT="8080", MI210_API_KEY=key,
                   MAX_MODEL_LEN="131072", BATCHED_TOKENS="2048",
                   MI210_UA_3D_MAXQ=settings[label].get("MI210_UA_3D_MAXQ", "16"),
                   GPU_NODES="/dev/dri/renderD128",
                   MI210_COMPILE_CACHE=f"/data/cache/mi210-vllm/{name}",
                   MI210_TRITON_CACHE="/data/cache/mi210-vllm/triton-cdna2")
        row = {"label": label, "container": name, "image_id": image_id,
               "settings": settings[label], "execution": args.execution, "status": "STARTING"}
        report["arms"].append(row)
        save()
        emit({"arm": label, "phase": "launch", "image_id": image_id})
        launched = False
        try:
            subprocess.run(["bash", str(ROOT / "scripts/serve_mi210_long_context.sh"), args.execution],
                           env=env, check=True, stdout=subprocess.DEVNULL)
            launched = True
            # Retain the experiment container without starting it on a host reboot.
            subprocess.run(["docker", "update", "--restart=no", name], check=True,
                           stdout=subprocess.DEVNULL)
            start = time.monotonic()
            while True:
                state = docker_json("inspect", name)[0]
                if not state["State"]["Running"] or state["RestartCount"]:
                    raise RuntimeError("Worker exited or restarted during startup")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() - start > 1200:
                    raise TimeoutError("Worker health timeout")
                time.sleep(5)
            row["startup_seconds"] = time.monotonic() - start
            subprocess.run(["docker", "cp", str(Path(__file__).parent) + '/.',
                            f"{name}:{TOOLS}"], check=True)

            def execute(phase, *argv):
                emit({"arm": label, "phase": phase})
                with (output / (phase + ".log")).open("w") as logfile:
                    command = ["docker", "exec", "-e", "MI210_API_KEY=" + key, name, PYTHON, *argv]
                    with subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT) as process:
                        while True:
                            try:
                                code = process.wait(timeout=5)
                                if code:
                                    raise subprocess.CalledProcessError(code, command)
                                break
                            except subprocess.TimeoutExpired:
                                logs = subprocess.run(["docker", "logs", "--tail", "16", name],
                                                      capture_output=True, text=True)
                                if any(marker in logs.stdout + logs.stderr for marker in
                                       ("Memory access fault by GPU", "GPU core dump failed", "EngineCore encountered a fatal error")):
                                    process.terminate()
                                    try:
                                        process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                    raise RuntimeError("GPU/worker fatal error during " + phase)

            execute("warmup", TOOLS + "/benchmark.py", "--url", "http://127.0.0.1:8080",
                    "--model", "qwen-ptqr", "--concurrency", "1", "--requests", "2",
                    "--max-tokens", "128", "--repetitions", "1", "--output", remote + "/warmup.jsonl")
            quality_args = []
            if reference:
                subprocess.run(["docker", "cp", str(reference), name + ":/tmp/reference-quality.json"], check=True)
                quality_args = ["--reference", "/tmp/reference-quality.json"]
            execute("quality", TOOLS + "/quality_probe.py", "--url", "http://127.0.0.1:8080",
                    "--output", remote + "/quality.json", *quality_args)
            execute("smoke", TOOLS + "/smoke.py", "--url", "http://127.0.0.1:8080",
                    "--output", remote + "/smoke.json")
            for workload in ("short", "long"):
                workload_args = ["--short", "--requests", "4"] if workload == "short" else ["--depths", "48000"]
                execute(workload, TOOLS + "/runtime_campaign.py", "--label", label,
                        "--image-id", image_id, "--output-dir", remote + "/" + workload,
                        "--repetitions", str(args.repetitions), "--max-tokens", "512", *workload_args)
            if args.capture_fidelity or args.arm_config:
                execute('fidelity', TOOLS + '/fidelity_compare.py', 'capture',
                        '--image-id', image_id, '--output', remote + '/fidelity.json')
                row['fidelity_scope'] = '52 diverse fixed prompts; token-ID top-20 logprobs'
            if args.extended_validation:
                execute("parallel", TOOLS + "/parallel_smoke.py", "--url", "http://127.0.0.1:8080",
                        "--output", remote + "/parallel.json")
                execute("retrieval120k", TOOLS + "/long_quality.py", "--output", remote + "/retrieval120k.json")
                execute("generation120k", TOOLS + "/runtime_campaign.py", "--label", label,
                        "--image-id", image_id, "--output-dir", remote + "/generation120k",
                        "--repetitions", "3", "--max-tokens", "512", "--depths", "120000")
            row["status"] = "MEASURED"
        except Exception as error:
            row.update(status="FAILED", error=str(error).replace(key, "<redacted>"))
            emit({"arm": label, "phase": "failed", "error": row["error"]})
        finally:
            if launched:
                # Persist the serving logs even when a gate or benchmark failed.
                logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
                (output / "server.log").write_text((logs.stdout + logs.stderr).replace(key, "<redacted>"))
                subprocess.run(["docker", "cp", name + ':' + remote + '/.', str(output)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["docker", "stop", "-t", "30", name], check=True, stdout=subprocess.DEVNULL)
            if (output / "quality.json").exists():
                quality = json.loads((output / "quality.json").read_text())
                row["quality"] = {k: v for k, v in quality.items() if k != "records"}
                if reference is None and quality["passed"] == quality["total"]:
                    reference = output / "quality.json"
            row["measurements"] = summarize(output)
            row["regressions"] = {}
            for name in ("parallel", "retrieval120k"):
                if (output / (name + ".json")).exists():
                    records = json.loads((output / (name + ".json")).read_text())
                    row["regressions"][name] = {"passed": sum(r["passed"] for r in records),
                                               "total": len(records)}
            save()
            emit(row)
        if row["status"] == "FAILED":
            report["status"] = "FAILED"
            save()
            raise SystemExit(1)
    report["status"] = "MEASUREMENTS_COMPLETE_SELECTION_PENDING"
    save()


if __name__ == "__main__":
    main()
