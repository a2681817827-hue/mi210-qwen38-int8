"""Execute the saved serial resume plan on the original MI210 host."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify_stage(stage, directory):
    argv = stage["argv"]
    if argv[0] == "docker":
        result = directory / Path(argv[argv.index("--output") + 1]).name
        if result.suffix == ".jsonl":
            rows = [json.loads(line) for line in result.read_text().splitlines()
                    if line.strip()]
            if len(rows) != 1:
                raise RuntimeError("Expected exactly one graph gate result")
            row = rows[0]
            if (row.get("candidate_launches", 0) < 1
                    or row.get("ragged_numeric") != "PASS"
                    or row.get("graph_replay") != "PASS"):
                raise RuntimeError("Native numeric/graph gate failed")
        else:
            row = json.loads(result.read_text())
            cases = row.get("cases", [])
            if (row.get("status") != "PASS" or len(cases) != 20
                    or row.get("stateful_replay_checks", 0) < 20
                    or any(case.get("numeric") != "PASS"
                           or case.get("graph_replay") != "PASS"
                           for case in cases)):
                raise RuntimeError("Native GDN state/output/graph gate failed")
    else:
        output = Path(argv[argv.index("--output-dir") + 1])
        report = json.loads((output / "campaign.json").read_text())
        labels = argv[argv.index("--arms") + 1:argv.index("--reference-quality")]
        arms = report.get("arms", [])
        if (report.get("status") != "MEASUREMENTS_COMPLETE_SELECTION_PENDING"
                or [arm["label"] for arm in arms] != labels
                or any(arm.get("status") != "MEASURED" for arm in arms)):
            raise RuntimeError("Serving campaign incomplete")
        for arm in arms:
            quality = arm.get("quality", {})
            if quality.get("passed") != 52 or quality.get("total") != 52:
                raise RuntimeError("Answer quality gate failed")
            for workload in ("short", "long"):
                if workload not in arm.get("measurements", {}):
                    raise RuntimeError("Missing workload measurements")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    directory = args.plan.resolve().parent
    root = Path(__file__).resolve().parents[2]
    seen = {"HOST_READY_GPU_IDLE"}
    for stage in plan["stages"]:
        if stage["name"] in seen or not set(stage["requires"]) <= seen:
            parser.error("Duplicate stage or out-of-order dependency")
        seen.add(stage["name"])
    if not args.execute:
        print(json.dumps({"stages": [s["name"] for s in plan["stages"]],
                          "execution": "NOT_RUN"}, indent=2))
        return
    state_path = directory / "execution.json"
    if state_path.exists():
        parser.error("Execution record exists; inspect it before creating a fresh plan")
    for name, expected in plan["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            parser.error("Source changed since plan creation: " + name)
    reference = Path(plan["reference_quality"])
    if hashlib.sha256(reference.read_bytes()).hexdigest() != plan["reference_sha256"]:
        parser.error("Quality reference changed")
    # All image IDs must be present before allocating the GPU.
    for image in sorted({stage["argv"][stage["argv"].index(
            "/opt/int8-vllm/.venv/bin/python") + 1]
            if stage["argv"][0] == "docker" else
            stage["argv"][stage["argv"].index("--image") + 1]
            for stage in plan["stages"]}):
        subprocess.run(["docker", "image", "inspect", image], check=True,
                       stdout=subprocess.DEVNULL)
    subprocess.run(plan["preflight_argv"], check=True, cwd=root)
    if subprocess.check_output(["docker", "ps", "-q"], text=True).strip():
        parser.error("Stop the previous GPU service before running this serial plan")
    report = {"status": "RUNNING", "stages": [], "performance_claim": None}

    def save():
        state_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for stage in plan["stages"]:
        row = {"name": stage["name"], "status": "RUNNING"}
        report["stages"].append(row)
        save()
        print("Starting " + stage["name"], flush=True)
        try:
            with (directory / (stage["name"] + ".log")).open("x") as log:
                subprocess.run(stage["argv"], check=True, cwd=root,
                               stdout=log, stderr=subprocess.STDOUT)
            verify_stage(stage, directory)
        except Exception as error:
            row.update(status="FAILED", error=str(error))
            report["status"] = "FAILED"
            save()
            raise
        row["status"] = "COMPLETE"
        save()
        print("Completed " + stage["name"], flush=True)
    report["status"] = "MEASUREMENTS_COMPLETE_SELECTION_PENDING"
    report["remaining"] = plan["final_validation_pending"]
    save()
    print("Measurements complete; winner selection and final validation pending.")


if __name__ == "__main__":
    main()
