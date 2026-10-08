"""Inventory retained experiment containers, images, and raw artifact hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def docker(*args):
    return json.loads(subprocess.check_output(["docker", *args], text=True))


def redact_command(command):
    result = list(command or [])
    for index, argument in enumerate(result):
        if argument == "--api-key" and index + 1 < len(result):
            result[index + 1] = "<redacted>"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    campaigns = sorted(directory.glob("*/campaign.json"))
    names = set()
    measurements = []
    names.update(subprocess.check_output(
        ["docker", "ps", "-a", "--filter", "name=mi210", "--format", "{{.Names}}"],
        text=True).splitlines())
    for path in campaigns:
        report = json.loads(path.read_text())
        for arm in report.get("arms", []):
            if arm.get("container"):
                names.add(arm["container"])
            measurements.append({"campaign": str(path.relative_to(directory)), **arm})
    containers = []
    image_ids = set()
    image_ids.update(subprocess.check_output(
        ["docker", "image", "ls", "--no-trunc", "--filter",
         "reference=int8-vllm-mi210:*", "--format", "{{.ID}}"], text=True).splitlines())
    for name in sorted(names):
        info = docker("inspect", name)[0]
        image_ids.add(info["Image"])
        containers.append({
            "name": name, "id": info["Id"], "image_id": info["Image"],
            "state": info["State"], "restart_policy": info["HostConfig"]["RestartPolicy"],
            "command": redact_command(info["Config"]["Cmd"]),
            "entrypoint": info["Config"]["Entrypoint"], "mounts": info["Mounts"],
            "devices": info["HostConfig"]["Devices"],
            "operator_env": [value for value in info["Config"]["Env"]
                             if value.startswith(("VLLM_", "HIP_", "GPU_", "AITER_"))
                             and not any(key in value.split("=", 1)[0]
                                         for key in ("KEY", "TOKEN", "SECRET"))],
        })
    images = []
    for image_id in sorted(image_ids):
        info = docker("image", "inspect", image_id)[0]
        images.append({key: info[key] for key in ("Id", "RepoTags", "Created", "Size")})
    output = directory / "retained-results.json"
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != output and path.suffix not in (".tar", ".gz"):
            artifacts.append({"path": str(path.relative_to(directory)),
                              "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {"policy": "Retain successful containers and images; stop services after testing",
              "containers": containers, "images": images,
              "measurements": measurements, "artifacts": artifacts,
              "external_assets": "Bind-mounted models and compilation caches remain under /data; "
                                 "container retention does not copy those assets."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "containers": len(containers),
                      "images": len(images), "artifacts": len(artifacts)}))


if __name__ == "__main__":
    main()
