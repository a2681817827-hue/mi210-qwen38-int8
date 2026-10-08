"""Clone a stopped MI210 test container with explicit experiment overrides."""
import argparse
import json
import subprocess


def inspect(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--spec-tokens", type=int, default=9)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.spec_tokens <= 32 or not 1 <= args.port <= 65535:
        parser.error("Invalid speculative count or port")
    meta = inspect(args.source)
    if meta["State"]["Running"]:
        raise RuntimeError("Stop the source service first; do not overlap experiments on one GPU")
    config, host = meta["Config"], meta["HostConfig"]
    if host["NetworkMode"] != "host" or len(config["Entrypoint"] or []) != 1:
        raise ValueError("This helper requires the existing single-entrypoint host-network recipe")
    cmd = list(config["Cmd"])
    cmd[cmd.index("--port") + 1] = str(args.port)
    si = cmd.index("--speculative-config") + 1
    spec = json.loads(cmd[si])
    spec["num_speculative_tokens"] = args.spec_tokens
    cmd[si] = json.dumps(spec)
    env = dict(entry.split("=", 1) for entry in config["Env"])
    selected = {}
    for override in args.env:
        key, value = override.split("=", 1)
        if not key.startswith(("VLLM_", "AITER_")):
            raise ValueError("Only operator environment overrides are supported")
        env[key] = value
        selected[key] = value
    create = ["docker", "create", "--name", args.name, "--restart", "unless-stopped",
              "--network", "host", "--ipc", host["IpcMode"],
              "--entrypoint", config["Entrypoint"][0]]
    for device in host["Devices"]:
        create += ["--device", f'{device["PathOnHost"]}:{device["PathInContainer"]}:{device["CgroupPermissions"]}']
    for bind in host["Binds"] or []:
        # Isolate compilation/JIT caches while retaining model and HF mounts.
        parts = bind.split(":")
        if parts[1] in {"/root/.cache/vllm", "/root/.triton", "/mi210-jit"}:
            parts[0] += "-" + args.name
        create += ["--volume", ":".join(parts)]
    for key, value in env.items():
        create += ["--env", key + "=" + value]
    create += [args.image, *cmd]
    print(json.dumps({"name": args.name, "source": args.source, "image": args.image,
                      "port": args.port, "spec_tokens": args.spec_tokens,
                      "operator_overrides": selected, "execute": args.execute}), flush=True)
    if args.execute:
        subprocess.run(create, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "start", args.name], check=True)


if __name__ == "__main__":
    main()
