# SPDX-License-Identifier: Apache-2.0
"""Launch a MI210 PTQR candidate inside an already device-isolated container."""

import argparse
import json
import os
from pathlib import Path


def command(profile: str, model: str | None = None, draft: str | None = None,
            extra: list[str] | None = None,
            spec_tokens: int = 13,
            attention_profile: str = "generic") -> tuple[dict, list[str]]:
    config = json.loads(Path(__file__).with_name("profiles.json").read_text())
    p = config["profiles"][profile]
    if attention_profile not in {"generic", "cdna2", "cdna2-tp2-experimental"}:
        raise ValueError("Unknown attention_profile")
    if attention_profile != "generic":
        if attention_profile == "cdna2" and p["tp"] != 1:
            raise ValueError("Native CDNA2 24:4 attention is validated for TP1 only")
        if attention_profile == "cdna2-tp2-experimental" and p["tp"] != 2:
            raise ValueError("The experimental 12:2 attention profile requires TP2")
        config["env"].update({
            "VLLM_UA_3D_MAXQ": "16", "VLLM_UA_TILE": "32",
            "VLLM_G128_GLUON": "64", "VLLM_G128_GLUON_MMA": "bf16",
            "VLLM_G128_DECODE_CDNA2": "1", "VLLM_G128_DECODE_CDNA2_TILE": "64",
            "VLLM_G128_PREFILL_GLUON": "1", "VLLM_G128_PREFILL_GLUON_MMA": "bf16",
            "VLLM_G128_PREFILL_CDNA2": "1", "VLLM_G128_PREFILL_CDNA2_TILE": "64",
        })
        if attention_profile == "cdna2-tp2-experimental":
            config["env"]["VLLM_MI210_TP2_CDNA2"] = "1"
    if not 1 <= spec_tokens <= 32:
        raise ValueError("spec_tokens must be in [1, 32]")
    spec = {**config["speculative"], "model": draft or config["draft"],
            "num_speculative_tokens": spec_tokens}
    argv = ["/opt/int8-vllm/.venv/bin/vllm", "serve", model or config["model"],
            "--host", "0.0.0.0", "--port", "8000",
            "--tensor-parallel-size", str(p["tp"]),
            *config["common_args"], "--speculative-config", json.dumps(spec),
            *(extra or [])]
    return config["env"], argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["single", "replicas2", "tp2"])
    parser.add_argument("--model")
    parser.add_argument("--draft")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--spec-tokens", type=int, default=13)
    parser.add_argument("--attention-profile", choices=("generic", "cdna2",
                                                       "cdna2-tp2-experimental"),
                        default="generic")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    env, argv = command(args.profile, args.model, args.draft, extra, args.spec_tokens,
                        args.attention_profile)
    for override in args.env:
        if "=" not in override:
            parser.error("--env requires KEY=VALUE")
        key, value = override.split("=", 1)
        if not key:
            parser.error("--env requires a nonempty key")
        env[key] = value
    if args.print_config:
        print(json.dumps({"env": env, "argv": argv, "hardware_validated": False},
                         indent=2))
    else:
        os.environ.update(env)
        # Visibility is defined by the device nodes supplied by the host.
        if not os.environ.get("HIP_VISIBLE_DEVICES"):
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.execv(argv[0], argv)
