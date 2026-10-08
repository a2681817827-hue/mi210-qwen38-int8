"""Measure the existing small BF16 gate projection on MI210, with graph replay."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from vllm.model_executor.layers.mamba.gdn.gfx908_tiny_ba import tiny_ba_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
    torch.manual_seed(741)
    records = []
    for rows in (1, 8, 12, 14, 16, 18, 32, 64):
        x = (torch.randn(rows, 5120, device='cuda') * 0.1).bfloat16()
        weight = (torch.randn(96, 5120, device='cuda') * 0.1).bfloat16()
        reference = x.float() @ weight.float().T
        output = tiny_ba_projection(x, weight)
        error = (torch.linalg.vector_norm(output.float() - reference) /
                 torch.linalg.vector_norm(reference)).item()
        torch.testing.assert_close(output.float(), reference, rtol=0.008, atol=0.003)
        assert error < 0.003

        def measure(operation):
            for _ in range(3):
                operation()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = operation()
            captured.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured.float(), reference, rtol=0.008, atol=0.003)
            timings = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) / 20)
            return statistics.median(timings), timings

        baseline, baseline_raw = measure(lambda: x @ weight.T)
        candidate, candidate_raw = measure(lambda: tiny_ba_projection(x, weight))
        records.append(dict(rows=rows, cols=96, inner=5120, relative_l2=error,
                            numeric='PASS', graph_replay='PASS', baseline_ms=baseline,
                            candidate_ms=candidate, speedup=baseline / candidate,
                            baseline_raw_ms=baseline_raw, candidate_raw_ms=candidate_raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
