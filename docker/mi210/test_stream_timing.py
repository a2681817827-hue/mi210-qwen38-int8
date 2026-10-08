"""Exercise stream timing without HTTP or GPU dependencies."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock


class StreamTimingTests(unittest.IsolatedAsyncioTestCase):
    async def sample(self, contents):
        # Load the real coroutine; the host need not have the serving HTTP stack.
        path = Path(__file__).with_name("benchmark.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body
                        if isinstance(n, ast.AsyncFunctionDef) and n.name == "request")
        clock = Mock(side_effect=[0, *range(1, len(contents) + 1), 100])
        namespace = {"json": json, "time": SimpleNamespace(perf_counter=clock)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
             namespace)

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for content in contents:
                    yield "data: " + json.dumps({"choices": [
                        {"delta": {"content": content}}]})
                yield 'data: {"choices":[],"usage":{"completion_tokens":5,"prompt_tokens":9}}'
                yield "data: [DONE]"

        client = SimpleNamespace(stream=lambda *a, **kw: Response())
        return await namespace["request"](client, "http://worker", "model", "prompt", 5)

    async def test_usage_tail_does_not_inflate_generation_time(self):
        sample = await self.sample(["first ", "last"])
        self.assertEqual(sample["seconds"], 100)
        self.assertEqual(sample["ttft_seconds"], 1)
        self.assertEqual(sample["generation_seconds"], 1)
        self.assertEqual(sample["tpot_estimate_seconds"], 0.25)
        self.assertEqual(sample["output"], "first last")

    async def test_single_speculative_chunk_has_no_generation_estimate(self):
        sample = await self.sample(["five tokens in one chunk"])
        self.assertIsNone(sample["tpot_estimate_seconds"])
        self.assertEqual(sample["usage"]["completion_tokens"], 5)

    async def test_no_content_has_no_generation_estimate(self):
        sample = await self.sample([])
        self.assertIsNone(sample["generation_seconds"])
        self.assertIsNone(sample["tpot_estimate_seconds"])

    def test_summary_separates_historical_and_corrected_timings(self):
        path = Path(__file__).with_name("summarize_depth.py")
        spec = importlib.util.spec_from_file_location("summary", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = {"label": "control", "model": "model", "image_id": "sha256:test",
               "concurrency": 1, "repetition": 0,
               "wall_output_tokens_per_second": 3,
               "samples": [{"usage": {"prompt_tokens": 48081, "completion_tokens": 512},
                            "tpot_estimate_seconds": 0.02, "ttft_seconds": 30}]}
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "records.jsonl"
            records.write_text(json.dumps(row) + "\n" + json.dumps(
                {**row, "timing_schema": "first_last_content_v2"}) + "\n")
            result = module.summarize([records])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual({r["timing_schema"] for r in result["results"]},
                         {"stream_end_v1", "first_last_content_v2"})


if __name__ == "__main__":
    unittest.main()
