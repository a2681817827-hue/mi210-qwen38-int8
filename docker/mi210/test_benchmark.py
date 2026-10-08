import unittest

import httpx

from benchmark import request


class StreamingUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_count_comes_from_usage_not_chunk_count(self):
        body = ('data: {"choices":[{"delta":{"content":"three tokens here"}}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":3,"prompt_tokens":9}}\n\n'
                'data: [DONE]\n\n')
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            sample = await request(client, "http://worker", "model", "prompt", 10)
        self.assertEqual(sample["usage"]["completion_tokens"], 3)
        self.assertEqual(sample["output"], "three tokens here")

    async def test_missing_usage_fails_instead_of_counting_chunks(self):
        body = 'data: {"choices":[{"delta":{"content":"text"}}]}\n\ndata: [DONE]\n\n'
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaisesRegex(RuntimeError, "completion_tokens"):
                await request(client, "http://worker", "model", "prompt", 10)
