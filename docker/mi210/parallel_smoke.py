"""Check distinct answers across simultaneous TP1 decode requests."""
import asyncio
import argparse
import json
import os
from pathlib import Path

import httpx

async def main(args):
    semaphore = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=1800, headers={
        'Authorization': 'Bearer ' + os.environ['MI210_API_KEY'],
    }) as client:
        async def probe(n):
            async with semaphore:
                response = await client.post(args.url.rstrip('/') + '/v1/chat/completions', json={
                    'model': 'qwen-ptqr', 'temperature': 0, 'max_tokens': 48,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'messages': [{'role': 'user', 'content':
                                  f'Return only the result of {100+n} + {200+n}.'}],
                })
                response.raise_for_status()
                body = response.json()
                output = body['choices'][0]['message']['content'].strip()
                return dict(index=n, expected=str(300+2*n), output=output,
                            passed=output == str(300+2*n), usage=body['usage'])
        rows = await asyncio.gather(*(probe(n) for n in range(16)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(dict(passed=sum(r['passed'] for r in rows), total=len(rows))), flush=True)
    assert all(r['passed'] for r in rows), rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://10.168.1.4:8080')
    parser.add_argument('--output', type=Path, default=Path('/tmp/parallel-smoke.json'))
    asyncio.run(main(parser.parse_args()))
