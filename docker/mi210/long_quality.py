"""Check exact retrieval from three positions in a 120K-token input."""
import argparse
import json
import os
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=120000)
    args = parser.parse_args()
    if not 1024 <= args.tokens <= 128000:
        parser.error('tokens must be in [1024, 128000]')
    tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.8-27B-PTQR-R10S60',
                                              local_files_only=True)
    filler = tokenizer.encode('The notebook contains ordinary background text. ' *
                               (args.tokens // 4 + 1000), add_special_tokens=False)[:args.tokens]
    rows = []
    with httpx.Client(timeout=1800, headers={
        'Authorization': 'Bearer ' + os.environ['MI210_API_KEY']}) as client:
        for index, fraction in enumerate((0.0, 0.5, 1.0)):
            code = ('739184', '826395', '951762')[index]
            cut = int(len(filler) * fraction)
            prompt = ('Read the following notes and remember the access code.\n' +
                      tokenizer.decode(filler[:cut]) + f'\nThe access code is {code}.\n' +
                      tokenizer.decode(filler[cut:]) +
                      '\nReturn only the access code from the notes, without explanation.')
            response = client.post(args.url.rstrip('/') + '/v1/chat/completions', json={
                'model': 'qwen-ptqr', 'temperature': 0, 'max_tokens': 64,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': prompt}]})
            response.raise_for_status()
            body = response.json()
            output = body['choices'][0]['message']['content'].strip()
            row = dict(position=fraction, expected=code, output=output,
                       usage=body['usage'], passed=output == code and
                       body['usage']['prompt_tokens'] >= args.tokens)
            rows.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if not all(row['passed'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
