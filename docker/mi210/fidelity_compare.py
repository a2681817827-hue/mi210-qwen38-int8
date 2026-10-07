"""Capture greedy token distributions; compare only matching model histories.

Top-20 distributions yield coarsened KL (a lower bound on full-vocabulary
KL), never a claim of full-vocabulary KL or a downstream accuracy score.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import urllib.request


def coarsened_kl(reference, candidate):
    if any(not math.isfinite(value) or value > 1e-6
           for row in (reference, candidate) for value in row.values()):
        raise ValueError('Invalid log probabilities')
    common = reference.keys() & candidate.keys()
    if not common:
        raise ValueError('No common token IDs')
    p = [math.exp(reference[k]) for k in sorted(common)]
    q = [math.exp(candidate[k]) for k in sorted(common)]
    if sum(p) > 1.00001 or sum(q) > 1.00001:
        raise ValueError('Invalid probability mass')
    p.append(max(0.0, 1.0 - sum(p)))
    q.append(max(0.0, 1.0 - sum(q)))
    return max(0.0, sum(a * math.log(a / max(b, 1e-30))
                        for a, b in zip(p, q) if a > 0)), len(common)


def compare(reference, candidate):
    if reference['corpus_sha256'] != candidate['corpus_sha256']:
        raise ValueError('Different evaluation prompts')
    left, right = reference['records'], candidate['records']
    if len(left) != len(right) or not left:
        raise ValueError('Missing evaluation records')
    divergences, kls, supports = [], [], []
    exact = 0
    for a, b in zip(left, right):
        if a['prompt_sha256'] != b['prompt_sha256']:
            raise ValueError('Different prompt or record order')
        if len(a['tokens']) != len(a['top_logprobs']) or len(b['tokens']) != len(b['top_logprobs']):
            raise ValueError('Incomplete token logprobs')
        exact += a['tokens'] == b['tokens']
        count = 0
        # At the first divergent token the preceding history is still equal,
        # so that position is comparable; subsequent positions are excluded.
        for pos in range(min(len(a['tokens']), len(b['tokens']))):
            kl, support = coarsened_kl(a['top_logprobs'][pos], b['top_logprobs'][pos])
            kls.append(kl)
            supports.append(support)
            count += 1
            if a['tokens'][pos] != b['tokens'][pos]:
                break
        if not count:
            raise ValueError('Empty model output')
        divergences.append({'index': a['index'], 'comparable_positions': count,
                            'reference_tokens': len(a['tokens']), 'candidate_tokens': len(b['tokens'])})
    return {'scope': f'{len(left)} diverse prompts; shared-history top-20 coarsened KL',
            'limitation': 'Lower bound on full-vocabulary KL; divergent histories excluded. '
                          'This is not a downstream task accuracy benchmark.',
            'reference_image': reference['image_id'], 'candidate_image': candidate['image_id'],
            'exact_greedy_token_sequences': exact, 'total_prompts': len(left),
            'comparable_positions': len(kls), 'mean_coarsened_kl': statistics.mean(kls),
            'comparable_post_prefill_positions': sum(max(0, r['comparable_positions'] - 1)
                                                     for r in divergences),
            'median_coarsened_kl': statistics.median(kls),
            'minimum_common_token_support': min(supports), 'positions': divergences}


def capture(args):
    tree = ast.parse(args.corpus.read_text())
    prompts = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'PROMPTS'
                           for target in node.targets))
    background = ('The notes below contain ordinary background text. ' * 256)
    prompts = [background + '\nIgnore the background notes. Respond to this request:\n' + p
               for p in prompts]
    corpus_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    report = {'status': 'RUNNING', 'image_id': args.image_id,
              'corpus_sha256': corpus_hash, 'max_tokens': args.max_tokens,
              'records': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Use a fresh output file')
    key = os.environ.get('MI210_API_KEY')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    for index, prompt in enumerate(prompts):
        payload = {'model': 'qwen-ptqr', 'prompt': prompt, 'temperature': 0,
                   'seed': 20261007, 'max_tokens': args.max_tokens, 'logprobs': 20,
                   'return_tokens_as_token_ids': True, 'return_token_ids': True}
        request = urllib.request.Request(args.url.rstrip('/') + '/v1/completions',
                                         data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.load(response)
        choice = body['choices'][0]
        tokens = choice.get('token_ids')
        probs = (choice.get('logprobs') or {}).get('top_logprobs')
        if not tokens or not probs or len(tokens) != len(probs):
            raise ValueError('Incomplete token-ID logprob response')
        if any(not row or any(not k.startswith('token_id:') for k in row) for row in probs):
            raise ValueError('Stable token IDs are required for distribution comparison')
        report['records'].append({'index': index, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                                  'tokens': tokens, 'top_logprobs': probs,
                                  'text': choice['text'], 'usage': body['usage']})
        args.output.write_text(json.dumps(report, ensure_ascii=False) + '\n')
        print(json.dumps({'prompt': index, 'tokens': len(tokens)}), flush=True)
    report['status'] = 'COMPLETE'
    args.output.write_text(json.dumps(report, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'compare'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--image-id')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--corpus', type=Path,
                        default=Path('/opt/int8-vllm/scripts/kld_probe_v2.py'))
    args = parser.parse_args()
    if args.mode == 'capture':
        if not args.image_id or args.max_tokens < 14:
            parser.error('Supply the image ID and at least 14 generation tokens')
        capture(args)
    else:
        if not args.reference or not args.candidate:
            parser.error('Supply reference and candidate captures')
        reference = json.loads(args.reference.read_text())
        candidate = json.loads(args.candidate.read_text())
        if reference['status'] != 'COMPLETE' or candidate['status'] != 'COMPLETE':
            parser.error('Incomplete capture cannot qualify')
        result = compare(reference, candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            json.dump(result, file, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != 'positions'}))


if __name__ == '__main__':
    main()
