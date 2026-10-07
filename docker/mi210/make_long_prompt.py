"""Recreate the measured synthetic long-context retrieval/generation prompt."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', type=int, required=True)
parser.add_argument('--model', default='/models/Qwen3.8-27B-PTQR-R10S60')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.tokens < 1:
    parser.error('--tokens must be positive')
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
background = 'This line contains ordinary background text. ' * (args.tokens // 4 + 1000)
tokens = tokenizer.encode(background, add_special_tokens=False)[:args.tokens]
prompt = 'Remember the secret code MI210-7391. ' + tokenizer.decode(tokens)
prompt += ' What was the secret code? Then explain matrix multiplication with examples in detail.'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(prompt.replace('\n', ' ') + '\n')
print({'raw_prompt_tokens': len(tokenizer.encode(prompt)),
       'note': 'Authoritative API token count also includes the chat template.'})
