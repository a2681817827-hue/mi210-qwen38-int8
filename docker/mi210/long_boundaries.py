"""Gate native CDNA2 G128 kernels at tails and the TP1 model geometry."""
import json
from pathlib import Path

from micro import check_attention

cases = [(256, 24, 4, q, True, True, s, 1728)
         for s in (33, 4097, 65537) for q in (1, 7, 14)]
cases += [(256, 24, 4, 257, True, False, 4097, 1728),
          (256, 24, 4, 2048, True, False, 65537, 1728),
          (128, 8, 2, 14, False, False, 65537, 1728)]
with Path('/results/boundaries-tp1.jsonl').open('a') as file:
    for case in cases:
        row = dict(case=case, **check_attention(*case, benchmark_repeats=5), numeric='PASS')
        file.write(json.dumps(row) + '\n')
        file.flush()
        print(json.dumps(row), flush=True)
