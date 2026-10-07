"""Install opt-in, numerically gated CK dispatch for exact MI210 shapes."""
import argparse
import hashlib
import json
from pathlib import Path


def patch(root, report_path):
    report = json.loads(report_path.read_text())
    if (report['status'] != 'OPERATOR_GATES_COMPLETE' or report['arch'] != 'gfx90a'
            or report['cu_num'] != 104 or report['scale_dtype'] != 'float32'
            or report['output_dtype'] != 'bfloat16' or report['timing'] != 'graph'):
        raise ValueError('Completed MI210 FP32-scale/BF16 graph gates required')
    trials = [json.loads(line) for line in report_path.with_name('trials.jsonl').read_text().splitlines()]
    selections = {}
    for row in report['selections']:
        if row['kernel_id'] is None:
            continue
        if row['baseline_numeric'] != 'PASS' or row['graph_replay'] != 'PASS':
            raise ValueError('Missing baseline/graph gate')
        if row['selected_ms'] > row['baseline_ms'] / 1.05:
            raise ValueError('Selected operator must improve by at least 5%')
        matching = [trial for trial in trials if all(trial[key] == row[key]
                    for key in ('M', 'N', 'K', 'kernel_id', 'splitK'))]
        if len(matching) != 1 or matching[0]['numeric'] != 'PASS' or matching[0]['graph_replay'] != 'PASS':
            raise ValueError('Selected trial has no numerical/replay proof')
        key = tuple(row[k] for k in ('M', 'N', 'K'))
        if key in selections:
            raise ValueError('Duplicate selection')
        selections[key] = (row['kernel_id'], row['splitK'], row['kernel_name'])
    if not selections:
        raise ValueError('No operator passed the improvement gate')
    target = root / 'aiter/ops/gemm_op_a8w8.py'
    body = target.read_text()
    marker = '# MI210_GATED_CK_DISPATCH'
    if marker in body:
        raise ValueError('Dispatch already installed; use the original validated source')
    for relative, expected in report['aiter_source_sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Tuned AITER source drift: ' + relative)
    declaration = ('\n' + marker + '\n_MI210_CK_SELECTED = ' + repr(selections) +
                   '\n_MI210_CK_SEEN = set()\n\n')
    anchor = 'def gemm_a8w8_CK(\n'
    if body.count(anchor) != 1:
        raise ValueError('CK wrapper source drift')
    body = body.replace(anchor, declaration + anchor)
    start = body.index(anchor)
    prefix, wrapper = body[:start], body[start:]
    anchor = '    q_dtype_w = WQ.dtype if WQ.dtype in [dtypes.fp8, dtypes.i8] else dtypes.i8\n'
    if wrapper.count(anchor) != 1:
        raise ValueError('CK dispatch anchor drift')
    dispatch = '''    if (
        os.getenv("AITER_MI210_TUNED_CK", "0") == "1"
        and bias is None and splitK is None
        and dtype == torch.bfloat16
        and XQ.dtype == torch.int8 and WQ.dtype == torch.int8
        and x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32
        and XQ.is_cuda and WQ.device == XQ.device
        and x_scale.device == XQ.device and w_scale.device == XQ.device
        and XQ.is_contiguous() and WQ.is_contiguous()
        and x_scale.is_contiguous() and w_scale.is_contiguous()
        and x_scale.numel() == m and w_scale.numel() == n
        and get_gfx() == "gfx90a" and get_cu_num() == 104
    ):
        selected = _MI210_CK_SELECTED.get((m, n, k))
        if selected is not None:
            kernel_id, selected_split, kernel_name = selected
            if (m, n, k) not in _MI210_CK_SEEN:
                logger.info(f"MI210 gated CK dispatch M={m} N={n} K={k} kernel={kernel_id} splitK={selected_split} name={kernel_name}")
                _MI210_CK_SEEN.add((m, n, k))
            Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
            return gemm_a8w8_tune(XQ, WQ, x_scale, w_scale, Y, kernel_id, selected_split)

'''
    target.write_text(prefix + wrapper.replace(anchor, dispatch + anchor))
    return {'selected_shapes': len(selections), 'enabled_by': 'AITER_MI210_TUNED_CK=1',
            'source_sha256_after': hashlib.sha256(target.read_bytes()).hexdigest(),
            'serving_validated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/opt/aiter'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch(args.root, args.report), indent=2))
