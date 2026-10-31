"""Execute an actual notebook training cell with bounded or original settings."""
import argparse
import ast
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from IPython.core.interactiveshell import InteractiveShell
    parser = argparse.ArgumentParser()
    parser.add_argument('chapter', type=int)
    parser.add_argument('cell', type=int)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'mps'))
    parser.add_argument('--profile', default='bounded', choices=('bounded', 'full'))
    parser.add_argument('--seed', type=int, default=12)
    args = parser.parse_args()
    suffix = '-local' if args.chapter >= 8 else ''
    notebook = json.loads((ROOT / 'notebooks' / f'chapter_{args.chapter:02}' / f'chapter-{args.chapter:02}{suffix}.ipynb').read_text())
    shell = InteractiveShell.instance()
    run_name = f'ch{args.chapter}-cell{args.cell}-{args.device}'
    if args.profile == 'full' or args.seed != 12:
        run_name += f'-{args.profile}-seed{args.seed}'
    run_dir = ROOT / 'validation' / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    os.environ['DRL_BOOK_DEVICE'] = args.device

    def run(source, label):
        result = shell.run_cell(source, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f'{label}: {error}') from error

    run(''.join(notebook['cells'][2]['source']).replace('%matplotlib inline', ''), 'imports')
    for index, cell in enumerate(notebook['cells'][3:args.cell], start=3):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        definitions = tree.body and all(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body)
        support_import = source.startswith('from drl_book.')
        if definitions or support_import:
            run(source, f'definitions cell {index}')
    source = ''.join(notebook['cells'][args.cell]['source'])
    if args.profile == 'bounded':
        source = re.sub(r"('max_episodes'\s*:\s*)\d+", rf'\g<1>{args.episodes}', source)
        source = re.sub(r"('max_minutes'\s*:\s*)\d+", r'\g<1>2', source)
        source = source.replace('batch_size = 1024', 'batch_size = 8').replace('epochs = 40', 'epochs = 1')
        source = source.replace('batch_size=256', 'batch_size=8').replace('batch_size=64', 'batch_size=8')
        source = source.replace('n_warmup_batches = 10', 'n_warmup_batches = 1').replace('n_warmup_batches = 5', 'n_warmup_batches = 1')
        source = source.replace('n_workers = 8', 'n_workers = 2').replace('max_buffer_episodes = 32', 'max_buffer_episodes = 4')
    source = source.replace('_ = BEEP()', 'pass')
    shell.user_ns['SEEDS'] = (args.seed,)
    run(source, f'chapter {args.chapter}, training cell {args.cell}')
    agent = shell.user_ns['agent']
    models = [getattr(agent, name) for name in ('online_model','online_policy_model','online_value_model','policy_model','online_value_model_a','value_model','ac_model') if hasattr(agent,name)]
    if args.device == 'mps':
        assert models and all(next(model.parameters()).device.type == 'mps' for model in models)
    print('MODEL DEVICES', [str(next(model.parameters()).device) for model in models], flush=True)
    evidence = {'chapter': args.chapter, 'cell': args.cell, 'profile': args.profile,
                'seed': args.seed, 'device': args.device, 'execution': 'passed',
                'final_eval_score': float(shell.user_ns['final_eval_score']),
                'training_seconds': float(shell.user_ns['training_time']),
                'wallclock_seconds': float(shell.user_ns['wallclock_time'])}
    (run_dir / 'result.json').write_text(json.dumps(evidence, indent=2) + '\n')
    settings = f'episodes={args.episodes}' if args.profile == 'bounded' else 'original-settings'
    print(f'PASS TRAINING chapter={args.chapter} cell={args.cell} {settings} seed={args.seed} device={args.device}', flush=True)


if __name__ == '__main__':
    main()
