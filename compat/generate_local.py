"""Export notebook definitions to importable modules for macOS spawn workers."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = '''# Derived from the author's notebook; local changes are described in LOCAL-SETUP.md.\n'''
SHARED_ADAM = '''class SharedAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                state['step'] = torch.zeros(()).share_memory_()
                state['exp_avg'] = torch.zeros_like(parameter).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(parameter).share_memory_()
                if group['amsgrad']:
                    state['max_exp_avg_sq'] = torch.zeros_like(parameter).share_memory_()
'''
SHARED_RMS = '''class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                state['step'] = torch.zeros(()).share_memory_()
                state['square_avg'] = torch.zeros_like(parameter).share_memory_()
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(parameter).share_memory_()
                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(parameter).share_memory_()
'''


def patched_definition(source, chapter, names):
    source = source.replace('.view(-1)', '.reshape(-1)')
    source = re.sub(r'np\.(int|float|bool|object)\b', lambda match: match.group(1), source)
    source = source.replace('np.Inf','np.inf').replace('np.array(self.experiences)', 'np.array(self.experiences, dtype=object)').replace('np.array(sample)', 'np.array(sample, dtype=object)')
    source = source.replace('device = "cpu"', 'device = device_for()')
    source = source.replace('self.training_strategy = training_strategy_fn(', 'self.training_strategy = self.training_strategy_fn(')
    source = source.replace('self.evaluation_strategy = evaluation_strategy_fn(', 'self.evaluation_strategy = self.evaluation_strategy_fn(')
    source = source.replace('dtype=np.float128', 'dtype=np.float64')
    source = source.replace('self.actions_mem[:] = np.nan', 'self.actions_mem[:] = 0')
    source = source.replace('.numpy().ipynb_checkpoints/squeeze()', '.numpy().squeeze()')
    if 'EpisodeBuffer' in names:
        # Methods must use instance state, not the notebook's global n_workers.
        lines = source.splitlines()
        in_method = False
        for index, line in enumerate(lines):
            if line.startswith('    def clear('):
                in_method = True
            if in_method:
                lines[index] = re.sub(r'(?<![\w.])n_workers\b', 'self.n_workers', line)
        source = '\n'.join(lines) + '\n'
    if 'get_make_env_fn' in names and chapter == 12:
        source = source.replace('        if render:', '        if render and not (sys.platform == \"darwin\" and \"Bullet\" in env_name):')
    if 'SharedAdam' in names:
        return SHARED_ADAM
    if 'SharedRMSprop' in names:
        return SHARED_RMS
    if 'MultiprocessEnv' in names:
        return 'MultiprocessEnv = ColumnSpawnEnvironments\n' if chapter == 11 else 'MultiprocessEnv = SpawnEnvironments\n'
    if names and names[0] in ('A3C', 'GAE'):
        source = re.sub(r'self\.(\w+_fn) = (\w+_fn)\b', r'self.\1 = SerializableCallable(\2)', source)
        source = source.replace("        self.stats['n_active_workers'].add_(1)", "        with self.get_out_lock:\n            self.stats['n_active_workers'].add_(1)")
        source = source.replace("        global_episode_idx = self.stats['episode'].add_(1).item() - 1", "        with self.get_out_lock:\n            if self.stats['episode'].item() >= self.max_episodes:\n                self.stats['n_active_workers'].sub_(1)\n                env.close()\n                return\n            global_episode_idx = self.stats['episode'].add_(1).item() - 1", 1)
        source = source.replace("        env.close() ; del env\n        self.stats['n_active_workers'].sub_(1)", "        env.close() ; del env\n        with self.get_out_lock:\n            self.stats['n_active_workers'].sub_(1)")
        source = source.replace("        while rank == 0 and self.stats['n_active_workers'].item() > 1:\n            pass", "        while rank == 0 and self.stats['n_active_workers'].item() > 1:\n            time.sleep(0.01)")
        source = source.replace('        self.shared_policy_optimizer.zero_grad()', '        self.shared_policy_optimizer.zero_grad()\n        local_policy_model.zero_grad()')
        source = source.replace('        self.shared_value_optimizer.zero_grad()', '        self.shared_value_optimizer.zero_grad()\n        local_value_model.zero_grad()')
        source = source.replace('[w.start() for w in workers] ; [w.join() for w in workers]',
                                '[w.start() for w in workers]\n        join_workers(workers, max_minutes * 60 + 120)')
    if 'SAC' in names:
        first = source.index('        self.policy_optimizer.zero_grad()', source.index('    def optimize_model'))
        end = source.index('        self.policy_optimizer.step()', first) + len('        self.policy_optimizer.step()')
        block = source[first:end]
        source = source[:first] + source[end:]
        insert = source.index('        self.value_optimizer_a.zero_grad()', source.index('    def optimize_model'))
        source = source[:insert] + '        # Backpropagate before changing critic parameters used by the policy graph.\n' + block + '\n\n' + source[insert:]
    return source


def main():
    for chapter in range(2, 13):
        folder = ROOT / 'notebooks' / f'chapter_{chapter:02}'
        book = json.loads((folder / f'chapter-{chapter:02}.ipynb').read_text())
        exported = []
        if chapter in (11, 12):
            imports = ''.join(book['cells'][2]['source']).replace('%matplotlib inline', '')
            exported = [HEADER, imports, '\nfrom .devices import device_for\n', '\nfrom .processes import SerializableCallable, SpawnEnvironments, ColumnSpawnEnvironments, join_workers\nmp = torch.multiprocessing.get_context("spawn")\n']
        for index, cell in enumerate(book['cells']):
            if cell['cell_type'] != 'code':
                continue
            cell['execution_count'] = None
            cell['outputs'] = []
            source = ''.join(cell['source'])
            source = source.replace('.numpy().ipynb_checkpoints/squeeze()', '.numpy().squeeze()')
            source = re.sub(r'np\.(int|float|bool|object)\b', lambda match: match.group(1), source)
            source = source.replace('np.Inf','np.inf').replace('np.array(self.experiences)', 'np.array(self.experiences, dtype=object)').replace('np.array(sample)', 'np.array(sample, dtype=object)')
            if chapter == 4 and index == 18:
                source = source.replace('    for e in tqdm(range(n_episodes),', '    min_temp = float(min_temp)  # Avoid NumPy scalar narrowing at the zero-temperature boundary.\n    for e in tqdm(range(n_episodes),')
            if chapter in (8, 9, 10):
                source = source.replace('device = "cpu"', 'device = device_for()')
                if index == 2: source += '\nfrom drl_book.devices import device_for\n'
            if source.strip() == '!nvidia-smi':
                source = 'import torch\nprint("PyTorch:", torch.__version__)\nprint("CUDA:", torch.cuda.is_available(), "MPS:", torch.backends.mps.is_available())'
            if chapter in (11, 12):
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    tree = None
                if tree and tree.body and all(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body):
                    names = [node.name for node in tree.body]
                    source = patched_definition(source, chapter, names)
                    exported.append(f'\n# Original notebook cell {index}\n{source}\n')
                    source = f'from drl_book.chapter{chapter} import ' + ', '.join(names)
            cell['source'] = source.splitlines(keepends=True)
        if exported:
            (ROOT / 'drl_book' / f'chapter{chapter}.py').write_text('\n'.join(exported))
        book['metadata']['kernelspec'] = {'display_name': 'GDRL (Python 3.11)', 'language': 'python', 'name': 'gdrl-python311'}
        # Keep cell numbers aligned with the author's original notebooks.
        first = book['cells'][0]
        first['source'] = ['# Local macOS edition. See LOCAL-SETUP.md. Definitions for spawn workers live in drl_book/.\n'] + first.get('source', [])
        (folder / f'chapter-{chapter:02}-local.ipynb').write_text(json.dumps(book, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
