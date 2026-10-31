"""Print current validation evidence and refresh the Markdown report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def read_status(filename):
    # Existing running jobs truncate and rewrite their JSON status files.
    # Retry a concurrent read rather than stopping the report/disk monitor.
    for attempt in range(5):
        try:
            return json.loads(filename.read_text())
        except (json.JSONDecodeError, OSError):
            if attempt < 4:
                time.sleep(0.02)
    return None


def main():
    lines=['# Validation report','',f'Updated {datetime.now(timezone.utc).isoformat(timespec="seconds")}.','',
           'The tables distinguish completed notebook execution from ongoing default-setting training. “Bounded” executes every cell with one seed, eight training episodes, two workers and smaller batches; it does not establish convergence. Chapters 2–7 retain their original training settings.','',
           '| Environment / profile | Chapter | State | Last cell | PNG outputs | GIF outputs |',
           '|---|---:|---|---:|---:|---:|']
    for profile in ('full','full-from-sac','full-from-ppo','bounded','bounded-from-sac','bounded-py314','full-py314'):
        for filename in sorted((ROOT/'validation/notebook-runs'/profile).glob('chapter_*/status.json')):
            row=read_status(filename)
            if row is None:
                chapter=filename.parent.name.removeprefix('chapter_')
                lines.append(f'| {profile} | {chapter} | status temporarily unavailable | — | — | — |')
                continue
            lines.append(f'| {profile} | {row["chapter"]} | {row["status"]} | {row["current_cell"]} | {row["pngs"]} | {row["gifs"]} |')
    continuation = ROOT/'validation/chapter12-continuation.json'
    if continuation.exists():
        lines.extend(['', 'Chapter 12 completed all five DDPG and five TD3 full training runs, then stopped at a notebook cell containing only `stop` (`NameError`). The user notebook is preserved. The `full-from-sac` profile runs SAC and PPO with their original settings in a fresh kernel; earlier training cells are explicitly skipped. A continuation pass is not a fresh execution of the whole notebook.'])
        continued = ROOT/'validation/notebook-runs/full-from-sac/chapter_12/status.json'
        if continued.exists():
            progress = read_status(continued)
            if progress:
                total = 10 + progress.get('training_runs_completed', 0)
                lines.extend(['', f'Chapter 12 full training runs completed across the original run and continuation: **{total}/20**.'])
                if progress.get('status') == 'passed':
                    lines.extend(['', 'The SAC/PPO continuation completed successfully, including its training, post-processing and media cells. The original whole-notebook entry remains failed as a historical record of the `stop` cell.'])
    lines.extend(['','## Media checks',''])
    for filename in sorted((ROOT/'validation').glob('media-python*.json')):
        rows=json.loads(filename.read_text());lines.append(f'- {filename.stem}: {len(rows)} repeated environment checks passed, recording and decoding two videos per check and generating GIFs.')
    score_reviews = (
        ('Chapters 8–10', ROOT/'validation/training-results-chapters8-10-full.json'),
        ('Chapter 12 completed methods', ROOT/'validation/training-results-chapter12-full.json'),
    )
    if any(filename.exists() for _, filename in score_reviews):
        lines.extend(['', '## Completed training score review', ''])
        for label, filename in score_reviews:
            if filename.exists():
                rows=json.loads(filename.read_text())
                reached=sum(row.get('final_reward_at_least_goal') is True for row in rows)
                lines.append(f'- {label}: {reached}/{len(rows)} recorded runs finished at or above their configured final evaluation reward targets.')
                if label.startswith('Chapter 12'):
                    for method in dict.fromkeys(row['method'] for row in rows):
                        scores=', '.join(f"{row['mean_reward']:g}" for row in rows if row['method']==method)
                        lines.append(f'  - {method} final mean rewards: {scores}.')
                    if any(row['method']=='PPO' for row in rows):
                        lines.extend(['', 'PPO completed all five full training runs without execution errors, but its final mean rewards were below 200. Four runs crossed the book\'s rolling stopping threshold; final evaluation over 100 separate episodes scored lower. These results establish execution coverage, not reliable achievement of the PPO reward target.'])
    chapter11_results = ROOT/'validation/training-results-chapter11-full.json'
    if chapter11_results.exists():
        rows = json.loads(chapter11_results.read_text())
        lines.extend(['', '## Chapter 11 learning results', '',
                      'Full notebook execution completed for all 25 runs. Final evaluation rewards by seed (12, 34, 56, 78, 90):', ''])
        for method in dict.fromkeys(row['method'] for row in rows):
            scores = ', '.join(f"{row['mean_reward']:g}" for row in rows if row['method'] == method)
            lines.append(f'- {method}: {scores}.')
        repeat = ROOT/'validation/a3c-seed90-repeat-result.json'
        if repeat.exists():
            score = json.loads(repeat.read_text())['final_eval_score']
            lines.extend(['', f'A3C seed 90 initially scored 9.27; an independent repeat using the same original settings scored {score:g}. Both observations are retained. This is consistent with asynchronous training variability; every run is not guaranteed to reach the target reward.'])
    lines.extend(['','## Other checks','',
                  '- Both Python environments pass `python -m pip check`.',
                  '- Rebuilding Python 3.11 from `requirements-local.txt` passed dependency and import checks.',
                  '- Python 3.11 MPS forward/backward and short training tests passed for chapters 8–10 and the four chapter 12 methods. Shared-model A3C/GAE/A2C use CPU.',
                  '- Every emitted PNG and GIF in the executed notebook reports was decoded and checked; this verifies the media artifacts, not numerical convergence.',
                  '- On 2026-10-31, the user manually confirmed video play/pause/replay and the saved chapter 12 Trial 1/Trial 2 animations in VS Code. Computer-use access remains denied; kernel selection and restart controls have not been manually confirmed.',
                  '', '## Coverage limitations', '',
                  '- The working chapter 12 notebook retains its standalone `stop` cell; remove or comment it before Run All. TD3 post-processing after that cell was previously verified with reduced settings, but was not reached in the original full run.',
                  '- Full-length later-chapter training was verified on Python 3.11. Python 3.14 chapters 8–12 and MPS were checked with reduced training runs.',
                  '- Learning scores vary. PPO final rewards remain below its target, and some chapter 11 seeds also fell below their targets.',
                  '', 'The per-chapter files are under `validation/notebook-runs/*/chapter_*/status.json`. A passed execution is not a guarantee of convergence.',''])
    output='\n'.join(lines);(ROOT/'VALIDATION-REPORT.md').write_text(output);print(output)


if __name__=='__main__':main()
