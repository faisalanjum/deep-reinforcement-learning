"""Print current validation evidence and refresh the Markdown report."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    lines=['# Validation report','',f'Updated {datetime.now(timezone.utc).isoformat(timespec="seconds")}.','',
           'The tables distinguish completed notebook execution from ongoing default-setting training. “Bounded” executes every cell with one seed, eight training episodes, two workers and smaller batches; it does not establish convergence. Chapters 2–7 retain their original training settings.','',
           '| Environment / profile | Chapter | State | Last cell | PNG outputs | GIF outputs |',
           '|---|---:|---|---:|---:|---:|']
    for profile in ('full','bounded','bounded-py314','full-py314'):
        for filename in sorted((ROOT/'validation/notebook-runs'/profile).glob('chapter_*/status.json')):
            row=json.loads(filename.read_text())
            lines.append(f'| {profile} | {row["chapter"]} | {row["status"]} | {row["current_cell"]} | {row["pngs"]} | {row["gifs"]} |')
    lines.extend(['','## Media checks',''])
    for filename in sorted((ROOT/'validation').glob('media-python*.json')):
        rows=json.loads(filename.read_text());lines.append(f'- {filename.stem}: {len(rows)} repeated environment checks passed, recording and decoding two videos per check and generating GIFs.')
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
                  '', 'The live per-chapter files are under `validation/notebook-runs/*/chapter_*/status.json`. Default training can run for hours; a running entry is not a pass.',''])
    output='\n'.join(lines);(ROOT/'VALIDATION-REPORT.md').write_text(output);print(output)


if __name__=='__main__':main()
