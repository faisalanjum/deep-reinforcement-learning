# Validation report

Updated 2026-10-31T13:56:35+00:00.

The tables distinguish completed notebook execution from ongoing default-setting training. “Bounded” executes every cell with one seed, eight training episodes, two workers and smaller batches; it does not establish convergence. Chapters 2–7 retain their original training settings.

| Environment / profile | Chapter | State | Last cell | PNG outputs | GIF outputs |
|---|---:|---|---:|---:|---:|
| full | 2 | passed | 8 | 0 | 0 |
| full | 3 | passed | 71 | 0 | 0 |
| full | 4 | passed | 34 | 4 | 0 |
| full | 5 | passed | 124 | 55 | 0 |
| full | 6 | passed | 155 | 57 | 0 |
| full | 7 | passed | 214 | 86 | 0 |
| full | 8 | passed | 19 | 1 | 8 |
| full | 9 | passed | 52 | 14 | 16 |
| full | 10 | passed | 51 | 12 | 16 |
| full | 11 | passed | 57 | 6 | 40 |
| full | 12 | running | 28 | 3 | 6 |
| bounded | 8 | passed | 19 | 1 | 8 |
| bounded | 9 | passed | 52 | 14 | 16 |
| bounded | 10 | passed | 51 | 12 | 16 |
| bounded | 11 | passed | 57 | 6 | 40 |
| bounded | 12 | passed | 58 | 9 | 24 |
| bounded-py314 | 2 | passed | 8 | 0 | 0 |
| bounded-py314 | 3 | passed | 71 | 0 | 0 |
| bounded-py314 | 4 | passed | 34 | 4 | 0 |
| bounded-py314 | 5 | passed | 124 | 55 | 0 |
| bounded-py314 | 6 | passed | 155 | 57 | 0 |
| bounded-py314 | 8 | passed | 19 | 1 | 8 |
| bounded-py314 | 9 | passed | 52 | 14 | 16 |
| bounded-py314 | 10 | passed | 51 | 12 | 16 |
| bounded-py314 | 11 | passed | 57 | 6 | 40 |
| bounded-py314 | 12 | passed | 58 | 9 | 24 |
| full-py314 | 7 | passed | 214 | 86 | 0 |

## Media checks

- media-python311: 15 repeated environment checks passed, recording and decoding two videos per check and generating GIFs.
- media-python314: 15 repeated environment checks passed, recording and decoding two videos per check and generating GIFs.

## Chapter 11 learning results

Full notebook execution completed for all 25 runs. Final evaluation rewards by seed (12, 34, 56, 78, 90):

- REINFORCE: 461.16, 500, 500, 500, 494.4.
- VPG: 395.7, 498.67, 500, 500, 492.68.
- A3C: 500, 500, 500, 500, 9.27.
- GAE: 500, 500, 500, 500, 500.
- A2C: 500, 500, 500, 500, 500.

A3C seed 90 initially scored 9.27; an independent repeat using the same original settings scored 500. Both observations are retained. This is consistent with asynchronous training variability; every run is not guaranteed to reach the target reward.

## Other checks

- Both Python environments pass `python -m pip check`.
- Rebuilding Python 3.11 from `requirements-local.txt` passed dependency and import checks.
- Python 3.11 MPS forward/backward and short training tests passed for chapters 8–10 and the four chapter 12 methods. Shared-model A3C/GAE/A2C use CPU.
- Every emitted PNG and GIF in the executed notebook reports was decoded and checked; this verifies the media artifacts, not numerical convergence.
- On 2026-10-31, the user manually confirmed video play/pause/replay and the saved chapter 12 Trial 1/Trial 2 animations in VS Code. Computer-use access remains denied; kernel selection and restart controls have not been manually confirmed.

The live per-chapter files are under `validation/notebook-runs/*/chapter_*/status.json`. Default training can run for hours; a running entry is not a pass.
