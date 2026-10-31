# Validation report

Updated 2026-10-31T20:59:56+00:00.

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
| full | 12 | failed | 31 | 3 | 12 |
| full-from-sac | 12 | passed | 59 | 4 | 12 |
| bounded | 8 | passed | 19 | 1 | 8 |
| bounded | 9 | passed | 52 | 14 | 16 |
| bounded | 10 | passed | 51 | 12 | 16 |
| bounded | 11 | passed | 57 | 6 | 40 |
| bounded | 12 | passed | 58 | 9 | 24 |
| bounded-from-sac | 12 | passed | 59 | 4 | 12 |
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

Chapter 12 completed all five DDPG and five TD3 full training runs, then stopped at a notebook cell containing only `stop` (`NameError`). The user notebook is preserved. The `full-from-sac` profile runs SAC and PPO with their original settings in a fresh kernel; earlier training cells are explicitly skipped. A continuation pass is not a fresh execution of the whole notebook.

Chapter 12 full training runs completed across the original run and continuation: **20/20**.

The SAC/PPO continuation completed successfully, including its training, post-processing and media cells. The original whole-notebook entry remains failed as a historical record of the `stop` cell.

## Media checks

- media-python311: 15 repeated environment checks passed, recording and decoding two videos per check and generating GIFs.
- media-python314: 15 repeated environment checks passed, recording and decoding two videos per check and generating GIFs.

## Completed training score review

- Chapters 8–10: 25/25 recorded runs finished at or above their configured final evaluation reward targets.
- Chapter 12 completed methods: 15/20 recorded runs finished at or above their configured final evaluation reward targets.
  - DDPG final mean rewards: -130.96, -148.54, -135.71, -149.41, -144.1.
  - TD3 final mean rewards: 2057.51, 1792.77, 2091.47, 1936.23, 1930.52.
  - SAC final mean rewards: 2134.26, 2324.46, 2100.14, 2137.2, 2142.26.
  - PPO final mean rewards: 122.57, 150.04, 172.67, 160.76, 195.34.

PPO completed all five full training runs without execution errors, but its final mean rewards were below 200. Four runs crossed the book's rolling stopping threshold; final evaluation over 100 separate episodes scored lower. These results establish execution coverage, not reliable achievement of the PPO reward target.

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

## Coverage limitations

- The working chapter 12 notebook retains its standalone `stop` cell; remove or comment it before Run All. TD3 post-processing after that cell was previously verified with reduced settings, but was not reached in the original full run.
- Full-length later-chapter training was verified on Python 3.11. Python 3.14 chapters 8–12 and MPS were checked with reduced training runs.
- Learning scores vary. PPO final rewards remain below its target, and some chapter 11 seeds also fell below their targets.

The per-chapter files are under `validation/notebook-runs/*/chapter_*/status.json`. A passed execution is not a guarantee of convergence.
