# Local GDRL setup

Open this folder in VS Code and select **GDRL (Python 3.11)**. Its environment is `.venv-gdrl`, and its internal kernel name is `gdrl-python311`. Open the `chapter-XX-local.ipynb` notebooks; the author's original notebooks remain available alongside them. The earlier duplicate `.venv` was removed.

The second environment is `.venv-gdrl-py314`, shown as **GDRL (Python 3.14 - compatibility tests)**. It uses the installed Python 3.14 and newer NumPy, SciPy and Matplotlib packages. The two Python environments are intentionally separate; each kernel shares packages with its own environment. See `VALIDATION-REPORT.md` for the exact scope of testing.

## Packages and kernels

Run `%pip install package-name` in a notebook to install into its selected environment. In a terminal, activate the same environment with `source .venv-gdrl/bin/activate` before running `python -m pip install package-name`. Both use the same installed packages. Restart the notebook kernel after package changes when needed. Arbitrary upgrades still require retesting.

To recreate the tested environments on this Mac, run `./setup-local.sh 3.11` or `./setup-local.sh 3.14`. The matching requirements files pin packages, author environment commits and local compatibility wheels. The binary Box2D/PyBullet wheels target this Mac's ARM platform and Python versions. FFmpeg and ImageMagick are installed at `/opt/homebrew/bin`; the local package adds that location when a GUI-launched kernel has a minimal PATH.

## Local compatibility changes

- The local Gym and pyglet wheels fix the Mac renderer's pixel-data API, viewer closure, and stale Cocoa object cache. Their metadata declares the tested dependencies; no hidden `sitecustomize` patch is needed, and `pip check` passes.
- Chapters 11 and 12 import their implementations from `drl_book/` so macOS can spawn importable workers. Notebook-defined factories are serialized with cloudpickle; shared-model algorithms use CPU shared memory. Worker errors propagate to the parent and workers are cleaned up. Forcing macOS `fork` is not used.
- Shared optimizer state and local gradient clearing support current PyTorch. A2C uses reshape for noncontiguous tensors. SAC backpropagates its policy before updating critic parameters needed by that graph.
- PPO uses supported float precision and valid integer buffer initialization. Removed NumPy aliases, mixed replay sample arrays, a softmax temperature scalar conversion, and a typo in a chapter 10 strategy were corrected for newer NumPy/Python.
- Bullet notebook recordings use direct rendering on macOS, avoiding a separate GUI physics connection. The book's uint8 frame wrapper remains in use.
- The NVIDIA-only diagnostic is replaced with a local CUDA/MPS availability check.

`compat/generate_local.py` regenerates the local notebook editions and exported chapter modules from the original notebooks. The patched Gym/pyglet source and wheels are in `compat/` and `wheels/` with their original licenses.

## CPU and MPS

CPU is the default. To use MPS for the supported single-model algorithms, run `import os; os.environ['DRL_BOOK_DEVICE'] = 'mps'` before creating the model. This is supported in chapters 8–10 and 12; chapter 11's shared-model methods stay on CPU. MPS passes the tested training paths, but the small networks in this book can run faster on CPU because per-step device overhead is significant.

## Verification

`VALIDATION-REPORT.md` records completed and ongoing tests. Refresh it with `.venv-gdrl/bin/python validation/status.py`.

To run a whole chapter with reduced training settings: `.venv-gdrl/bin/python validation/run_notebook.py 12`. To run the original training settings: add `--profile full --timeout 108000`. Validation runs use separate result directories and do not overwrite the author's saved results. The book's default later-chapter training settings can take many hours; reduced runs demonstrate functionality and do not prove convergence.

`validation/check_media.py` tests repeated rendering, MP4 recording/decoding and GIF generation for CartPole, Pendulum, LunarLander, Hopper and HalfCheetah. Executed validation notebooks are saved under `validation/notebook-runs/`.

On 2026-10-31, the user manually confirmed video play/pause/replay and the saved chapter 12 Trial 1/Trial 2 animations in VS Code. Computer-use access remains denied. Kernel execution is tested automatically; kernel selection and restart controls in VS Code have not been manually confirmed.

## Long validation currently running

The default-setting notebook runs for chapters 2–11 have completed on Python 3.11. Chapter 12 completed all five DDPG and five TD3 full training runs, then stopped at a standalone `stop` cell in the working notebook. SAC and PPO continue in a fresh kernel with their original settings. Chapter 7 also completed with its original settings on Python 3.14. A completed notebook execution does not guarantee that every random seed reaches its learning target; the report retains observed reward results, including a low A3C seed and its successful independent repeat.

The working chapter 12 notebook retains the standalone `stop` cell. Comment out or remove that cell before using Run All across the whole notebook. To validate just the remaining SAC/PPO sections without repeating DDPG/TD3, use `.venv-gdrl/bin/python validation/run_notebook.py 12 --profile full --from-section SAC --timeout 108000`. The continuation skips earlier training cells explicitly and does not modify the working notebook. Training arrays and executed outputs are saved after each training cell completes.

To stop this task's validation processes cleanly, run `.venv-gdrl/bin/python validation/watch_runs.py --stop`. The monitor also interrupts them if disk space falls below 512 MiB. To continue a stopped default-setting queue, run `.venv-gdrl/bin/python validation/run_full_remaining.py 7 8 9 10 11 12`; completed chapters are skipped.
