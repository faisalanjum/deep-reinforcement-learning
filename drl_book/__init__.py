"""Local compatibility support for Grokking Deep Reinforcement Learning."""
import os
from pathlib import Path
import sys

# GUI-launched VS Code kernels may start without Homebrew's command directory.
# The book invokes FFmpeg and ImageMagick when it builds notebook animations.
_candidates = [Path(sys.prefix) / 'bin', Path('/opt/homebrew/bin'), Path('/usr/local/bin')]
_existing = os.environ.get('PATH', '').split(os.pathsep)
for _directory in reversed(_candidates):
    if _directory.is_dir() and str(_directory) not in _existing:
        _existing.insert(0, str(_directory))
os.environ['PATH'] = os.pathsep.join(_existing)
