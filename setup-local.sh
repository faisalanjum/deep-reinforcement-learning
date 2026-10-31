#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "${1:-3.11}" in
  3.11) drl_env=.venv-gdrl; drl_python=python3.11; drl_requirements=requirements-local.txt; drl_kernel=gdrl-python311; drl_label='GDRL (Python 3.11)' ;;
  3.14) drl_env=.venv-gdrl-py314; drl_python=python3.14; drl_requirements=requirements-python314.txt; drl_kernel=gdrl-python314; drl_label='GDRL (Python 3.14 - compatibility tests)' ;;
  *) printf '%s\n' 'Use: ./setup-local.sh 3.11 or ./setup-local.sh 3.14' >&2; exit 2 ;;
esac
if [ ! -d "$drl_env" ]; then uv venv --python "$drl_python" "$drl_env"; fi
uv pip install --python "$drl_env/bin/python" 'swig==4.5.1' 'setuptools==80.9.0'
PATH="$PWD/$drl_env/bin:$PATH" CFLAGS='-include stdio.h' uv pip sync --python "$drl_env/bin/python" "$drl_requirements"
"$drl_env/bin/python" -m pip check
"$drl_env/bin/python" -m ipykernel install --user --name "$drl_kernel" --display-name "$drl_label"
