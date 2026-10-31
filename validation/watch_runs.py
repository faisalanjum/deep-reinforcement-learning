"""Refresh the validation report and stop this task's runs before disk fills."""
import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import time

import psutil
from status import main as refresh_report

ROOT=Path(__file__).resolve().parents[1]


def runners():
    found=[]
    for process in psutil.process_iter(['pid','cmdline','cwd']):
        try:
            if process.info['cwd']!=str(ROOT):continue
            command=process.info['cmdline'] or []
            if not command or not Path(command[0]).name.lower().startswith('python'):continue
            if any(Path(argument).name in ('run_notebook.py','run_full_remaining.py') for argument in command[1:]):found.append(process)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found


def stop(processes):
    # NotebookClient catches KeyboardInterrupt, shuts down its kernel and saves a failure status.
    for process in processes:
        try:process.send_signal(signal.SIGINT)
        except psutil.NoSuchProcess:pass


def main():
    if '--stop' in sys.argv:
        processes=runners();stop(processes);print('Interrupted validation processes:',[p.pid for p in processes]);return
    while True:
        processes=runners()
        if not processes:break
        free=psutil.disk_usage(str(ROOT)).free
        if free<512*1024**2:
            stop(processes)
            (ROOT/'validation/disk-stop.json').write_text(json.dumps({'reason':'Less than 512 MiB free disk space','pids':[p.pid for p in processes],'time':time.time()},indent=2)+'\n')
        with contextlib.redirect_stdout(io.StringIO()):refresh_report()
        time.sleep(15)
    with contextlib.redirect_stdout(io.StringIO()):refresh_report()


if __name__=='__main__':main()
