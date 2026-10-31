"""Continue full default-setting notebook runs, retaining a resumable status file."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    chapters=[int(number) for number in sys.argv[1:]] or [10,11,12]
    status_path=ROOT/'validation/full-queue-status.json'
    status={'chapters':chapters,'completed':[],'status':'running','pid':os.getpid(),'started':time.time()}
    for chapter in chapters:
        prior=ROOT/f'validation/notebook-runs/full/chapter_{chapter:02}/status.json'
        if prior.exists() and json.loads(prior.read_text()).get('status')=='passed':
            status['completed'].append(chapter);continue
        free=shutil.disk_usage(ROOT).free
        if free<1024**3:
            status.update(status='stopped',reason='Less than 1 GiB of free disk space remains');status_path.write_text(json.dumps(status,indent=2)+'\n');return 2
        status['chapter']=chapter;status_path.write_text(json.dumps(status,indent=2)+'\n')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (ROOT/f'validation/notebook{chapter}-full.log').open('w') as log:
            process=subprocess.run([sys.executable,str(ROOT/'validation/run_notebook.py'),str(chapter),'--profile','full','--timeout','108000'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        if process.returncode:
            status.update(status='failed',returncode=process.returncode);status_path.write_text(json.dumps(status,indent=2)+'\n');return process.returncode
        status['completed'].append(chapter)
    status.update(status='passed',finished=time.time());status_path.write_text(json.dumps(status,indent=2)+'\n');return 0


if __name__=='__main__':raise SystemExit(main())
