"""Run every deep RL training implementation with a small real workload."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = [(8,14),(9,21),(9,39),(10,18),(10,46),(11,14),(11,24),(11,34),(11,42),(11,52),(12,16),(12,28),(12,39),(12,52)]


def execute(job, episodes, device, timeout):
    chapter, cell = job
    log = ROOT / 'validation' / f'train-{chapter}-{cell}-{device}.log'
    command = [sys.executable, str(ROOT/'validation/run_training.py'),str(chapter),str(cell),'--episodes',str(episodes),'--device',device]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            code = 'timeout'
    return {'chapter':chapter,'cell':cell,'device':device,'exit':code,'seconds':round(time.monotonic()-start,2),'log':str(log.relative_to(ROOT))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--chapter',type=int)
    args=parser.parse_args()
    jobs=[job for job in JOBS if args.chapter is None or job[0]==args.chapter]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,job,args.episodes,args.device,args.timeout) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    (ROOT/'validation'/(f'training-{args.device}' + (f'-ch{args.chapter}' if args.chapter else '') + '.json')).write_text(json.dumps(results,indent=2)+'\n')
    if any(row['exit']!=0 for row in results):sys.exit(1)


if __name__=='__main__':main()
