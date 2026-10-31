"""Run every cell from a fresh local kernel and validate emitted image artifacts."""
import argparse
import base64
import io
import json
import os
import re
import shutil
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def bounded(source):
    source=re.sub(r"('max_episodes'\s*:\s*)\d+",r'\g<1>8',source)
    source=re.sub(r"('max_minutes'\s*:\s*)\d+",r'\g<1>2',source)
    source=re.sub(r'(?m)^SEEDS\s*=.*$', 'SEEDS = (12,)',source)
    source=source.replace('batch_size = 1024','batch_size = 8').replace('epochs = 40','epochs = 1')
    source=source.replace('batch_size=256','batch_size=8').replace('batch_size=64','batch_size=8')
    source=source.replace('n_warmup_batches = 10','n_warmup_batches = 1').replace('n_warmup_batches = 5','n_warmup_batches = 1')
    source=source.replace('n_workers = 8','n_workers = 2').replace('max_buffer_episodes = 32','max_buffer_episodes = 4')
    return source.replace('_ = BEEP()', 'pass')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('chapter',type=int)
    parser.add_argument('--profile',choices=('bounded','full'),default='bounded')
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--python-kernel',default='gdrl-python311')
    parser.add_argument('--from-section', choices=('SAC', 'PPO'),
                        help='Continue chapter 12 from this section in a fresh kernel.')
    args=parser.parse_args()
    if args.from_section and args.chapter != 12:
        parser.error('--from-section is only supported for chapter 12')
    suffix='-local'
    source=ROOT/'notebooks'/f'chapter_{args.chapter:02}'/f'chapter-{args.chapter:02}{suffix}.ipynb'
    notebook=nbformat.read(source,as_version=4)
    notebook.metadata.kernelspec = {'name':args.python_kernel,'language':'python','display_name':'GDRL (Python 3.14 - compatibility tests)' if args.python_kernel=='gdrl-python314' else 'GDRL (Python 3.11)'}
    version_tag = '-py314' if args.python_kernel == 'gdrl-python314' else ''
    section_tag = f'-from-{args.from_section.lower()}' if args.from_section else ''
    workspace=ROOT/'validation/notebook-runs'/(args.profile + version_tag + section_tag)
    cwd=workspace/f'chapter_{args.chapter:02}';cwd.mkdir(parents=True,exist_ok=True)
    results=workspace/'results'
    if not results.exists():shutil.copytree(ROOT/'notebooks/results',results)
    # macOS Unix-domain sockets (including torch_shm_manager) require short paths.
    temporary=Path('/tmp/gdrl-v')/(args.profile + version_tag + section_tag)/f'ch{args.chapter:02}'
    temporary.mkdir(parents=True,exist_ok=True)
    kernel_environment=os.environ.copy(); kernel_environment['TMPDIR']=str(temporary)
    output=cwd/source.name
    status={'chapter':args.chapter,'profile':args.profile,'kernel':args.python_kernel,'completed_cells':[],'pngs':0,'gifs':0,'videos':0,'status':'running'}
    section_start = None
    if args.from_section:
        section_start = next(index for index, cell in enumerate(notebook.cells)
                             if cell.cell_type == 'markdown' and cell.source.strip() == f'# {args.from_section}')
        status.update(from_section=args.from_section, skipped_cells=[])
    started=time.monotonic()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
            if args.profile=='bounded' and args.chapter>=8:cell.source=bounded(cell.source)
            if section_start is not None and index < section_start and index > 4 and not cell.source.startswith('from drl_book.'):
                cell.metadata.setdefault('tags', []).append('validation-skip')
                status['skipped_cells'].append(index)
            # Preserve completed training arrays even if a later display cell fails.
            for match in re.finditer(r'(?m)^(\w+)_results = np\.array\(\1_results\)', cell.source):
                method = match.group(1)
                cell.source += f'\nnp.save("training-results-{method}.npy", {method}_results)\n'
    def persist():
        status['seconds']=round(time.monotonic()-started,2)
        pending=cwd/'status.json.tmp'
        pending.write_text(json.dumps(status,indent=2)+'\n')
        pending.replace(cwd/'status.json')
    def begin(cell,cell_index,**kwargs):
        status['current_cell']=cell_index;persist();print('START',cell_index,flush=True)
    def completed(cell,cell_index,**kwargs):
        if cell.cell_type!='code':return
        if any(item.get('output_type') == 'error' for item in cell.get('outputs',[])): return
        for item in cell.get('outputs',[]):
            data=item.get('data',{})
            if 'image/png' in data:
                with Image.open(io.BytesIO(base64.b64decode(data['image/png']))) as image:image.verify()
                status['pngs']+=1
            html=data.get('text/html','')
            for gif in re.findall(r'data:image/gif;base64,([^\"\s]+)',html):
                with Image.open(io.BytesIO(base64.b64decode(gif))) as image:
                    assert image.n_frames>0;image.verify()
                status['gifs']+=1
            status['videos']+=html.count('<video')
        status['completed_cells'].append(cell_index);persist();print('DONE',cell_index,flush=True)
        nbformat.write(notebook,output)
    class ProgressClient(NotebookClient):
        last_progress_write = 0.0
        def process_message(self, message, cell, cell_index):
            result = super().process_message(message, cell, cell_index)
            if message.get('msg_type') == 'stream':
                text = message.get('content',{}).get('text','')
                if 'ep ' in text or 'Training complete' in text:
                    status['training_progress'] = text[-400:]
                    if 'Training complete' in text: status['training_runs_completed'] = status.get('training_runs_completed',0) + 1
                    if time.monotonic() - self.last_progress_write > 1:
                        if shutil.disk_usage(ROOT).free < 512 * 1024**2:
                            raise RuntimeError('Validation stopped because less than 512 MiB of disk space remains')
                        persist(); self.last_progress_write = time.monotonic()
            return result
    try:
        ProgressClient(notebook,timeout=args.timeout,kernel_name=args.python_kernel,skip_cells_with_tag='validation-skip',on_cell_start=begin,on_cell_executed=completed,resources={'metadata':{'path':str(cwd)}}).execute(env=kernel_environment)
        status['status']='passed'
    except BaseException as error:
        status['status']='failed';status['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        persist();nbformat.write(notebook,output);print(json.dumps({k:v for k,v in status.items() if k!='completed_cells'}),flush=True)


if __name__=='__main__':main()
