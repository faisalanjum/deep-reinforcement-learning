"""Check repeated rendering, recording, decoding and the book's HTML helpers."""
import base64
import io
import json
from pathlib import Path
import subprocess
import sys

import drl_book
import gym
from gym import wrappers
import numpy as np
from PIL import Image

from drl_book.chapter12 import RenderUint8, get_gif_html, get_videos_html

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[]
    names=('CartPole-v1','Pendulum-v0','LunarLander-v2','HopperBulletEnv-v0','HalfCheetahBulletEnv-v0')
    for name in names:
        for repeat in range(3):
            folder=ROOT/'validation/media'/f'{sys.version_info.major}.{sys.version_info.minor}'/name/str(repeat)
            folder.mkdir(parents=True,exist_ok=True)
            env=gym.make(name)
            if 'Bullet' in name:env=RenderUint8(env)
            env=wrappers.TimeLimit(env,max_episode_steps=12)
            env=wrappers.Monitor(env,str(folder),force=True,mode="evaluation",video_callable=lambda _:True)
            try:
                for episode in range(2):
                    env.reset()
                    for step in range(12):
                        _,_,done,_=env.step(env.action_space.sample())
                        if done:break
            finally:env.close()
            pairs=[(str(video),str(video.with_suffix('.meta.json'))) for video in folder.glob('*.mp4')]
            assert len(pairs)==2, (name,repeat,pairs)
            for video,_ in pairs:
                subprocess.run(['ffmpeg','-v','error','-i',video,'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',video]))['streams'][0]
                assert int(info['nb_frames'])>1 and info['codec_name']=='h264'
            html=get_videos_html(pairs,name)
            assert html.count('<video')==2
            gif_html=get_gif_html(pairs,name)
            for encoded in __import__('re').findall(r'data:image/gif;base64,([^\"\s]+)',gif_html):
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                    assert image.n_frames>0
                    for frame in range(image.n_frames):image.seek(frame);image.load()
            rows.append({'environment':name,'repeat':repeat,'videos':len(pairs),'status':'passed'})
            print('PASS',name,repeat,flush=True)
    target=ROOT/'validation'/f'media-python{sys.version_info.major}{sys.version_info.minor}.json'
    target.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
