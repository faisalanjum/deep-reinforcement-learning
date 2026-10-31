# Derived from the author's notebook; local changes are described in LOCAL-SETUP.md.

import warnings ; warnings.filterwarnings('ignore')
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import threading

import numpy as np
from IPython.display import display
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle, count
from textwrap import wrap

import matplotlib
import subprocess
import os.path
import tempfile
import random
import base64
import pprint
import glob
import time
import json
import sys
import gym
import io
import os
import gc

from gym import wrappers
from subprocess import check_output
from IPython.display import HTML

LEAVE_PRINT_EVERY_N_SECS = 30
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)



from .devices import device_for


from .processes import SerializableCallable, SpawnEnvironments, ColumnSpawnEnvironments, join_workers
mp = torch.multiprocessing.get_context("spawn")


# Original notebook cell 5
def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None, 
                    inner_wrappers=None, outer_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True, 
            mode=monitor_mode, 
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env
    return make_env_fn, kargs


# Original notebook cell 6
def get_videos_html(env_videos, title, max_n_videos=5):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        video = io.open(video_path, 'r+b').read()
        encoded = base64.b64encode(video)

        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <video width="960" height="540" controls>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm


# Original notebook cell 7
def get_gif_html(env_videos, title, subtitle_eps=None, max_n_videos=4):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        basename = os.path.splitext(video_path)[0]
        gif_path = basename + '.gif'
        if not os.path.exists(gif_path):
            ps = subprocess.Popen(
                ('ffmpeg', 
                 '-i', video_path, 
                 '-r', '7',
                 '-f', 'image2pipe', 
                 '-vcodec', 'ppm',
                 '-crf', '20',
                 '-vf', 'scale=512:-1',
                 '-'), 
                stdout=subprocess.PIPE)
            output = subprocess.check_output(
                ('convert',
                 '-coalesce',
                 '-delay', '7',
                 '-loop', '0',
                 '-fuzz', '2%',
                 '+dither',
                 '-deconstruct',
                 '-layers', 'Optimize',
                 '-', gif_path), 
                stdin=ps.stdout)
            ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)
            
        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        prefix = 'Trial ' if subtitle_eps is None else 'Episode '
        sufix = str(meta['episode_id'] if subtitle_eps is None \
                    else subtitle_eps[meta['episode_id']])
        strm += html_tag.format(prefix + sufix, encoded.decode('ascii'))
    return strm


# Original notebook cell 9
class DiscountedCartPole(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    def step(self, a):
        o, r, d, _ = self.env.step(a)
        (x, x_dot, theta, theta_dot) = o
        pole_fell =  x < -self.env.unwrapped.x_threshold \
                    or x > self.env.unwrapped.x_threshold \
                    or theta < -self.env.unwrapped.theta_threshold_radians \
                    or theta > self.env.unwrapped.theta_threshold_radians
        r = -1 if pole_fell else 0
        return o, r, d, _


# Original notebook cell 10
class MCCartPole(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    def step(self, a):
        o, r, d, _ = self.env.step(a)
        (x, x_dot, theta, theta_dot) = o
        pole_fell =  x < -self.env.unwrapped.x_threshold \
                    or x > self.env.unwrapped.x_threshold \
                    or theta < -self.env.unwrapped.theta_threshold_radians \
                    or theta > self.env.unwrapped.theta_threshold_radians
        if d:
            if pole_fell:
                r = 0 # done, in failure
            else:
                r = self.env._max_episode_steps # done, but successfully
        return o, r, d, _


# Original notebook cell 12
class FCDAP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCDAP, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
        
    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())


# Original notebook cell 13
class REINFORCE():
    def __init__(self, policy_model_fn, policy_optimizer_fn, policy_optimizer_lr):
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])

        discounts = torch.FloatTensor(discounts).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        self.logpas = torch.cat(self.logpas)

        policy_loss = -(discounts * returns * self.logpas).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def interaction_step(self, state, env):
        action, is_exploratory, logpa, _ = self.policy_model.full_pass(state)
        new_state, reward, is_terminal, _ = env.step(action)

        self.logpas.append(logpa)
        self.rewards.append(reward)
        
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(is_exploratory)

        return new_state, is_terminal

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.episode_exploration = []
        self.evaluation_scores = []
        
        self.policy_model = self.policy_model_fn(nS, nA)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, 
                                                         self.policy_optimizer_lr)
                    
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            # collect rollout
            self.logpas, self.rewards = [], []
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                if is_terminal:
                    gc.collect()
                    break

            self.optimize_model()
            
            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.save_checkpoint(episode-1, self.policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed
            
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
                
        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else: 
                    a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


# Original notebook cell 22
class FCV(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)


# Original notebook cell 23
class VPG():
    def __init__(self, 
                 policy_model_fn, 
                 policy_model_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_model_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 entropy_loss_weight):
        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.entropy_loss_weight = entropy_loss_weight

    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        self.logpas = torch.cat(self.logpas)
        self.entropies = torch.cat(self.entropies) 
        self.values = torch.cat(self.values)

        value_error = returns - self.values
        policy_loss = -(discounts * value_error.detach() * self.logpas).mean()
        entropy_loss = -self.entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        self.policy_optimizer.step()

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        self.value_optimizer.step()
        
    def interaction_step(self, state, env):
        action, is_exploratory, logpa, entropy = self.policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']

        self.logpas.append(logpa)
        self.entropies.append(entropy)
        self.rewards.append(reward)
        self.values.append(self.value_model(state))

        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(is_exploratory)
        return new_state, is_terminal, is_truncated

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.episode_exploration = []
        self.evaluation_scores = []

        self.policy_model = self.policy_model_fn(nS, nA)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, 
                                                         self.policy_optimizer_lr)
        
        self.value_model = self.value_model_fn(nS)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, 
                                                       self.value_optimizer_lr)
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            # collect rollout
            self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
            for step in count():
                state, is_terminal, is_truncated = self.interaction_step(state, env)
                if is_terminal:
                    gc.collect()
                    break

            is_failure = is_terminal and not is_truncated
            next_value = 0 if is_failure else self.value_model(state).detach().item()
            self.rewards.append(next_value)
            self.optimize_model()

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.save_checkpoint(episode-1, self.policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed
            
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else: 
                    a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


# Original notebook cell 31
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                state['step'] = torch.zeros(()).share_memory_()
                state['exp_avg'] = torch.zeros_like(parameter).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(parameter).share_memory_()
                if group['amsgrad']:
                    state['max_exp_avg_sq'] = torch.zeros_like(parameter).share_memory_()



# Original notebook cell 32
class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                state['step'] = torch.zeros(()).share_memory_()
                state['square_avg'] = torch.zeros_like(parameter).share_memory_()
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(parameter).share_memory_()
                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(parameter).share_memory_()



# Original notebook cell 33
class A3C():
    def __init__(self, 
                 policy_model_fn, 
                 policy_model_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_model_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 entropy_loss_weight, 
                 max_n_steps, 
                 n_workers):
        self.policy_model_fn = SerializableCallable(policy_model_fn)
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = SerializableCallable(policy_optimizer_fn)
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.value_model_fn = SerializableCallable(value_model_fn)
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = SerializableCallable(value_optimizer_fn)
        self.value_optimizer_lr = value_optimizer_lr
        
        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers

    def optimize_model(self, logpas, entropies, rewards, values, 
                       local_policy_model, local_value_model):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        value_error = returns - values
        policy_loss = -(discounts * value_error.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        local_policy_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(), 
                                       self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        local_value_model.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(), 
                                       self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    @staticmethod
    def interaction_step(state, env, local_policy_model, local_value_model,
                         logpas, entropies, rewards, values):
        action, is_exploratory, logpa, entropy = local_policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']

        logpas.append(logpa)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_value_model(state))

        return new_state, reward, is_terminal, is_truncated, is_exploratory

    def work(self, rank):
        last_debug_time = float('-inf')
        with self.get_out_lock:
            self.stats['n_active_workers'].add_(1)
        
        local_seed = self.seed + rank
        env = self.make_env_fn(**self.make_env_kargs, seed=local_seed)
        torch.manual_seed(local_seed) ; np.random.seed(local_seed) ; random.seed(local_seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        with self.get_out_lock:
            if self.stats['episode'].item() >= self.max_episodes:
                self.stats['n_active_workers'].sub_(1)
                env.close()
                return
            global_episode_idx = self.stats['episode'].add_(1).item() - 1
        while not self.get_out_signal:            
            episode_start = time.time()
            state, is_terminal = env.reset(), False
            
            # collect n_steps rollout
            n_steps_start, total_episode_rewards = 0, 0
            total_episode_steps, total_episode_exploration = 0, 0
            logpas, entropies, rewards, values = [], [], [], []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory = self.interaction_step(
                    state, env, local_policy_model, local_value_model, 
                    logpas, entropies, rewards, values)

                total_episode_steps += 1
                total_episode_rewards += reward
                total_episode_exploration += int(is_exploratory)
                
                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else local_value_model(state).detach().item()
                    rewards.append(next_value)

                    self.optimize_model(logpas, entropies, rewards, values, 
                                        local_policy_model, local_value_model)
                    logpas, entropies, rewards, values = [], [], [], []
                    n_steps_start = step
                
                if is_terminal:
                    gc.collect()
                    break

            # save global stats
            episode_elapsed = time.time() - episode_start
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.save_checkpoint(global_episode_idx, local_policy_model)
            
            self.stats['episode_elapsed'][global_episode_idx].add_(episode_elapsed)
            self.stats['episode_timestep'][global_episode_idx].add_(total_episode_steps)
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            self.stats['episode_exploration'][global_episode_idx].add_(total_episode_exploration/total_episode_steps)
            self.stats['evaluation_scores'][global_episode_idx].add_(evaluation_score)

            mean_10_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-10:].mean().item()
            mean_100_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-100:].mean().item()
            mean_100_eval_score = self.stats[
                'evaluation_scores'][:global_episode_idx+1][-100:].mean().item()
            mean_100_exp_rat = self.stats[
                'episode_exploration'][:global_episode_idx+1][-100:].mean().item()
            std_10_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-10:].std().item()
            std_100_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-100:].std().item()
            std_100_eval_score = self.stats[
                'evaluation_scores'][:global_episode_idx+1][-100:].std().item()
            std_100_exp_rat = self.stats[
                'episode_exploration'][:global_episode_idx+1][-100:].std().item()
            if std_10_reward != std_10_reward: std_10_reward = 0            
            if std_100_reward != std_100_reward: std_100_reward = 0
            if std_100_eval_score != std_100_eval_score: std_100_eval_score = 0
            if std_100_exp_rat != std_100_exp_rat: std_100_exp_rat = 0
            global_n_steps = self.stats[
                'episode_timestep'][:global_episode_idx+1].sum().item()
            global_training_elapsed = self.stats[
                'episode_elapsed'][:global_episode_idx+1].sum().item()
            wallclock_elapsed = time.time() - self.training_start
            
            self.stats['result'][global_episode_idx][0].add_(global_n_steps)
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            self.stats['result'][global_episode_idx][2].add_(mean_100_eval_score)
            self.stats['result'][global_episode_idx][3].add_(global_training_elapsed)
            self.stats['result'][global_episode_idx][4].add_(wallclock_elapsed)

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, global_episode_idx, global_n_steps, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            
            if rank == 0:
                print(debug_message, end='\r', flush=True)
                if time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()

            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_goal_mean_reward.add_(
                    mean_100_eval_score >= self.goal_mean_100_reward)
                self.reached_max_minutes.add_(
                    time.time() - self.training_start >= self.max_minutes * 60)
                self.reached_max_episodes.add_(
                    potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes or \
                   self.reached_max_minutes or \
                   self.reached_goal_mean_reward:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            time.sleep(0.01)

        if rank == 0:
            print(ERASE_LINE + debug_message)
            if self.reached_max_minutes: print(u'--> reached_max_minutes \u2715')
            if self.reached_max_episodes: print(u'--> reached_max_episodes \u2715')
            if self.reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')

        env.close() ; del env
        with self.get_out_lock:
            self.stats['n_active_workers'].sub_(1)

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = SerializableCallable(make_env_fn)
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        self.max_minutes = max_minutes
        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        nS, nA = env.observation_space.shape[0], env.action_space.n
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)

        self.stats = {}
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['evaluation_scores'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_timestep'] = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.stats['episode_exploration'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()

        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model, 
                                                                self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model, 
                                                              self.value_optimizer_lr)

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.reached_goal_mean_reward  = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.training_start = time.time()
        workers = [mp.Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        [w.start() for w in workers]
        join_workers(workers, max_minutes * 60 + 120)
        wallclock_time = time.time() - self.training_start

        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)
        env.close() ; del env

        final_episode = self.stats['episode'].item()
        training_time = self.stats['episode_elapsed'][:final_episode+1].sum().item()

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))

        self.stats['result'] = self.stats['result'].numpy()
        self.stats['result'][final_episode:, ...] = np.nan
        self.get_cleaned_checkpoints()
        return self.stats['result'], final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else: 
                    a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.shared_policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.shared_policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


# Original notebook cell 41
class GAE():
    def __init__(self,
                 policy_model_fn,
                 policy_model_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn,
                 value_model_max_grad_norm,
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 entropy_loss_weight, 
                 max_n_steps,
                 n_workers,
                 tau):
        self.policy_model_fn = SerializableCallable(policy_model_fn)
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = SerializableCallable(policy_optimizer_fn)
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = SerializableCallable(value_model_fn)
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = SerializableCallable(value_optimizer_fn)
        self.value_optimizer_lr = value_optimizer_lr

        self.entropy_loss_weight = entropy_loss_weight

        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

    def optimize_model(self, logpas, entropies, rewards, values, 
                       local_policy_model, local_value_model):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])

        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        np_values = values.reshape(-1).data.numpy()
        tau_discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]  
        gaes = np.array([np.sum(tau_discounts[:T-1-t] * advs[t:]) for t in range(T-1)])

        values = values[:-1,...]
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)
        gaes = torch.FloatTensor(gaes).unsqueeze(1)

        policy_loss = -(discounts * gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        local_policy_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(), 
                                       self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_error = returns - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        local_value_model.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(), 
                                       self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    @staticmethod
    def interaction_step(state, env, local_policy_model, local_value_model,
                         logpas, entropies, rewards, values):
        action, is_exploratory, logpa, entropy = local_policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']

        logpas.append(logpa)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_value_model(state))

        return new_state, reward, is_terminal, is_truncated, is_exploratory

    def work(self, rank):
        last_debug_time = float('-inf')
        with self.get_out_lock:
            self.stats['n_active_workers'].add_(1)
        
        local_seed = self.seed + rank
        env = self.make_env_fn(**self.make_env_kargs, seed=local_seed)
        torch.manual_seed(local_seed) ; np.random.seed(local_seed) ; random.seed(local_seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        with self.get_out_lock:
            if self.stats['episode'].item() >= self.max_episodes:
                self.stats['n_active_workers'].sub_(1)
                env.close()
                return
            global_episode_idx = self.stats['episode'].add_(1).item() - 1
        while not self.get_out_signal:            
            episode_start = time.time()
            state, is_terminal = env.reset(), False
            
            # collect n_steps rollout
            n_steps_start, total_episode_rewards = 0, 0
            total_episode_steps, total_episode_exploration = 0, 0
            logpas, entropies, rewards, values = [], [], [], []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory = self.interaction_step(
                    state, env, local_policy_model, local_value_model, 
                    logpas, entropies, rewards, values)

                total_episode_steps += 1
                total_episode_rewards += reward
                total_episode_exploration += int(is_exploratory)
                
                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else local_value_model(state).detach().item()
                    rewards.append(next_value)
                    values.append(torch.FloatTensor([[next_value,],]))

                    self.optimize_model(logpas, entropies, rewards, values, 
                                        local_policy_model, local_value_model)
                    logpas, entropies, rewards, values = [], [], [], []
                    n_steps_start = step
                
                if is_terminal:
                    gc.collect()
                    break

            # save global stats
            episode_elapsed = time.time() - episode_start
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.save_checkpoint(global_episode_idx, local_policy_model)

            self.stats['episode_elapsed'][global_episode_idx].add_(episode_elapsed)
            self.stats['episode_timestep'][global_episode_idx].add_(total_episode_steps)
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            self.stats['episode_exploration'][global_episode_idx].add_(total_episode_exploration/total_episode_steps)
            self.stats['evaluation_scores'][global_episode_idx].add_(evaluation_score)

            mean_10_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-10:].mean().item()
            mean_100_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-100:].mean().item()
            mean_100_eval_score = self.stats[
                'evaluation_scores'][:global_episode_idx+1][-100:].mean().item()
            mean_100_exp_rat = self.stats[
                'episode_exploration'][:global_episode_idx+1][-100:].mean().item()
            std_10_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-10:].std().item()
            std_100_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-100:].std().item()
            std_100_eval_score = self.stats[
                'evaluation_scores'][:global_episode_idx+1][-100:].std().item()
            std_100_exp_rat = self.stats[
                'episode_exploration'][:global_episode_idx+1][-100:].std().item()
            if std_10_reward != std_10_reward: std_10_reward = 0            
            if std_100_reward != std_100_reward: std_100_reward = 0
            if std_100_eval_score != std_100_eval_score: std_100_eval_score = 0
            if std_100_exp_rat != std_100_exp_rat: std_100_exp_rat = 0
            global_n_steps = self.stats[
                'episode_timestep'][:global_episode_idx+1].sum().item()
            global_training_elapsed = self.stats[
                'episode_elapsed'][:global_episode_idx+1].sum().item()
            wallclock_elapsed = time.time() - self.training_start
            
            self.stats['result'][global_episode_idx][0].add_(global_n_steps)
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            self.stats['result'][global_episode_idx][2].add_(mean_100_eval_score)
            self.stats['result'][global_episode_idx][3].add_(global_training_elapsed)
            self.stats['result'][global_episode_idx][4].add_(wallclock_elapsed)

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, global_episode_idx, global_n_steps, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)

            if rank == 0:
                print(debug_message, end='\r', flush=True)
                if time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()

            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_goal_mean_reward.add_(
                    mean_100_eval_score >= self.goal_mean_100_reward)
                self.reached_max_minutes.add_(
                    time.time() - self.training_start >= self.max_minutes * 60)
                self.reached_max_episodes.add_(
                    potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes or \
                   self.reached_max_minutes or \
                   self.reached_goal_mean_reward:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            time.sleep(0.01)

        if rank == 0:
            print(ERASE_LINE + debug_message)
            if self.reached_max_minutes: print(u'--> reached_max_minutes \u2715')
            if self.reached_max_episodes: print(u'--> reached_max_episodes \u2715')
            if self.reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')

        env.close() ; del env
        with self.get_out_lock:
            self.stats['n_active_workers'].sub_(1)


    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = SerializableCallable(make_env_fn)
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        self.max_minutes = max_minutes
        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        nS, nA = env.observation_space.shape[0], env.action_space.n
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)

        self.stats = {}
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['evaluation_scores'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_timestep'] = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.stats['episode_exploration'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()

        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model, 
                                                                self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model, 
                                                              self.value_optimizer_lr)

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.reached_goal_mean_reward  = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.training_start = time.time()
        workers = [mp.Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        [w.start() for w in workers]
        join_workers(workers, max_minutes * 60 + 120)
        wallclock_time = time.time() - self.training_start

        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)
        env.close() ; del env

        final_episode = self.stats['episode'].item()
        training_time = self.stats['episode_elapsed'][:final_episode+1].sum().item()

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))

        self.stats['result'] = self.stats['result'].numpy()
        self.stats['result'][final_episode:, ...] = np.nan
        self.get_cleaned_checkpoints()
        return self.stats['result'], final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else: 
                    a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.shared_policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.shared_policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


# Original notebook cell 49
class FCAC(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCAC, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.value_output_layer = nn.Linear(hidden_dims[-1], 1)
        self.policy_output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.policy_output_layer(x), self.value_output_layer(x)
    
    def full_pass(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        is_exploratory = action != np.argmax(logits.detach().numpy(), axis=int(len(state)!=1))
        return action, is_exploratory, logpa, entropy, value

    def select_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action
    
    def select_greedy_action(self, state):
        logits, _ = self.forward(state)
        return np.argmax(logits.detach().numpy())
    
    def evaluate_state(self, state):
        _, value = self.forward(state)
        return value


# Original notebook cell 50
MultiprocessEnv = ColumnSpawnEnvironments



# Original notebook cell 51
class A2C():
    def __init__(self, 
                 ac_model_fn, 
                 ac_model_max_grad_norm, 
                 ac_optimizer_fn, 
                 ac_optimizer_lr,
                 policy_loss_weight, 
                 value_loss_weight,
                 entropy_loss_weight,
                 max_n_steps,
                 n_workers,
                 tau):
        assert n_workers > 1
        self.ac_model_fn = ac_model_fn
        self.ac_model_max_grad_norm = ac_model_max_grad_norm
        self.ac_optimizer_fn = ac_optimizer_fn
        self.ac_optimizer_lr = ac_optimizer_lr

        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

    def optimize_model(self):
        logpas = torch.stack(self.logpas).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        values = torch.stack(self.values).squeeze()

        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()
        returns = np.array([[np.sum(discounts[:T-t] * rewards[t:, w]) for t in range(T)] 
                             for w in range(self.n_workers)])

        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([[np.sum(tau_discounts[:T-1-t] * advs[t:, w]) for t in range(T-1)] 
                             for w in range(self.n_workers)])
        discounted_gaes = discounts[:-1] * gaes
        
        values = values[:-1,...].reshape(-1).unsqueeze(1)
        logpas = logpas.reshape(-1).unsqueeze(1)
        entropies = entropies.reshape(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).reshape(-1).unsqueeze(1)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).reshape(-1).unsqueeze(1)
        
        T -= 1
        T *= self.n_workers
        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert logpas.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = self.policy_loss_weight * policy_loss + \
                self.value_loss_weight * value_loss + \
                self.entropy_loss_weight * entropy_loss

        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 
                                       self.ac_model_max_grad_norm)
        self.ac_optimizer.step()

    def interaction_step(self, states, envs):
        actions, is_exploratory, logpas, entropies, values = self.ac_model.full_pass(states)
        new_states, rewards, is_terminals, _ = envs.step(actions)

        self.logpas.append(logpas) ; self.entropies.append(entropies)
        self.rewards.append(rewards) ; self.values.append(values)
        
        self.running_reward += rewards
        self.running_timestep += 1
        self.running_exploration += is_exploratory[:,np.newaxis].astype(int)

        return new_states, is_terminals

    def train(self, make_envs_fn, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_envs_fn = make_envs_fn
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        envs = self.make_envs_fn(make_env_fn, make_env_kargs, self.seed, self.n_workers)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.running_timestep = np.array([[0.],] * self.n_workers)
        self.running_reward = np.array([[0.],] * self.n_workers)
        self.running_exploration = np.array([[0.],] * self.n_workers)
        self.running_seconds = np.array([[time.time()],] * self.n_workers)
        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.evaluation_scores = [], []
        self.episode_exploration = []

        self.ac_model = self.ac_model_fn(nS, nA)
        self.ac_optimizer = self.ac_optimizer_fn(self.ac_model, 
                                                 self.ac_optimizer_lr)
        
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        states = envs.reset()
        
        # collect n_steps rollout
        episode, n_steps_start = 0, 0
        self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
        for step in count(start=1):
            states, is_terminals = self.interaction_step(states, envs)
            
            if is_terminals.sum() or step - n_steps_start == self.max_n_steps:
                past_limits_enforced = envs._past_limit()
                is_failure = np.logical_and(is_terminals, np.logical_not(past_limits_enforced))
                next_values = self.ac_model.evaluate_state(
                    states).detach().numpy() * (1 - is_failure)
                self.rewards.append(next_values) ; self.values.append(torch.Tensor(next_values))
                self.optimize_model()
                self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
                n_steps_start = step
                
            # stats
            if is_terminals.sum():
                episode_done = time.time()
                evaluation_score, _ = self.evaluate(self.ac_model, env)
                self.save_checkpoint(episode, self.ac_model)
            
                for i in range(self.n_workers):
                    if is_terminals[i]:
                        states[i] = envs.reset(rank=i)
                        self.episode_timestep.append(self.running_timestep[i][0])
                        self.episode_reward.append(self.running_reward[i][0])
                        self.episode_exploration.append(self.running_exploration[i][0]/self.running_timestep[i][0])
                        self.episode_seconds.append(episode_done - self.running_seconds[i][0])
                        training_time += self.episode_seconds[-1]
                        self.evaluation_scores.append(evaluation_score)
                        episode += 1

                        mean_10_reward = np.mean(self.episode_reward[-10:])
                        std_10_reward = np.std(self.episode_reward[-10:])
                        mean_100_reward = np.mean(self.episode_reward[-100:])
                        std_100_reward = np.std(self.episode_reward[-100:])
                        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
                        std_100_eval_score = np.std(self.evaluation_scores[-100:])
                        mean_100_exp_rat = np.mean(self.episode_exploration[-100:])
                        std_100_exp_rat = np.std(self.episode_exploration[-100:])
                        
                        total_step = int(np.sum(self.episode_timestep))
                        wallclock_elapsed = time.time() - training_start
                        result[episode-1] = total_step, mean_100_reward, \
                            mean_100_eval_score, training_time, wallclock_elapsed

                # debug stuff
                reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
                reached_max_minutes = wallclock_elapsed >= max_minutes * 60            
                reached_max_episodes = episode + self.n_workers >= max_episodes
                reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
                training_is_over = reached_max_minutes or \
                                   reached_max_episodes or \
                                   reached_goal_mean_reward

                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                debug_message = 'el {}, ep {:04}, ts {:06}, '
                debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
                debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
                debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
                debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
                debug_message = debug_message.format(
                    elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                    mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                    mean_100_eval_score, std_100_eval_score)
                print(debug_message, end='\r', flush=True)
                if reached_debug_time or training_is_over:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()
                if training_is_over:
                    if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                    if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                    if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                    break

                # reset running variables for next time around
                self.running_timestep *= 1 - is_terminals
                self.running_reward *= 1 - is_terminals
                self.running_exploration *= 1 - is_terminals
                self.running_seconds[is_terminals.astype(bool)] = time.time()

        final_eval_score, score_std = self.evaluate(self.ac_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        envs.close() ; del envs
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else: 
                    a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.ac_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.ac_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.ac_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.ac_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))
