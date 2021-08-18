'''Replay buffer, learner and actor'''
import time
import random
import os
import math
from copy import deepcopy
from typing import List, Tuple
import threading
from dataclasses import dataclass
import ray
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import numba as nb
from model import Network
from environment import create_env
from priority_tree import create_ptree, ptree_sample, ptree_update
import config

DEFAULT_NP_FLOAT = np.float16 if config.amp else np.float32

############################## Replay Buffer ##############################

@ray.remote(num_cpus=1)
class ReplayBuffer:
    def __init__(self, buffer_capacity=config.buffer_capacity, sequence_len=config.block_length,
                alpha=config.prio_exponent, beta=config.importance_sampling_exponent,
                batch_size=config.batch_size, frame_stack=config.frame_stack):

        self.buffer_capacity = buffer_capacity
        self.sequence_len = config.learning_steps
        self.num_sequences = buffer_capacity//self.sequence_len
        self.block_len = config.block_length
        self.num_blocks = self.buffer_capacity // self.block_len
        self.seq_pre_block = self.block_len // self.sequence_len

        self.block_ptr = 0

        # prioritized experience replay
        self.num_layers, self.priority_tree = create_ptree(self.num_sequences)
        self.prio_exponent = alpha
        self.is_exponent = beta

        self.batch_size = batch_size
        self.frame_stack = frame_stack

        self.env_steps = 0
        self.last_env_steps = 0

        self.num_episodes = 0
        self.episode_reward = 0

        self.num_training_steps = 0
        self.last_training_steps = 0
        self.sum_loss = 0

        self.lock = threading.Lock()
        
        # obs_buffer, last_action, hidden, action_buffer, reward_buffer, gamma, td_errors, num_sequences, burn_in_steps, learning_steps, forward_steps

        self.obs_buf = [None] * self.num_blocks
        self.last_action_buf = [None] * self.num_blocks
        self.hidden_buf = np.empty((self.num_blocks, self.seq_pre_block, 2, config.hidden_dim), dtype=DEFAULT_NP_FLOAT)
        self.act_buf = [None] * self.num_blocks
        self.rew_buf = [None] * self.num_blocks
        self.gamma_buf = [None] * self.num_blocks
        self.seq_pre_block_buf = np.zeros(self.num_blocks, dtype=np.uint8)
        self.learning_steps = np.zeros((self.num_blocks, self.seq_pre_block), dtype=np.long)
        self.burn_in_steps = np.zeros((self.num_blocks, self.seq_pre_block), dtype=np.long)
        self.forward_steps = np.zeros((self.num_blocks, self.seq_pre_block), dtype=np.long)

    def __len__(self):
        return np.sum(self.learning_steps).item()


    def add(self, block):
        '''Call by actors to add data to replaybuffer

        Args:
            block: tuples of data, each tuple represents a slot
                obs_buffer 0, last_action 1, hidden 2, action_buffer 3, reward_buffer 4, gamma 5, 
                td_errors 6, num_sequences 7, burn_in_steps 8, learning_steps 9, forward_steps 10
        '''

        with self.lock:

            curr_block_ptr = self.block_ptr
            self.block_ptr = (self.block_ptr+1) % self.num_blocks

            idxes = np.arange(curr_block_ptr*self.seq_pre_block, (curr_block_ptr+1)*self.seq_pre_block, dtype=np.int64)

            ptree_update(self.num_layers, self.priority_tree, self.prio_exponent, block[6], idxes)

            self.obs_buf[curr_block_ptr] = np.copy(block[0])
            self.last_action_buf[curr_block_ptr] = np.copy(block[1])
            self.hidden_buf[curr_block_ptr, :block[7]] = block[2]
            self.act_buf[curr_block_ptr] = np.copy(block[3])
            self.rew_buf[curr_block_ptr] = np.copy(block[4])
            self.gamma_buf[curr_block_ptr] = np.copy(block[5])

            self.seq_pre_block_buf[curr_block_ptr] = block[7]

            self.burn_in_steps[curr_block_ptr, :block[7]] = block[8]
            self.learning_steps[curr_block_ptr, :block[7]] = block[9]
            self.forward_steps[curr_block_ptr, :block[7]] = block[10]

            self.env_steps += np.sum(block[9], dtype=np.int)
            if block[11]:
                self.episode_reward += block[11]
                self.num_episodes += 1

    def sample_batch(self):
        '''sample one batch of training data'''
        batch_obs, batch_last_action, batch_hidden, batch_action, batch_reward, batch_gamma = [], [], [], [], [], []


        with self.lock:

            idxes, is_weights = ptree_sample(self.num_layers, self.priority_tree, self.is_exponent, self.batch_size)
            block_idxes = idxes // self.seq_pre_block
            sequence_idxes = idxes % self.seq_pre_block

            burn_in_steps = self.burn_in_steps[block_idxes, sequence_idxes]
            learning_steps = self.learning_steps[block_idxes, sequence_idxes]
            forward_steps = self.forward_steps[block_idxes, sequence_idxes]

            batch_hidden = self.hidden_buf[block_idxes, sequence_idxes]


            for block_idx, sequence_idx, burn_in_step, learning_step, forward_step  in zip(block_idxes, sequence_idxes, burn_in_steps, learning_steps, forward_steps):
                
                assert sequence_idx < self.seq_pre_block_buf[block_idx].item(), 'index is {} but size is {}'.format(sequence_idx, self.seq_pre_block_buf[block_idx])
                
                start_idx = self.burn_in_steps[block_idx, 0]+np.sum(self.learning_steps[block_idx, :sequence_idx]).item()

                obs_seq_len = self.obs_buf[block_idx].shape[0]
                assert start_idx-burn_in_step >= 0, start_idx-burn_in_step
                assert start_idx+learning_step+forward_step+self.frame_stack-1 <= obs_seq_len, f'{start_idx+learning_step+forward_step+self.frame_stack-1}, {obs_seq_len}'
                obs = self.obs_buf[block_idx][start_idx-burn_in_step:start_idx+learning_step+forward_step+self.frame_stack-1]
                last_action = self.last_action_buf[block_idx][start_idx-burn_in_step:start_idx+learning_step+forward_step]

                if burn_in_step + learning_step + forward_step < config.seq_len:
                    pad_len = config.seq_len - burn_in_step - learning_step - forward_step
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0)))
                    last_action = np.pad(last_action, ((0, pad_len), (0, 0)))
                
                start_idx = np.sum(self.learning_steps[block_idx, :sequence_idx])
                end_idx = start_idx + self.learning_steps[block_idx, sequence_idx]
                action = self.act_buf[block_idx][start_idx:end_idx]
                reward = self.rew_buf[block_idx][start_idx:end_idx]
                gamma = self.gamma_buf[block_idx][start_idx:end_idx]
                
                batch_obs.append(obs)
                batch_last_action.append(last_action)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_gamma.append(gamma)

        is_weights = np.repeat(is_weights, learning_steps)

        data = (
            torch.from_numpy(np.stack(batch_obs)),
            torch.from_numpy(np.stack(batch_last_action)),
            torch.from_numpy(batch_hidden).transpose(0, 1),

            torch.LongTensor(np.concatenate(batch_action)).unsqueeze(1),
            torch.from_numpy(np.concatenate(batch_reward)),
            torch.from_numpy(np.concatenate(batch_gamma)),

            torch.from_numpy(burn_in_steps),
            torch.from_numpy(learning_steps),
            torch.from_numpy(forward_steps),

            idxes,
            torch.from_numpy(is_weights.astype(DEFAULT_NP_FLOAT)),
            self.block_ptr,

            self.env_steps
        )

        return data

    def update_priorities(self, idxes: np.ndarray, td_errors: np.ndarray, old_ptr: int, loss: float):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the idxes that already been replaced by new data in replay buffer during training
            if self.block_ptr > old_ptr:
                # range from [old_ptr, self.seq_ptr)
                mask = (idxes < old_ptr*self.seq_pre_block) | (idxes >= self.block_ptr*self.seq_pre_block)
                idxes = idxes[mask]
                td_errors = td_errors[mask]
            elif self.block_ptr < old_ptr:
                # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.seq_pre_block) & (idxes >= self.block_ptr*self.seq_pre_block)
                idxes = idxes[mask]
                td_errors = td_errors[mask]

            # self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities)**self.alpha)
            ptree_update(self.num_layers, self.priority_tree, self.prio_exponent, td_errors, idxes)

        self.num_training_steps += 1
        self.sum_loss += loss

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False

    def log(self, log_interval):
        print(f'buffer size: {np.sum(self.learning_steps)}')
        print(f'buffer update speed: {(self.env_steps-self.last_env_steps)/log_interval}/s')
        print(f'number of environment steps: {self.env_steps}')
        if self.num_episodes != 0:
            print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
            self.episode_reward = 0
            self.num_episodes = 0
        print(f'number of training steps: {self.num_training_steps}')
        print(f'training speed: {(self.num_training_steps-self.last_training_steps)/log_interval}/s')
        if self.num_training_steps != self.last_training_steps:
            print(f'loss: {self.sum_loss/(self.num_training_steps-self.last_training_steps):.4f}')
            self.last_training_steps = self.num_training_steps
            self.sum_loss = 0
        self.last_env_steps = self.env_steps



############################## Learner ##############################

@nb.jit(nopython=True, cache=True)
def caculate_mixed_td_errors(td_error, learning_steps):
    
    start_idx = 0
    mixed_td_errors = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        mixed_td_errors[i] = 0.9*td_error[start_idx:start_idx+steps].max() + 0.1*td_error[start_idx:start_idx+steps].mean()
        start_idx += steps
    
    return mixed_td_errors

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: ReplayBuffer, game_name: str = config.game_name, grad_norm: int = config.grad_norm,
                lr: float = config.lr, eps:float = config.eps, amp: bool = config.amp,
                target_net_update_interval: int = config.target_net_update_interval, save_interval: int = config.save_interval):

        self.game_name = game_name
        self.online_net = Network(create_env().action_space.n)
        self.online_net.cuda()
        self.online_net.train()
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = Adam(self.online_net.parameters(), lr=lr, eps=eps)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.grad_norm = grad_norm
        self.buffer = buffer
        self.counter = 0
        self.done = False

        self.target_net_update_interval = target_net_update_interval
        self.save_interval = save_interval
        self.amp = amp

        self.batched_data = []

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.online_net.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()
        time.sleep(2)
        background_thread = threading.Thread(target=self.train, daemon=True)
        background_thread.start()
    
    def prepare_data(self):

        while True:
            while len(self.batched_data) < 4:
                data = ray.get(self.buffer.sample_batch.remote())
                self.batched_data.append(data)
            else:
                time.sleep(0.1)

    def train(self):
        scaler = GradScaler()
        obs_idx = torch.LongTensor([i+j for i in range(config.seq_len) for j in range(config.frame_stack)])
        torch.save((self.online_net.state_dict(), 0, 0), os.path.join('models', '{}0.pth'.format(self.game_name)))
        while self.counter < config.training_steps:

            if self.batched_data:
                data = self.batched_data.pop(0)
            else:
                print('empty')
                data = ray.get(self.buffer.sample_batch.remote())

            batch_obs, batch_last_action, batch_hidden, batch_action, batch_n_step_reward, batch_n_step_gamma, burn_in_steps, learning_steps, forward_steps, idxes, is_weights, old_ptr, env_steps = data
            batch_obs, batch_last_action, batch_hidden = batch_obs.cuda(), batch_last_action.cuda(), batch_hidden.cuda()
            batch_action, batch_n_step_reward, batch_n_step_gamma = batch_action.cuda(), batch_n_step_reward.cuda(), batch_n_step_gamma.cuda()
            is_weights = is_weights.cuda()

            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            with autocast(enabled=self.amp):
                
                # stack observation and preprocess
                batch_obs = torch.stack([obs[obs_idx] for obs in batch_obs]).view(config.batch_size, config.seq_len, config.frame_stack, 84, 84)
                batch_obs = batch_obs / 255
                batch_last_action = batch_last_action.float()

                # double q learning
                batch_action_ = self.online_net.caculate_q_(batch_obs, batch_last_action, batch_hidden, burn_in_steps, learning_steps, forward_steps).argmax(1).unsqueeze(1)
                batch_q_ = self.target_net.caculate_q_(batch_obs, batch_last_action, batch_hidden, burn_in_steps, learning_steps, forward_steps).gather(1, batch_action_).squeeze(1)
                
                target_q = self.value_rescale(batch_n_step_reward + batch_n_step_gamma * self.inverse_value_rescale(batch_q_))
                # target_q = batch_n_step_reward + batch_n_step_gamma * batch_q_

                batch_q = self.online_net.caculate_q(batch_obs, batch_last_action, batch_hidden, burn_in_steps, learning_steps).gather(1, batch_action).squeeze(1)
                
                loss = 0.5 * (is_weights * self.loss_fn(batch_q, target_q)).mean()
            
            td_errors = (target_q-batch_q).detach().clone().squeeze().abs().cpu().float().numpy()

            priorities = caculate_mixed_td_errors(td_errors, learning_steps.numpy())

            # automatic mixed precision training
            if self.amp:
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
                self.optimizer.step()

            self.counter += 1

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr, loss.item())

            # store new weights in shared memory
            if self.counter % 2  == 0:
                self.store_weights()

            # update target net
            if self.counter % self.target_net_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            # save model 
            if self.counter % self.save_interval == 0:
                torch.save((self.online_net.state_dict(), self.counter, env_steps), os.path.join('models', '{}{}.pth'.format(self.game_name, self.counter//self.save_interval)))

    @staticmethod
    def value_rescale(value, eps=1e-2):
        return value.sign()*((value.abs()+1).sqrt()-1) + eps*value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-2):
        temp = ((1 + 4*eps*(value.abs()+1+eps)).sqrt() - 1) / (2*eps)
        return value.sign() * (temp.square() - 1)


############################## Actor ##############################

class LocalBuffer:
    '''store transition of one episode'''
    def __init__(self, action_dim: int, forward_steps: int = config.forward_steps, frame_stack: int = config.frame_stack,
                burn_in_steps = config.burn_in_steps, learning_steps: int = config.learning_steps, 
                gamma: float = config.gamma, hidden_dim: int = config.hidden_dim, block_length: int = config.block_length):
        
        self.action_dim = action_dim
        self.frame_stack = frame_stack
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.forward_steps = forward_steps
        self.learning_steps = learning_steps
        self.burn_in_steps = burn_in_steps
        self.block_length = block_length
        self.curr_burn_in_steps = 0
        
    def __len__(self):
        return self.size
    
    def reset(self, init_obs: np.ndarray):
        self.obs_buffer = [init_obs for _ in range(self.frame_stack)]
        self.last_action_buffer = [np.zeros(self.action_dim, dtype=np.bool)]
        self.hidden_buffer = [np.zeros((2, self.hidden_dim), dtype=DEFAULT_NP_FLOAT)]
        self.action_buffer = []
        self.reward_buffer = []
        self.qval_buffer = []
        self.curr_burn_in_steps = 0
        self.size = 0
        self.sum_reward = 0
        self.done = False

    def add(self, action: int, reward: float, next_obs: np.ndarray, q_value: np.ndarray, hidden_state: np.ndarray):
        self.hidden_buffer.append(hidden_state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.obs_buffer.append(next_obs)
        last_action = np.zeros(self.action_dim, dtype=np.bool)
        last_action[action] = 1
        self.last_action_buffer.append(last_action)
        self.qval_buffer.append(q_value)
        self.sum_reward += reward
        self.size += 1
    
    def finish(self, last_qval: np.ndarray = None) -> Tuple:
        assert self.size <= self.block_length
        assert len(self.obs_buffer) == self.frame_stack + self.curr_burn_in_steps + self.size, f'{len(self.obs_buffer)} {self.frame_stack+self.curr_burn_in_steps+self.size}'
        assert len(self.last_action_buffer) == self.curr_burn_in_steps + self.size + 1

        num_sequences = math.ceil(self.size/self.learning_steps)

        max_forward_steps = min(self.size, self.forward_steps)
        n_step_gamma = [self.gamma**self.forward_steps] * (self.size-max_forward_steps)

        if last_qval is not None:
            self.qval_buffer.append(last_qval)
            n_step_gamma.extend([self.gamma**i for i in reversed(range(1, max_forward_steps+1))])
        else:
            self.done = True
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[0]))
            n_step_gamma.extend([0 for _ in range(max_forward_steps)]) # set gamma to 0 so don't need 'done'

        n_step_gamma = np.array(n_step_gamma, dtype=DEFAULT_NP_FLOAT)
        observations = np.stack(self.obs_buffer)
        last_action = np.stack(self.last_action_buffer)
        hiddens = np.stack(self.hidden_buffer[slice(0, self.size, self.learning_steps)])
        assert hiddens.shape[0] == num_sequences, f'{hiddens.shape} {num_sequences}'
        actions = np.array(self.action_buffer, dtype=np.uint8)
        qval_buffer = np.concatenate(self.qval_buffer).astype(DEFAULT_NP_FLOAT)
        reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps-1)]
        n_step_reward = np.convolve(reward_buffer, 
                                    [self.gamma**(self.forward_steps-1-i) for i in range(self.forward_steps)],
                                    'valid').astype(DEFAULT_NP_FLOAT)

        burn_in_steps = np.array([min(i*self.learning_steps+self.curr_burn_in_steps, self.burn_in_steps) for i in range(num_sequences)], dtype=np.uint8)
        learning_steps = np.array([min(self.learning_steps, self.size-i*self.learning_steps) for i in range(num_sequences)], dtype=np.long)
        forward_steps = np.array([min(self.forward_steps, self.size+1-np.sum(learning_steps[:i+1])) for i in range(num_sequences)], dtype=np.uint8)
        assert forward_steps[-1] == 1 and burn_in_steps[0] == self.curr_burn_in_steps
        assert last_action.shape[0] == self.curr_burn_in_steps + np.sum(learning_steps) + 1
        assert observations.shape[0] == self.curr_burn_in_steps + np.sum(learning_steps) + self.frame_stack

        max_qval = np.max(qval_buffer[max_forward_steps:self.size+1], axis=1)
        max_qval = np.pad(max_qval, (0, max_forward_steps-1), 'edge')
        target_qval = qval_buffer[np.arange(self.size), actions]
        td_errors = np.abs(n_step_reward + n_step_gamma*max_qval - target_qval, dtype=np.float32)
        priorities = np.zeros(self.block_length//self.learning_steps, dtype=np.float32)
        priorities[:num_sequences] = caculate_mixed_td_errors(td_errors, learning_steps)

        # save burn in information for next block
        self.obs_buffer = self.obs_buffer[-self.frame_stack-self.burn_in_steps:]
        self.last_action_buffer = self.last_action_buffer[-self.burn_in_steps-1:]
        self.hidden_buffer = self.hidden_buffer[-self.burn_in_steps-1:]
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.qval_buffer.clear()
        self.curr_burn_in_steps = len(self.last_action_buffer)-1
        self.size = 0
        
        return [observations, last_action, hiddens, actions, n_step_reward, n_step_gamma, priorities, num_sequences, burn_in_steps, learning_steps, forward_steps, self.sum_reward if self.done else None]


# @dataclass
# class AgentState:
#     stacked_obs: torch.Tensor
#     last_action: torch.Tensor
#     hidden_state: Tuple[torch.Tensor, torch.Tensor]


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, epsilon: float, learner: Learner, buffer: ReplayBuffer, obs_shape: np.ndarray = config.obs_shape,
                max_episode_steps: int = config.max_episode_steps, block_length: int = config.block_length):

        self.env = create_env(noop_start=True, clip_rewards=False)
        self.action_dim = self.env.action_space.n
        self.model = Network(self.env.action_space.n)
        self.model.eval()
        self.local_buffer = LocalBuffer(self.action_dim)
        
        self.stacked_obs = torch.empty((1, *obs_shape), dtype=torch.float32)
        self.epsilon = epsilon
        self.learner = learner
        self.replay_buffer = buffer
        self.max_episode_steps = max_episode_steps
        self.block_length = block_length
        self.counter = 0
        self.env_steps = 0
        self.done = False

        self.last_action = torch.zeros((1, self.action_dim), dtype=torch.float32)

    def run(self):
        
        self.reset()
        
        while True:
            obs = self.stacked_obs.clone()
            # print(self.last_action)
            action, qval, hidden = self.model.step(obs, self.last_action)

            if random.random() < self.epsilon:
                action = self.env.action_space.sample()

            # apply action in env
            next_obs, reward, self.done, _ = self.env.step(action)

            self.last_action.fill_(0)
            self.last_action[0, action] = 1

            self.stacked_obs = self.stacked_obs.roll(-1, 1)
            self.stacked_obs[0, -1] = torch.from_numpy(next_obs) / 255

            self.env_steps += 1

            self.local_buffer.add(action, reward, next_obs, qval, hidden)

            if self.done or self.env_steps == self.max_episode_steps:
                block = self.local_buffer.finish()
                if self.epsilon > 0.02:
                    block[-1] = None
                self.reset()
                self.replay_buffer.add.remote(block)

            elif len(self.local_buffer) == self.block_length:
                obs = self.stacked_obs.clone()
                with torch.no_grad():
                    q_val = self.model(obs, self.last_action, self.model.hidden_state)
                block = self.local_buffer.finish(q_val)
                self.replay_buffer.add.remote(block)
                
            self.counter += 1
            if self.counter == 400:
                self.update_weights()
                self.counter = 0
                
    def update_weights(self):
        '''load latest weights from learner'''
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)

    # def select_action(self, q_value: torch.Tensor) -> int:
    #     if random.random() < self.epsilon:
    #         return self.env.action_space.sample()
    #     else:
    #         return torch.argmax(q_value, 1).item()
    
    def reset(self):
        obs = self.env.reset()
        self.model.reset()
        self.stacked_obs[0, :] = torch.from_numpy(obs) / 255
        self.local_buffer.reset(obs)
        self.last_action.fill_(0)
        self.env_steps = 0
        self.done = False

