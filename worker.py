'''Replay buffer, learner and actor'''
import time
import random
import os
import math
from copy import deepcopy
from typing import List, Tuple
import threading
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from model import Network, AgentState
from environment import create_env
from priority_tree import PriorityTree
import config

############################## Replay Buffer ##############################


@dataclass
class Block:
    obs: np.array
    last_action: np.array
    last_reward: np.array
    action: np.array
    n_step_reward: np.array
    gamma: np.array
    hidden: np.array
    num_sequences: int
    burn_in_steps: np.array
    learning_steps: np.array
    forward_steps: np.array


class ReplayBuffer:
    def __init__(self, sample_queue_list, batch_queue, priority_queue, buffer_capacity=config.buffer_capacity, sequence_len=config.block_length,
                alpha=config.prio_exponent, beta=config.importance_sampling_exponent,
                batch_size=config.batch_size):

        self.buffer_capacity = buffer_capacity
        self.sequence_len = config.learning_steps
        self.num_sequences = buffer_capacity//self.sequence_len
        self.block_len = config.block_length
        self.num_blocks = self.buffer_capacity // self.block_len
        self.seq_pre_block = self.block_len // self.sequence_len

        self.block_ptr = 0

        self.priority_tree = PriorityTree(self.num_sequences, alpha, beta)

        self.batch_size = batch_size

        self.env_steps = 0
        
        self.num_episodes = 0
        self.episode_reward = 0

        self.training_steps = 0
        self.last_training_steps = 0
        self.sum_loss = 0

        self.lock = threading.Lock()

        self.size = 0
        self.last_size = 0

        self.buffer = [None] * self.num_blocks

        self.sample_queue_list, self.batch_queue, self.priority_queue = sample_queue_list, batch_queue, priority_queue

    def __len__(self):
        return self.size

    def run(self):
        background_thread = threading.Thread(target=self.add_data, daemon=True)
        background_thread.start()

        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()

        background_thread = threading.Thread(target=self.update_data, daemon=True)
        background_thread.start()

        log_interval = config.log_interval

        while True:
            print(f'buffer size: {self.size}')
            print(f'buffer update speed: {(self.size-self.last_size)/log_interval}/s')
            self.last_size = self.size
            print(f'number of environment steps: {self.env_steps}')
            if self.num_episodes != 0:
                print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
                # print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
                self.episode_reward = 0
                self.num_episodes = 0
            print(f'number of training steps: {self.training_steps}')
            print(f'training speed: {(self.training_steps-self.last_training_steps)/log_interval}/s')
            if self.training_steps != self.last_training_steps:
                print(f'loss: {self.sum_loss/(self.training_steps-self.last_training_steps):.4f}')
                self.last_training_steps = self.training_steps
                self.sum_loss = 0
            self.last_env_steps = self.env_steps
            print()

            if self.training_steps == config.training_steps:
                break
            else:
                time.sleep(log_interval)

    def prepare_data(self):
        while self.size < config.learning_starts:
            time.sleep(1)

        while True:
            if not self.batch_queue.full():
                data = self.sample_batch()
                self.batch_queue.put(data)
            else:
                time.sleep(0.1)

    def add_data(self):
        while True:
            for sample_queue in self.sample_queue_list:
                if not sample_queue.empty():
                    data = sample_queue.get_nowait()
                    self.add(*data)

    def update_data(self):

        while True:
            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)
            else:
                time.sleep(0.1)


    def add(self, block: Block, priority: np.array, episode_reward: float):

        with self.lock:

            idxes = np.arange(self.block_ptr*self.seq_pre_block, (self.block_ptr+1)*self.seq_pre_block, dtype=np.int64)

            self.priority_tree.update(idxes, priority)

            if self.buffer[self.block_ptr] is not None:
                self.size -= np.sum(self.buffer[self.block_ptr].learning_steps).item()

            self.size += np.sum(block.learning_steps).item()

            self.buffer[self.block_ptr] = block

            self.env_steps += np.sum(block.learning_steps, dtype=np.int32)

            self.block_ptr = (self.block_ptr+1) % self.num_blocks
            if episode_reward:
                self.episode_reward += episode_reward
                self.num_episodes += 1

    def sample_batch(self):
        '''sample one batch of training data'''
        batch_obs, batch_last_action, batch_last_reward, batch_hidden, batch_action, batch_reward, batch_gamma = [], [], [], [], [], [], []
        burn_in_steps, learning_steps, forward_steps = [], [], []

        with self.lock:

            idxes, is_weights = self.priority_tree.sample(self.batch_size)

            block_idxes = idxes // self.seq_pre_block
            sequence_idxes = idxes % self.seq_pre_block


            for block_idx, sequence_idx  in zip(block_idxes, sequence_idxes):

                block = self.buffer[block_idx]

                assert sequence_idx < block.num_sequences, 'index is {} but size is {}'.format(sequence_idx, self.seq_pre_block_buf[block_idx])

                burn_in_step = block.burn_in_steps[sequence_idx]
                learning_step = block.learning_steps[sequence_idx]
                forward_step = block.forward_steps[sequence_idx]
                
                start_idx = block.burn_in_steps[0] + np.sum(block.learning_steps[:sequence_idx])

                obs = block.obs[start_idx-burn_in_step:start_idx+learning_step+forward_step]
                last_action = block.last_action[start_idx-burn_in_step:start_idx+learning_step+forward_step]
                last_reward = block.last_reward[start_idx-burn_in_step:start_idx+learning_step+forward_step]
                obs, last_action, last_reward = torch.from_numpy(obs), torch.from_numpy(last_action), torch.from_numpy(last_reward)
                
                start_idx = np.sum(block.learning_steps[:sequence_idx])
                end_idx = start_idx + block.learning_steps[sequence_idx]
                action = block.action[start_idx:end_idx]
                reward = block.n_step_reward[start_idx:end_idx]
                gamma = block.gamma[start_idx:end_idx]
                hidden = block.hidden[sequence_idx]
                
                batch_obs.append(obs)
                batch_last_action.append(last_action)
                batch_last_reward.append(last_reward)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_gamma.append(gamma)
                batch_hidden.append(hidden)

                burn_in_steps.append(burn_in_step)
                learning_steps.append(learning_step)
                forward_steps.append(forward_step)

            batch_obs = pad_sequence(batch_obs, batch_first=True)
            batch_last_action = pad_sequence(batch_last_action, batch_first=True)
            batch_last_reward = pad_sequence(batch_last_reward, batch_first=True)

            is_weights = np.repeat(is_weights, learning_steps)


            data = (
                batch_obs,
                batch_last_action,
                batch_last_reward,
                torch.from_numpy(np.stack(batch_hidden)).transpose(0, 1),

                torch.from_numpy(np.concatenate(batch_action)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_reward)),
                torch.from_numpy(np.concatenate(batch_gamma)),

                torch.ByteTensor(burn_in_steps),
                torch.ByteTensor(learning_steps),
                torch.ByteTensor(forward_steps),

                idxes,
                torch.from_numpy(is_weights.astype(np.float32)),
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

            self.priority_tree.update(idxes, td_errors)

        self.training_steps += 1
        self.sum_loss += loss




############################## Learner ##############################

def calculate_mixed_td_errors(td_error, learning_steps):
    
    start_idx = 0
    mixed_td_errors = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        mixed_td_errors[i] = 0.9*td_error[start_idx:start_idx+steps].max() + 0.1*td_error[start_idx:start_idx+steps].mean()
        start_idx += steps
    
    return mixed_td_errors

class Learner:
    def __init__(self, batch_queue, priority_queue, model, grad_norm: int = config.grad_norm,
                lr: float = config.lr, eps:float = config.eps, game_name: str = config.game_name,
                target_net_update_interval: int = config.target_net_update_interval, save_interval: int = config.save_interval):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_net = deepcopy(model)
        self.online_net.to(self.device)
        self.online_net.train()
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=eps)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.grad_norm = grad_norm
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue
        self.num_updates = 0
        self.done = False

        self.target_net_update_interval = target_net_update_interval
        self.save_interval = save_interval

        self.batched_data = []

        self.shared_model = model

        self.game_name = game_name

    def store_weights(self):
        self.shared_model.load_state_dict(self.online_net.state_dict())

    def prepare_data(self):

        while True:
            if not self.batch_queue.empty() and len(self.batched_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batched_data.append(data)
            else:
                time.sleep(0.1)

    def run(self):
        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()
        time.sleep(2)

        start_time = time.time()
        while self.num_updates < config.training_steps:
            
            while not self.batched_data:
                time.sleep(1)
            data = self.batched_data.pop(0)

            batch_obs, batch_last_action, batch_last_reward, batch_hidden, batch_action, batch_n_step_reward, batch_n_step_gamma, burn_in_steps, learning_steps, forward_steps, idxes, is_weights, old_ptr, env_steps = data
            batch_obs, batch_last_action, batch_last_reward = batch_obs.to(self.device), batch_last_action.to(self.device), batch_last_reward.to(self.device)
            batch_hidden, batch_action = batch_hidden.to(self.device), batch_action.to(self.device)
            batch_n_step_reward, batch_n_step_gamma = batch_n_step_reward.to(self.device), batch_n_step_gamma.to(self.device)
            is_weights = is_weights.to(self.device)

            batch_obs, batch_last_action = batch_obs.float(), batch_last_action.float()
            batch_action = batch_action.long()
            burn_in_steps, learning_steps, forward_steps = burn_in_steps, learning_steps, forward_steps

            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            batch_obs = batch_obs / 255

            # double q learning
            with torch.no_grad():
                batch_action_ = self.online_net.calculate_q_(batch_obs, batch_last_action, batch_last_reward, batch_hidden, burn_in_steps, learning_steps, forward_steps).argmax(1).unsqueeze(1)
                batch_q_ = self.target_net.calculate_q_(batch_obs, batch_last_action, batch_last_reward, batch_hidden, burn_in_steps, learning_steps, forward_steps).gather(1, batch_action_).squeeze(1)
            
            target_q = self.value_rescale(batch_n_step_reward + batch_n_step_gamma * self.inverse_value_rescale(batch_q_))
            # target_q = batch_n_step_reward + batch_n_step_gamma * batch_q_

            batch_q = self.online_net.calculate_q(batch_obs, batch_last_action, batch_last_reward, batch_hidden, burn_in_steps, learning_steps).gather(1, batch_action).squeeze(1)
            
            loss = (is_weights * self.loss_fn(batch_q, target_q)).mean()

            
            td_errors = (target_q-batch_q).detach().clone().squeeze().abs().cpu().float().numpy()

            priorities = calculate_mixed_td_errors(td_errors, learning_steps.numpy())

            # automatic mixed precision training
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
            self.optimizer.step()

            self.num_updates += 1

            self.priority_queue.put((idxes, priorities, old_ptr, loss.item()))

            # store new weights in shared memory
            if self.num_updates % 4 == 0:
                self.store_weights()

            # update target net
            if self.num_updates % self.target_net_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            # save model 
            if self.num_updates % self.save_interval == 0:
                torch.save((self.online_net.state_dict(), self.num_updates, env_steps, (time.time()-start_time)/60), os.path.join('models', '{}{}.pth'.format(self.game_name, self.num_updates)))

    @staticmethod
    def value_rescale(value, eps=1e-3):
        return value.sign()*((value.abs()+1).sqrt()-1) + eps*value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4*eps*(value.abs()+1+eps)).sqrt() - 1) / (2*eps)
        return value.sign() * (temp.square() - 1)


############################## Actor ##############################

class LocalBuffer:
    '''store transitions of one episode'''
    def __init__(self, action_dim: int, forward_steps: int = config.forward_steps,
                burn_in_steps = config.burn_in_steps, learning_steps: int = config.learning_steps, 
                gamma: float = config.gamma, hidden_dim: int = config.hidden_dim, block_length: int = config.block_length):
        
        self.action_dim = action_dim
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
        self.obs_buffer = [init_obs]
        self.last_action_buffer = [np.array([1 if i == 0 else 0 for i in range(self.action_dim)], dtype=bool)]
        self.last_reward_buffer = [0]
        self.hidden_buffer = [np.zeros((2, self.hidden_dim), dtype=np.float32)]
        self.action_buffer = []
        self.reward_buffer = []
        self.qval_buffer = []
        self.curr_burn_in_steps = 0
        self.size = 0
        self.sum_reward = 0
        self.done = False

    def add(self, action: int, reward: float, next_obs: np.ndarray, q_value: np.ndarray, hidden_state: np.ndarray):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.hidden_buffer.append(hidden_state)
        self.obs_buffer.append(next_obs)
        self.last_action_buffer.append(np.array([1 if i == action else 0 for i in range(self.action_dim)], dtype=bool))
        self.last_reward_buffer.append(reward)
        self.qval_buffer.append(q_value)
        self.sum_reward += reward
        self.size += 1
    
    def finish(self, last_qval: np.ndarray = None) -> Tuple:
        assert self.size <= self.block_length
        # assert len(self.last_action_buffer) == self.curr_burn_in_steps + self.size + 1

        num_sequences = math.ceil(self.size/self.learning_steps)

        max_forward_steps = min(self.size, self.forward_steps)
        n_step_gamma = [self.gamma**self.forward_steps] * (self.size-max_forward_steps)

        # last_qval is none means episode done 
        if last_qval is not None:
            self.qval_buffer.append(last_qval)
            n_step_gamma.extend([self.gamma**i for i in reversed(range(1, max_forward_steps+1))])
        else:
            self.done = True
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[0]))
            n_step_gamma.extend([0 for _ in range(max_forward_steps)]) # set gamma to 0 so don't need 'done'

        n_step_gamma = np.array(n_step_gamma, dtype=np.float32)

        obs = np.stack(self.obs_buffer)
        last_action = np.stack(self.last_action_buffer)
        last_reward = np.array(self.last_reward_buffer, dtype=np.float32)

        hiddens = np.stack(self.hidden_buffer[slice(0, self.size, self.learning_steps)])

        actions = np.array(self.action_buffer, dtype=np.uint8)

        qval_buffer = np.concatenate(self.qval_buffer)
        reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps-1)]
        n_step_reward = np.convolve(reward_buffer, 
                                    [self.gamma**(self.forward_steps-1-i) for i in range(self.forward_steps)],
                                    'valid').astype(np.float32)

        burn_in_steps = np.array([min(i*self.learning_steps+self.curr_burn_in_steps, self.burn_in_steps) for i in range(num_sequences)], dtype=np.uint8)
        learning_steps = np.array([min(self.learning_steps, self.size-i*self.learning_steps) for i in range(num_sequences)], dtype=np.uint8)
        forward_steps = np.array([min(self.forward_steps, self.size+1-np.sum(learning_steps[:i+1])) for i in range(num_sequences)], dtype=np.uint8)
        assert forward_steps[-1] == 1 and burn_in_steps[0] == self.curr_burn_in_steps
        # assert last_action.shape[0] == self.curr_burn_in_steps + np.sum(learning_steps) + 1

        max_qval = np.max(qval_buffer[max_forward_steps:self.size+1], axis=1)
        max_qval = np.pad(max_qval, (0, max_forward_steps-1), 'edge')
        target_qval = qval_buffer[np.arange(self.size), actions]

        td_errors = np.abs(n_step_reward + n_step_gamma * max_qval - target_qval, dtype=np.float32)
        priorities = np.zeros(self.block_length//self.learning_steps, dtype=np.float32)
        priorities[:num_sequences] = calculate_mixed_td_errors(td_errors, learning_steps)

        # save burn in information for next block
        self.obs_buffer = self.obs_buffer[-self.burn_in_steps-1:]
        self.last_action_buffer = self.last_action_buffer[-self.burn_in_steps-1:]
        self.last_reward_buffer = self.last_reward_buffer[-self.burn_in_steps-1:]
        self.hidden_buffer = self.hidden_buffer[-self.burn_in_steps-1:]
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.qval_buffer.clear()
        self.curr_burn_in_steps = len(self.obs_buffer)-1
        self.size = 0
        
        block = Block(obs, last_action, last_reward, actions, n_step_reward, n_step_gamma, hiddens, num_sequences, burn_in_steps, learning_steps, forward_steps)
        return [block, priorities, self.sum_reward if self.done else None]


class Actor:
    def __init__(self, epsilon: float, model, sample_queue, obs_shape: np.ndarray = config.obs_shape,
                max_episode_steps: int = config.max_episode_steps, block_length: int = config.block_length):

        self.env = create_env(noop_start=True)
        self.action_dim = self.env.action_space.n
        self.model = Network(self.env.action_space.n)
        self.model.eval()
        self.local_buffer = LocalBuffer(self.action_dim)

        self.epsilon = epsilon
        self.shared_model = model
        self.sample_queue = sample_queue
        self.max_episode_steps = max_episode_steps
        self.block_length = block_length

    def run(self):
        
        actor_steps = 0

        while True:

            done = False
            agent_state = self.reset()
            episode_steps = 0

            while not done and episode_steps < self.max_episode_steps:
                
                with torch.no_grad():
                    q_value, hidden = self.model(agent_state)

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_value, 1).item()

                # apply action in env
                next_obs, reward, done, _ = self.env.step(action)

                agent_state.update(next_obs, action, reward, hidden)

                episode_steps += 1
                actor_steps += 1

                self.local_buffer.add(action, reward, next_obs, q_value.numpy(), torch.cat(hidden).numpy())

                if done:
                    block = self.local_buffer.finish()
                    self.sample_queue.put(block)

                elif len(self.local_buffer) == self.block_length or episode_steps == self.max_episode_steps:
                    with torch.no_grad():
                        q_value, hidden = self.model(agent_state)

                    block = self.local_buffer.finish(q_value.numpy())

                    if self.epsilon > 0.01:
                        block[2] = None
                    self.sample_queue.put(block)

                if actor_steps % 400 == 0:
                    self.update_weights()

                
    def update_weights(self):
        '''load the latest weights from shared model'''
        self.model.load_state_dict(self.shared_model.state_dict())
    
    def reset(self):
        obs = self.env.reset()
        self.local_buffer.reset(obs)

        state = AgentState(torch.from_numpy(obs).unsqueeze(0), self.action_dim)

        return state

