'''Replay buffer, learner and actor'''
import time
import random
import os
import math
from copy import deepcopy
from typing import List, Tuple
import threading
import ray
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import numba as nb
from model import Network
from environment import creat_env
import config

NP_DEFAULT_FLOAT = np.float16 if config.amp else np.float32


############################## Replay Buffer ##############################
@nb.jit
def caculate_priorities(td_error, learning_steps):
    # assert td_error.size == np.sum(learning_steps)
    start_idx = 0
    priorities = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        priorities[i] = 0.9*td_error[start_idx:start_idx+steps].max() + 0.1*td_error[start_idx:start_idx+steps].mean()
        start_idx += steps
    
    return priorities


@nb.jit
def sample(tree, layer, batch_size):
    p_sum = tree[0]
    interval = p_sum/batch_size

    prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, batch_size)

    idxes = np.zeros(batch_size, dtype=np.int64)
    for _ in range(layer-1):
        nodes = tree[idxes*2+1]
        idxes = np.where(prefixsums<nodes, idxes*2+1, idxes*2+2)
        prefixsums = np.where(idxes%2==0, prefixsums-tree[idxes-1], prefixsums)
    
    priorities = tree[idxes]
    idxes -= 2**(layer-1)-1

    return idxes, priorities

@nb.jit(nb.void(nb.float64[:], nb.int64, nb.int64[:], nb.float32[:]))
def update(tree, layer, idxes: np.ndarray, priorities: np.ndarray):
    idxes = idxes+2**(layer-1)-1
    tree[idxes] = priorities

    for _ in range(layer-1):
        idxes = (idxes-1) // 2
        idxes = np.unique(idxes)
        tree[idxes] = tree[2*idxes+1] + tree[2*idxes+2]


class SumTree:
    '''store priority for prioritized experience replay''' 
    def __init__(self, capacity: int):
        self.capacity = capacity

        self.layer = 1
        while capacity > 1:
            self.layer += 1
            capacity = math.ceil(capacity/2)

        self.tree = np.zeros(2**self.layer-1, dtype=np.float64)

    def batch_sample(self, batch_size: int):
        # p_sum = self.tree[0]
        # interval = p_sum/batch_size

        # prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, batch_size)

        # idxes = np.zeros(batch_size, dtype=np.int64)
        # for _ in range(self.layer-1):
        #     nodes = self.tree[idxes*2+1]
        #     idxes = np.where(prefixsums<nodes, idxes*2+1, idxes*2+2)
        #     prefixsums = np.where(idxes%2==0, prefixsums-self.tree[idxes-1], prefixsums)
        
        # priorities = self.tree[idxes]
        # idxes -= 2**(self.layer-1)-1

        # # assert np.all(priorities>0), 'idx: {}, priority: {}'.format(idxes, priorities)
        # # assert np.all(idxes>=0) and np.all(idxes<self.capacity)

        # return idxes, priorities
        return sample(self.tree, self.layer, batch_size)

    

    # @jit(void(int64, float32))
    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        # idxes = idxes+2**(self.layer-1)-1
        # self.tree[idxes] = priorities

        # for _ in range(self.layer-1):
        #     idxes = (idxes-1) // 2
        #     idxes = np.unique(idxes)
        #     self.tree[idxes] = self.tree[2*idxes+1] + self.tree[2*idxes+2]


        update(self.tree, self.layer, idxes, priorities)
        
        # check
        assert np.sum(self.tree[-(2**(self.layer-1)):])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])




@ray.remote(num_cpus=1)
class ReplayBuffer:
    def __init__(self, buffer_capacity=config.buffer_capacity, sequence_len=config.block_length,
                alpha=config.priority_exponent, beta=config.importance_sampling_exponent,
                batch_size=config.batch_size, frame_stack=config.frame_stack):

        self.buffer_capacity = buffer_capacity
        self.sequence_len = config.SequenceConfig.learning_steps
        self.num_sequences = buffer_capacity//self.sequence_len
        self.block_len = config.block_length
        self.num_blocks = self.buffer_capacity // self.block_len
        self.seq_pre_block = self.block_len // self.sequence_len

        self.block_ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(self.num_sequences)
        self.alpha = alpha
        self.beta = beta

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

        self.obs_buf = [None for _ in range(self.num_blocks)]
        # self.obs_buf = np.empty((self.num_blocks, self.frame_stack+self.burn_in_steps+self.block_len, 84, 84), dtype=np.uint8)
        self.last_action_buf = [None for _ in range(self.num_blocks)]
        self.hidden_buf = [None for _ in range(self.num_blocks)]
        self.act_buf = [None for _ in range(self.num_blocks)]
        self.rew_buf = [None for _ in range(self.num_blocks)]
        self.gamma_buf = [None for _ in range(self.num_blocks)]
        self.seq_pre_block_buf = np.zeros(self.num_blocks, dtype=np.uint8)
        self.learning_steps = np.zeros((self.num_blocks, self.seq_pre_block), dtype=np.uint8)
        self.burn_in_steps = np.zeros((self.num_blocks, self.seq_pre_block), dtype=np.uint8)
        self.forward_steps = np.zeros((self.num_blocks, self.seq_pre_block), dtype=np.uint8)

        self.obs_idx = np.array([i+j for i in range(config.seq_len) for j in range(config.frame_stack)], dtype=np.long)

        

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

            idxes = np.arange(self.block_ptr*self.seq_pre_block, (self.block_ptr+1)*self.seq_pre_block, dtype=np.int64)

            # self.seq_pre_block_buf[self.seq_ptr] = slot_size

            self.priority_tree.batch_update(idxes, block[6]**self.alpha)

            self.obs_buf[self.block_ptr] = np.copy(block[0])
            self.last_action_buf[self.block_ptr] = np.copy(block[1])
            self.hidden_buf[self.block_ptr] = np.copy(block[2])
            self.act_buf[self.block_ptr] = np.copy(block[3])
            self.rew_buf[self.block_ptr] = np.copy(block[4])
            self.gamma_buf[self.block_ptr] = np.copy(block[5])

            self.seq_pre_block_buf[self.block_ptr] = block[7]

            self.burn_in_steps[self.block_ptr].fill(0)
            self.learning_steps[self.block_ptr].fill(0)
            self.forward_steps[self.block_ptr].fill(0)

            self.burn_in_steps[self.block_ptr, :block[7]] = block[8]
            self.learning_steps[self.block_ptr, :block[7]] = block[9]
            self.forward_steps[self.block_ptr, :block[7]] = block[10]

            self.block_ptr = (self.block_ptr+1) % self.num_blocks

            self.env_steps += np.sum(block[9], dtype=np.int)
            if block[11]:
                self.episode_reward += block[11]
                self.num_episodes += 1

    def sample_batch(self):
        '''sample one batch of training data'''
        batch_obs, batch_last_action, batch_hidden, batch_action, batch_reward, batch_gamma = [], [], [], [], [], []
        batch_burn_in_steps, batch_learning_steps, batch_forward_steps = [], [], []
        
        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(self.batch_size)
            global_idxes = idxes // self.seq_pre_block
            local_idxes = idxes % self.seq_pre_block

            for block_idx, in_block_seq_idx in zip(global_idxes, local_idxes):
                
                assert in_block_seq_idx < self.seq_pre_block_buf[block_idx].item(), 'index is {} but size is {}'.format(in_block_seq_idx, self.seq_pre_block_buf[block_idx])
                
                burn_in_steps = self.burn_in_steps[block_idx, in_block_seq_idx].item()
                learning_steps = self.learning_steps[block_idx, in_block_seq_idx].item()
                forward_steps = self.forward_steps[block_idx, in_block_seq_idx].item()

                start_idx = self.burn_in_steps[block_idx, 0]+np.sum(self.learning_steps[block_idx, :in_block_seq_idx]).item()

                obs = self.obs_buf[block_idx][start_idx-burn_in_steps:start_idx+learning_steps+forward_steps+self.frame_stack-1]
                last_action = self.last_action_buf[block_idx][start_idx-burn_in_steps:start_idx+learning_steps+forward_steps]

                if burn_in_steps + learning_steps + forward_steps < config.seq_len:
                    pad_len = config.seq_len - burn_in_steps - learning_steps - forward_steps
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0)))
                    last_action = np.pad(last_action, ((0, pad_len), (0, 0)))

                # obs = obs[self.obs_idx].reshape(config.seq_len, config.frame_stack, 84, 84)
                hidden = self.hidden_buf[block_idx][in_block_seq_idx]
                action = self.act_buf[block_idx][np.sum(self.learning_steps[block_idx, :in_block_seq_idx]):np.sum(self.learning_steps[block_idx, :in_block_seq_idx+1])]
                reward = self.rew_buf[block_idx][np.sum(self.learning_steps[block_idx, :in_block_seq_idx]):np.sum(self.learning_steps[block_idx, :in_block_seq_idx+1])]
                gamma = self.gamma_buf[block_idx][np.sum(self.learning_steps[block_idx, :in_block_seq_idx]):np.sum(self.learning_steps[block_idx, :in_block_seq_idx+1])]
                
                batch_obs.append(obs)
                batch_last_action.append(last_action)
                batch_hidden.append(hidden)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_gamma.append(gamma)
                batch_burn_in_steps.append(burn_in_steps)
                batch_learning_steps.append(learning_steps)
                batch_forward_steps.append(forward_steps)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            data = [
                torch.from_numpy(np.stack(batch_obs)),
                torch.from_numpy(np.stack(batch_last_action)),
                torch.from_numpy(np.stack(batch_hidden)).transpose(0, 1),

                torch.LongTensor(np.concatenate(batch_action)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_reward)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_gamma)).unsqueeze(1),

                torch.LongTensor(batch_burn_in_steps),
                torch.ByteTensor(batch_learning_steps),
                torch.LongTensor(batch_forward_steps),

                idxes,
                torch.from_numpy(weights.astype(NP_DEFAULT_FLOAT)).unsqueeze(1),
                self.block_ptr,

                self.env_steps
            ]

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int, loss: float):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been replaced by new data in replay buffer during training
            if self.block_ptr > old_ptr:
                # range from [old_ptr, self.seq_ptr)
                mask = (idxes < old_ptr*self.seq_pre_block) | (idxes >= self.block_ptr*self.seq_pre_block)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.block_ptr < old_ptr:
                # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.seq_pre_block) & (idxes >= self.block_ptr*self.seq_pre_block)
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities)**self.alpha)

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
            print(f'episode return: {self.episode_reward/self.num_episodes:.4f}')
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

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer, env_name=config.EnvConfig.env_name, lr=config.lr, eps=config.eps):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_name = env_name
        self.model = Network(creat_env().action_space.n)
        self.model.to(self.device)
        self.model.train()
        self.tar_model = deepcopy(self.model)
        self.tar_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=lr, eps=eps)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.buffer = buffer
        self.counter = 0
        self.done = False
        

        self.data_list = []

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
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
            if len(self.data_list) < 4:
                data = ray.get(self.buffer.sample_batch.remote())
                self.data_list.append(data)
            else:
                time.sleep(0.2)

    def train(self):
        scaler = GradScaler()
        obs_idx = torch.LongTensor([i+j for i in range(config.seq_len) for j in range(config.frame_stack)])
        torch.save((self.model.state_dict(), 0, 0), os.path.join('models', '{}0.pth'.format(self.env_name)))
        while self.counter < config.training_steps:

            while not self.data_list:
                time.sleep(0.5)
                print('empty')
            data = self.data_list.pop(0)

            batch_obs, batch_last_action, batch_hidden, batch_action, batch_n_step_reward, batch_n_step_gamma, burn_in_steps, learning_steps, forward_steps, idxes, weights, old_ptr, env_steps = data
            batch_obs, batch_last_action, batch_hidden = batch_obs.to(self.device), batch_last_action.to(self.device), batch_hidden.to(self.device)
            batch_action, batch_n_step_reward, batch_n_step_gamma = batch_action.to(self.device), batch_n_step_reward.to(self.device), batch_n_step_gamma.to(self.device)
            weights = weights.to(self.device)

            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            with autocast(enabled=config.amp):

                batch_obs = torch.stack([obs[obs_idx] for obs in batch_obs]).view(config.batch_size, config.seq_len, config.frame_stack, 84, 84)
                batch_obs = batch_obs.float()
                batch_obs = batch_obs / 255

                batch_last_action = batch_last_action.float()

                # double q learning
                if config.DQNConfig.double_q_learning:
                    batch_action_ = self.model.caculate_q_(batch_obs, batch_last_action, batch_hidden, burn_in_steps, learning_steps, forward_steps).argmax(1).unsqueeze(1)
                    batch_q_ = self.tar_model.caculate_q_(batch_obs, batch_last_action, batch_hidden, burn_in_steps, learning_steps, forward_steps).gather(1, batch_action_)
                else:
                    pass
                
                target_q = self.value_rescale(batch_n_step_reward + batch_n_step_gamma * self.inverse_value_rescale(batch_q_))
                # target_q = batch_n_step_reward + batch_n_step_gamma * batch_q_

                batch_q = self.model.caculate_q(batch_obs, batch_last_action, batch_hidden, burn_in_steps, learning_steps).gather(1, batch_action)
                
                td_error = self.loss_fn(batch_q, target_q)
                loss = td_error.mean()
                # loss = (weights * td_error).mean()
            
            td_error = (target_q-batch_q).detach().clone().squeeze().abs().cpu().float().numpy()

            priorities = caculate_priorities(td_error, learning_steps.numpy())

            # automatic mixed precision training
            if config.amp:
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm)
                self.optimizer.step()

            self.counter += 1

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr, loss.item())

            # store new weights in shared memory
            if self.counter % 2  == 0:
                self.store_weights()

            # update target net
            if self.counter % config.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())
            
            # save model 
            if self.counter % config.save_interval == 0:
                torch.save((self.model.state_dict(), self.counter, env_steps), os.path.join('models', '{}{}.pth'.format(self.env_name, self.counter//config.save_interval)))

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
    def __init__(self, action_dim, forward_steps=config.forward_steps, frame_stack=config.frame_stack,
                learning_steps=config.SequenceConfig.learning_steps, gamma=config.gamma, network_config=config.NetworkConfig()):
        
        self.action_dim = action_dim
        self.forward_steps = forward_steps
        self.frame_stack = frame_stack
        self.learning_steps = learning_steps
        self.burn_in_steps = config.SequenceConfig.burn_in_steps
        self.curr_burn_in_steps = 0
        self.sub_seq_len = config.block_length
        self.gamma = gamma
        self.recurrent_dim = network_config.recurrent_dim
    
    def __len__(self):
        return self.size
    
    def reset(self, init_obs):
        self.obs_buffer = [init_obs for _ in range(self.frame_stack)]
        self.last_action_buffer = [np.zeros(self.action_dim, dtype=np.bool)]
        self.hidden_buffer = [np.zeros((2, self.recurrent_dim), dtype=NP_DEFAULT_FLOAT)]
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
        self.size += 1
    
    def finish(self, last_qval=None) -> Tuple:
        assert self.size <= config.block_length
        assert len(self.obs_buffer) == self.frame_stack + self.curr_burn_in_steps + self.size, f'{len(self.obs_buffer)} {self.frame_stack+self.curr_burn_in_steps+self.size}'
        assert len(self.last_action_buffer) == self.curr_burn_in_steps + self.size + 1

        num_sequences = math.ceil(self.size/self.learning_steps)

        max_forward_steps = min(self.size, self.forward_steps)
        n_step_gamma = [self.gamma**self.forward_steps for _ in range(self.size-max_forward_steps)]

        if last_qval is not None:
            self.qval_buffer.append(last_qval)
            n_step_gamma.extend([self.gamma**i for i in reversed(range(1, max_forward_steps+1))])
        else:
            self.done = True
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[0]))
            n_step_gamma.extend([0 for _ in range(max_forward_steps)]) # set gamma to 0 so don't need 'done'

        n_step_gamma = np.array(n_step_gamma, dtype=NP_DEFAULT_FLOAT)
        obs_buffer = np.stack(self.obs_buffer)
        last_action = np.stack(self.last_action_buffer)
        hidden = np.stack(self.hidden_buffer[slice(0, self.size, self.learning_steps)])
        assert hidden.shape[0] == num_sequences, f'{hidden.shape} {num_sequences}'
        action_buffer = np.array(self.action_buffer, dtype=np.uint8)
        qval_buffer = np.concatenate(self.qval_buffer).astype(NP_DEFAULT_FLOAT)
        self.sum_reward += np.sum(self.reward_buffer)
        reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps-1)]
        n_step_reward = np.convolve(reward_buffer, 
                                    [self.gamma**(self.forward_steps-1-i) for i in range(self.forward_steps)],
                                    'valid').astype(NP_DEFAULT_FLOAT)


        burn_in_steps = np.array([min(i*self.learning_steps+self.curr_burn_in_steps, self.burn_in_steps) for i in range(num_sequences)], dtype=np.uint8)
        learning_steps = np.array([min(self.learning_steps, self.size-i*self.learning_steps) for i in range(num_sequences)], dtype=np.uint8)
        forward_steps = np.array([min(self.forward_steps, self.size+1-np.sum(learning_steps[:i+1])) for i in range(num_sequences)], dtype=np.uint8)
        assert forward_steps[-1] == 1 and burn_in_steps[0] == self.curr_burn_in_steps

        max_qval = np.max(qval_buffer[max_forward_steps:self.size+1], axis=1)
        max_qval = np.pad(max_qval, (0, max_forward_steps-1), 'edge')
        target_qval = qval_buffer[np.arange(self.size), action_buffer]
        td_errors = np.abs(n_step_reward + n_step_gamma*max_qval - target_qval, dtype=np.float32)
        priorities = np.zeros(config.block_length//config.learning_steps, dtype=np.float32)
        priorities[:num_sequences] = caculate_priorities(td_errors, learning_steps)

        self.obs_buffer = self.obs_buffer[-self.frame_stack-self.burn_in_steps:]
        self.last_action_buffer = self.last_action_buffer[-self.burn_in_steps-1:]
        self.hidden_buffer = self.hidden_buffer[-self.burn_in_steps-1:]
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.qval_buffer.clear()
        self.curr_burn_in_steps = len(self.last_action_buffer)-1
        self.size = 0
        
        return [obs_buffer, last_action, hidden, action_buffer, n_step_reward, n_step_gamma, priorities, num_sequences, burn_in_steps, learning_steps, forward_steps, self.sum_reward if self.done else None]

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, epsilon, learner, buffer):

        self.env = creat_env()
        self.action_dim = self.env.action_space.n
        self.model = Network(self.env.action_space.n)
        self.model.eval()
        self.local_buffer = LocalBuffer(self.action_dim)
        
        self.stacked_obs = torch.empty((1, *config.EnvConfig.obs_shape), dtype=torch.float32)
        self.epsilon = epsilon
        self.learner = learner
        self.replay_buffer = buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0
        self.env_steps = 0

        self.last_action = torch.zeros((1, self.action_dim), dtype=torch.float32)

    def run(self):
        done = False
        self.reset()
        
        while True:
            obs = self.stacked_obs.clone()
            # print(self.last_action)
            action, qval, hidden = self.model.step(obs, self.last_action)

            if random.random() < self.epsilon:
                action = self.env.action_space.sample()

            # apply action in env
            next_obs, reward, done, _ = self.env.step(action)

            self.last_action.fill_(0)
            self.last_action[0, action] = 1

            self.stacked_obs = self.stacked_obs.roll(-1, 1)
            self.stacked_obs[0, -1] = torch.from_numpy(next_obs) / 255

            self.env_steps += 1

            self.local_buffer.add(action, reward, next_obs, qval, hidden)

            if done or self.env_steps == config.max_episode_length:
                block = self.local_buffer.finish()
                if self.epsilon >= 0.02:
                    block[-1] = None
                done = False
                self.reset()
                self.replay_buffer.add.remote(block)

            elif len(self.local_buffer) == config.block_length:
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

    def select_action(self, q_value: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(q_value, 1).item()
    
    def reset(self):
        obs = self.env.reset()
        self.model.reset()
        self.stacked_obs[0, :] = torch.from_numpy(obs) / 255
        self.local_buffer.reset(obs)
        self.last_action.fill_(0)
        self.env_steps = 0

