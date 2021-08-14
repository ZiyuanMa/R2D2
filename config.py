from dataclasses import dataclass
from typing import Tuple
import gym

frame_stack = 4

@dataclass()
class EnvConfig:
    env_name: str = 'Boxing'
    env_type: str = 'Deterministic-v4'
    frame_stack: int = frame_stack
    obs_shape: tuple = (frame_stack, 84, 84)
    def __init__(self):
        self.action_dim = gym.make(self.env_name+self.env_type).action_space.n

#################### worker.py ####################
lr = 1e-4
eps = 1e-3
grad_norm=40
batch_size = 128
learning_starts = 100000
save_interval = 500
target_network_update_freq = 2000
gamma = 0.997
priority_exponent = 0.9
importance_sampling_exponent = 0.6

training_steps = 100000
buffer_capacity = 500000
max_episode_length = 27000
actor_update_interval = 400
block_length = 400  # cut one episode to sequences to improve the buffer space utilization
class BufferConfig:
    buffer_capacity = buffer_capacity

amp = False

#################### train.py ####################
num_actors = 16
base_eps = 0.4
alpha = 7
log_interval = 20


burn_in_steps = 40
learning_steps = 20
forward_steps = 5
seq_len = burn_in_steps + learning_steps + forward_steps

@dataclass(frozen=True)
class SequenceConfig:
    burn_in_steps: int = burn_in_steps
    learning_steps: int = learning_steps
    forward_steps: int = forward_steps

#################### test.py ####################
render = False
save_plot = True
test_epsilon = 0.01

@dataclass(frozen=True)
class NetworkConfig:
    ''''''
    recurrent_dim: int = 512

@dataclass()
class DQNConfig:
    double_q_learning: bool = True
