
from dataclasses import dataclass




game_name = 'Boxing'
frameskip = 4
repeat_action_probability = 0
obs_shape = (1, 84, 84)


lr = 1e-4
eps = 1e-3
grad_norm = 40
batch_size = 128
learning_starts = 50000
save_interval = 500
target_net_update_interval = 2000
gamma = 0.997
prio_exponent = 0.9
importance_sampling_exponent = 0.6

training_steps = 100000
buffer_capacity = 500000
max_episode_steps = 27000
actor_update_interval = 400
block_length = 400  # cut one episode to numbers of blocks to improve the buffer space utilization

amp = False # mixed precision training

num_actors = 8
base_eps = 0.4
alpha = 7
log_interval = 20

# sequence setting
burn_in_steps = 40
learning_steps = 10
forward_steps = 5
seq_len = burn_in_steps + learning_steps + forward_steps

# network setting
hidden_dim = 512
cnn_out_dim = 1024

render = False
save_plot = True
test_epsilon = 0.01
