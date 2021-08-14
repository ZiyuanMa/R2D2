'''Neural network model'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import *


class Network(nn.Module):
    def __init__(self, action_dim, network_config: NetworkConfig = NetworkConfig(), env_config: EnvConfig = EnvConfig()):
        super().__init__()

        # 84 x 84 input

        self.hidden_dim = network_config.recurrent_dim

        self.action_dim = env_config.action_dim
        self.obs_shape = env_config.obs_shape

        self.max_forward_steps = 5

        self.feature = nn.Sequential(
            nn.Conv2d(env_config.frame_stack, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3136, 1024),
        )

        test_obs = torch.zeros((1, *self.obs_shape))
        self.cnn_out_dim = self.feature(test_obs).size()[1]

        self.recurrent = nn.LSTM(self.cnn_out_dim+self.action_dim, self.hidden_dim, batch_first=True)
        self.hidden_state = (torch.zeros((1, 1, self.hidden_dim)), torch.zeros((1, 1, self.hidden_dim)))

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, obs, last_action, hidden):
        latent = self.feature(obs)

        recurrent_input = torch.cat((latent, last_action), dim=1).unsqueeze(0)

        _, recurrent_output = self.recurrent(recurrent_input, hidden)

        hidden = recurrent_output[0].squeeze(1)

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value.numpy()

    @torch.no_grad()
    def step(self, obs, last_action):

        # assert last_action.size() == (1, self.action_dim), last_action.size()

        latent = self.feature(obs)

        recurrent_input = torch.cat((latent, last_action), dim=1).unsqueeze(0)

        _, self.hidden_state = self.recurrent(recurrent_input, self.hidden_state)

        hidden = self.hidden_state[0].squeeze(1)

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        action = torch.argmax(q_value, 1).item()

        return action, q_value.numpy(), torch.cat(self.hidden_state, dim=0).squeeze(1).numpy()

    def reset(self):
        self.hidden_state = (torch.zeros((1, 1, self.hidden_dim)), torch.zeros((1, 1, self.hidden_dim)))

    @torch.no_grad()
    def caculate_q_(self, obs, last_action, hidden_state, burn_in_steps, learning_steps, forward_steps):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)

        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps + forward_steps

        recurrent_input = torch.cat((latent, last_action), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, batch_first=True, enforce_sorted=False)

        # self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        seq_start_idx = burn_in_steps + self.max_forward_steps
        forward_pad_steps = torch.minimum(self.max_forward_steps - forward_steps, learning_steps)

        hidden = []
        for hidden_seq, start_idx, end_idx, padding_length in zip(recurrent_output, seq_start_idx, seq_len, forward_pad_steps):
            hidden.append(hidden_seq[start_idx:end_idx])
            if padding_length > 0:
                hidden.append(hidden_seq[end_idx-1].repeat(padding_length, 1))

        hidden = torch.cat(hidden)
        # hidden = torch.cat([torch.cat((hidden[start_idx:end_idx], hidden[end_idx-1].repeat(pad_len, 1)))  for hidden, start_idx, end_idx, pad_len in zip(recurrent_output, seq_start_idx, seq_len, forward_pad_steps)])

        assert hidden.size(0) == torch.sum(learning_steps)

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value


    def caculate_q(self, obs, last_action, hidden_state, burn_in_steps, learning_steps):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)

        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps

        recurrent_input = torch.cat((latent, last_action), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, batch_first=True, enforce_sorted=False)

        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        hidden = torch.cat([output[burn_in:burn_in+learning] for output, burn_in, learning in zip(recurrent_output, burn_in_steps, learning_steps)], dim=0)

        adv = self.advantage(hidden)
        val = self.value(hidden)

        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value

