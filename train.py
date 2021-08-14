import random
import time
import torch
import numpy as np
import ray
from worker import Learner, Actor, ReplayBuffer
import config

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(2)

def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (num_actors-1) * alpha
    return base_eps**exponent


def train(num_actors=config.num_actors, log_interval=config.log_interval):
    ray.init()

    buffer = ReplayBuffer.remote()
    learner = Learner.remote(buffer)
    actors = [Actor.remote(get_epsilon(i), learner, buffer) for i in range(num_actors)]

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()):
        time.sleep(log_interval)
        ray.get(buffer.log.remote(log_interval))
        print()

    print('start training')
    learner.run.remote()
    
    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(buffer.log.remote(log_interval))
        print()

if __name__ == '__main__':

    train()

# #%%
# import ray
# import ray.rllib.agents.
# rllib train --env=MsPacman-v0 --run=APEX-DQN --config '{"num_workers": 16, "framework": torch}'
