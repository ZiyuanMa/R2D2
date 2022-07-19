import random
import torch.multiprocessing as mp
import torch
import numpy as np
from worker import Learner, Actor, ReplayBuffer
from environment import create_env
from model import Network
import config

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)

def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (num_actors-1) * alpha
    return base_eps**exponent


def train(num_actors=config.num_actors, log_interval=config.log_interval):

    model = Network(create_env().action_space.n)
    model.share_memory()
    sample_queue_list = [mp.Queue() for _ in range(num_actors)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(sample_queue_list, batch_queue, priority_queue)
    learner = Learner(batch_queue, priority_queue, model)
    actors = [Actor(get_epsilon(i), model, sample_queue_list[i]) for i in range(num_actors)]

    actor_procs = [mp.Process(target=actor.run) for actor in actors]
    for proc in actor_procs:
        proc.start()

    buffer_proc = mp.Process(target=buffer.run)
    buffer_proc.start()

    learner.run()

    buffer_proc.join()

    for proc in actor_procs:
        proc.terminate()


if __name__ == '__main__':

    train()

