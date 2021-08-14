# R2D2 (Recurrent Experience Replay in Distributed Reinforcement Learning)
## introduction
An Implementation of [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX) (Horgan et al. 2018) DQN in PyTorch and Ray.

This implementation is for multi-core single machine, works for openai gym environment.

## How to train
First go to config.py to adjust parameter settings if you want.

Then run:
```
python3 train.py
```
## How to test
```
python3 test.py
```
you can also render the test result or plot the result.






