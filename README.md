# R2D2 (Recurrent Experience Replay in Distributed Reinforcement Learning)
## introduction
An Implementation of [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX) (Kapturowski et al. 2019) in PyTorch.

## Training
First adjust parameter settings in config.py (number of actors, environment name, etc.).

Then run:
```
python3 train.py
```
## Testing
```
python3 test.py
```
## Result
The code was trained and tested in Atari game 'MsPacmac'
 ![image](https://github.com/ZiyuanMa/R2D2/blob/main/images/MsPacmac.jpg)






