'''open.ai gym environment wrapper.
reference: https://github.com/openai/gym/tree/master/gym/wrappers'''
from collections import deque
import gym
import numpy as np
import cv2
import config

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """start the game with no-op actions to provide random starting positions
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """cilp reward in range [-1, 1]."""
        return np.clip(reward, -1, 1)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        """
        super().__init__(env)
        self._width = width
        self._height = height

        original_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),
            dtype=np.uint8,
        )

        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):

        frame = obs

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )

        obs = frame

        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,)+shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames))


def creat_env(env_config=config.EnvConfig, noop_start=True, clip_rewards=False, frame_stack=False, num_stack_frames=config.frame_stack):

    env = gym.make(env_config.env_name+env_config.env_type)

    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, num_stack_frames)
    if noop_start:
        env = NoopResetEnv(env)

    return env


#%%
nums = [0,1,2]
results = [[]]

for i in nums:
    for result in results.copy():
        result = result.copy()
        result.append(i)
        results.append(result)
        

# %%
