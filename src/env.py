from random import choice
from time import perf_counter, sleep, time

import numpy as np
import pystk
from gym import Env, Wrapper
from gym.spaces import Box, MultiDiscrete
from matplotlib import pyplot as plt

from utils import make_env


class STKAgent():
    """
    SuperTuxKart agent for handling actions and getting state information from the environment.
    The `STKEnv` class passes on the actions to this class for it to handle and gets the current
    state(image) and various other info.

    :param graphicConfig: `pystk.GraphicsConfig` object specifying various graphic configs
    :param raceConfig: `pystk.RaceConfig` object specifying the track and other configs
    """

    def __init__(self, graphicConfig: pystk.GraphicsConfig, raceConfig: pystk.RaceConfig, id=1):

        pystk.init(graphicConfig)
        self.id = id
        self.observation_shape = (graphicConfig.screen_width, graphicConfig.screen_height, 3)
        self.started = False
        self.graphicConfig = graphicConfig
        self.race = pystk.Race(raceConfig)
        self.state = pystk.WorldState()
        self.currentAction = pystk.Action()
        self.image = np.zeros((graphicConfig.screen_width, graphicConfig.screen_height, 3),
                dtype=np.float16)

    def _check_nitro(self) -> bool:
        kartLoc = np.array(self.playerKart.location)
        nitro = [pystk.Item.Type.NITRO_SMALL, pystk.Item.Type.NITRO_BIG]

        for item in self.state.items:
            if item.type in nitro:
                itemLoc = np.array(item.location)
                squared_dist = np.sum((kartLoc - itemLoc) ** 2, axis=0)
                dist = np.sqrt(squared_dist)
                if dist <= 1:
                    return True
        return False

    def _get_jumping(self) -> bool:
        return self.playerKart.jumping

    def _get_powerup(self):
        powerup = self.playerKart.powerup
        if powerup == pystk.Powerup.Type.NOTHING:
            return None
        return powerup

    def _get_position(self) -> int:
        overallDist = sorted([kart.overall_distance for kart in self.state.karts], reverse=True)
        return overallDist.index(self.playerKart.overall_distance) + 1

    def _get_attachment(self):
        attachment = self.playerKart.attachment
        if attachment == pystk.Attachment.Type.NOTHING:
            return None
        return attachment

    def _get_finish_time(self) -> int:
        return int(self.playerKart.finish_time)

    def _get_overall_distance(self) -> int:
        return max(0, int(self.playerKart.overall_distance))

    def _update_action(self, action: list):
        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        # action_space = [2, 2, 3, 2, 2, 2, 2]
        self.currentAction.acceleration = action[0]
        self.currentAction.brake = bool(action[1])
        self.currentAction.steer = action[2] - 1
        self.currentAction.fire = bool(action[3])
        self.currentAction.drift = bool(action[4])
        self.currentAction.nitro = bool(action[5])
        self.currentAction.rescue = bool(action[6])


    def get_env_info(self) -> dict:
        info = {}
        info['id'] = self.id
        info['laps'] = self.race.config.laps
        info['track'] = self.race.config.track
        info['reverse'] = self.race.config.reverse
        info['num_kart'] = self.race.config.num_kart
        info['step_size'] = self.race.config.step_size
        info['difficulty'] = self.race.config.difficulty
        return info

    def get_info(self) -> dict:
        info = {}
        info["done"] = self.done()
        info["nitro"] = self._check_nitro()
        info["jumping"] = self._get_jumping()
        info["powerup"] = self._get_powerup()
        info["position"] = self._get_position()
        info["attachment"] = self._get_attachment()
        info["finish_time"] = self._get_finish_time()
        info["overall_distance"] = self._get_overall_distance()
        return info

    def done(self) -> bool:
        """
        `playerKart.finish_time` > 0 when the kart finishes the race.
        Initially the finish time is < 0.
        """
        return self.playerKart.finish_time > 0

    def reset(self):
        self.race.start()
        self.race.step()
        self.state.update()
        self.playerKart = self.state.players[0].kart

    def step(self, action:list):
        if not self.started:
            self.reset()
            self.started = True
        self._update_action(action)
        self.race.step(self.currentAction)
        self.state.update()
        self.image = np.array(self.race.render_data[0].image)

        info = self.get_info()
        done = self.done()
        return self.image, 0, done, info

    def close(self):
        self.race.stop()
        del self.race
        pystk.clean()

class STKEnv(Env):
    """
    A simple gym compatible STK environment for controlling the kart and interacting with the
    environment. The goal is to place 1st among other karts.

    Observation:
        Image of shape `self.observation_shape`.

    Actions:
        -----------------------------------------------------------------
        |         ACTIONS               |       POSSIBLE VALUES         |
        -----------------------------------------------------------------
        |       Acceleration            |           (0, 1)              |
        |       Brake                   |           (0, 1)              |
        |       Steer                   |         (-1, 0, 1)            |
        |       Fire                    |           (0, 1)              |
        |       Drift                   |           (0, 1)              |
        |       Nitro                   |           (0, 1)              |
        |       Rescue                  |           (0, 1)              |
        -----------------------------------------------------------------

    References:
        1. https://blog.paperspace.com/creating-custom-environments-openai-gym
    """

    def __init__(self, env: STKAgent):
        super(STKEnv, self).__init__()
        self.env = env
        self.observation_space = Box(low=np.zeros(self.env.observation_shape),
                                    high=np.full(self.env.observation_shape, 255,
                                    dtype=np.float16))

        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        self.action_space = MultiDiscrete([2, 2, 3, 2, 2, 2, 2])

    def step(self, action):
        assert self.action_space.contains(action), f'Invalid Action {action}'
        return self.env.step(action)

    def reset(self):
        self.env.reset()

    def render(self, mode: str = 'human'):
        if mode == 'rgb_array':
            return np.transpose(self.env.image, (2, 1, 0))
        return self.env.image

    def close(self):
        self.env.close()

class STKReward(Wrapper):

    FINISH = 10
    POSITION = 5
    COLLECT_POWERUP = 3
    DRIFT = 3
    NITRO = 3
    USE_POWERUP = 1
    JUMP = -3
    RESCUE = -5

    def __init__(self, env: STKEnv):
        # TODO: add terminal states
        super(STKReward, self).__init__(env)
        self.reward = 0
        self.prevInfo = None

    def _get_reward(self, action, info):

        reward = 0
        if self.prevInfo is None:
            self.prevInfo = info

        # TODO: rewards for benificial attachments
        # action rewards
        #  0             1      2      3     4      5      6
        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        # action_space = [2, 2, 3, 2, 2, 2, 2]
        if action[5] and info["nitro"]:
            reward += STKReward.NITRO
        if action[4]:
            reward += STKReward.DRIFT
        if action[6]:
            reward += STKReward.RESCUE
        if action[3] and info["powerup"]:
            # TODO: give only if it damages other karts
            reward += STKReward.USE_POWERUP

        # finish rewards
        if info["done"]:
            reward += STKReward.FINISH

        # position rewards
        if info["position"] < self.prevInfo["position"]:
            reward += STKReward.POSITION
        elif info["position"] > self.prevInfo["position"]:
            reward -= STKReward.POSITION

        # don't go backwards
        reward += (info["overall_distance"] - self.prevInfo["overall_distance"])

        # rewards for collecting powerups
        if info["powerup"] is not None:
            reward += STKReward.COLLECT_POWERUP

        # misc rewards
        if info["jumping"]:
            reward += STKReward.JUMP

        self.prevInfo = info
        return reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += self._get_reward(action, info)
        return state, reward, done, info


def test_env():

    # TODO: use Reward and Action Wrapper
    # TODO: hook up human agent to the env
    env = make_env(0)()
    env.reset()

    action = [1, 0, 0, 0, 1, 0, 0]
    for _ in range(100):
        # action = env.action_space.sample()
        image, reward, _, info = env.step(action)
        plt.imshow(image)
        plt.pause(0.1)
        print(reward)

    env.close()


if __name__ == "__main__":
    test_env()
