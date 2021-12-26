import pystk
import numpy as np

from gym import Env, Wrapper
from gym.spaces import Box, MultiDiscrete
from time import time, sleep, perf_counter
from matplotlib import pyplot as plt
from random import choice
from config import STK, STKGraphicConfig, STKRaceConfig

class STKAgent():

    def __init__(self, graphicConfig, raceConfig):

        pystk.init(graphicConfig)
        self.graphicConfig = graphicConfig
        self.race = pystk.Race(raceConfig)
        self.state = pystk.WorldState()
        self.currentAction = pystk.Action()
        self.image = np.zeros((graphicConfig.screen_width, graphicConfig.screen_height, 3))

    @property
    def is_race_finished(self) -> bool:
        return int(self.playerKart.finish_time) != 0

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

    def _update_action(self, actions: list):
        self.currentAction.acceleration = int('w' in action)
        self.currentAction.brake = 's' in action
        self.currentAction.steer = int('d' in action) - int('a' in actions)
        self.currentAction.fire = ' ' in action
        self.currentAction.drift = 'm' in action
        self.currentAction.nitro = 'n' in action
        self.currentAction.rescue = 'r' in action

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
        return int(self.playerKart.finish_time) != 0

    def reset(self):
        # TODO: where should i start?
        self.race.start()
        self.race.step()
        self.state.update()
        self.playerKart = self.state.players[0].kart

    def step(self, action:list):
        self._update_action(actions)
        self.race.step(self.currentAction)
        self.state.update()
        self.image = self.race.render_data[0].image

        info = self.get_info()
        done = self.done()
        return self.image, 0, done, info

    def close(self):
        self.race.stop()
        del self.race
        pystk.clean()

class STKEnv(Env):

    def __init__(self, env: STKAgent):
        super(STKEnv, self).__init__()
        self.env = env
        self.observation_shape = (600, 400, 3)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                    high=np.full(self.observation_shape, 255,
                                    dtype=np.float16))

        # {'acceleration': 0.0, 'brake': False, 'steer': 0.0, 'fire': False, 'drift': False,
        # 'nitro': False, 'rescue': False}
        self.action_space = MultiDiscrete([2, 2, 3, 2, 2, 2, 2])

    def step(self, action):
        assert self.action_space.contains(action), f'Invalid Action {action}'
        image, reward, done, info = self.env.step(action)
        return image, reward, done, info

    def reset(self):
        self.env.reset()

    def render(self):
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
        super(STKReward, self).__init__(env)
        self.reward = 0
        self.prevInfo = None

    def _get_reward(self, action, info):

        reward = 0
        if self.prevInfo is None:
            self.prevInfo = info

        # TODO: rewards for benificial attachments
        # action rewards
        if 'n' in action and info["nitro"]:
            reward += STKReward.NITRO
        if 'm' in action:
            reward += STKReward.DRIFT
        if 'r' in action:
            reward += STKReward.RESCUE
        if ' ' in action and info["powerup"]:
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


def create_env(track):
    env = STKAgent(STKGraphicConfig(), STKRaceConfig(track))
    env = STKEnv(env)
    env = STKReward(env)
    env.reset()
    return env


def test_env():

    # TODO: use Reward and Action Wrapper
    # TODO: hook up human agent to the env
    track = choice(STK.TRACKS)
    env = create_env(track)
    print(env.action_space.sample())

    action = ['m', 'w']
    for _ in range(1000):
        image, reward, _, info = env.step(action)
        plt.imshow(image)
        plt.pause(0.1)
        print(reward, info)
        print()

    env.close()


if __name__ == "__main__":
    test_env()
