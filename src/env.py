import pystk
import numpy as np

from gym import Env, Wrapper
from gym.spaces import Box, MultiDiscrete
from time import time, sleep, perf_counter
from matplotlib import pyplot as plt
from random import choice
from config import DEBUG

class STKPlayer():

    tracks = ['abyss', 'black_forest', 'candela_city', 'cocoa_temple', 'cornfield_crossing', 'fortmagma',
              'gran_paradiso_island', 'hacienda', 'lighthouse', 'minigolf', 'olivermath', 'overworld',
              'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 'snowtuxpeak',
              'stk_enterprise', 'volcano_island', 'xr591', 'zengarden']

    graphics = { "hd": pystk.GraphicsConfig.hd,
                "sd": pystk.GraphicsConfig.sd,
                "ld": pystk.GraphicsConfig.ld,
                "none": pystk.GraphicsConfig.none }

    @staticmethod
    def make_graphic_config(quality="hd", width=600, height=400):
        config = STKPlayer.graphics[quality]()
        config.screen_width = width
        config.screen_height = height
        return config

    @staticmethod
    def make_race_config(track, numKarts=5, laps=1, reverse=False, difficulty=1, stepSize=0.07):

        assert 0 <= difficulty <= 2
        config = pystk.RaceConfig()
        config.difficulty = difficulty
        config.step_size = stepSize
        config.num_kart = numKarts
        config.reverse = reverse
        config.track = track
        config.laps = laps
        config.players[0].team = 0
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

        return config

    @property
    def is_race_finished(self):
        return int(self.playerKart.finish_time) != 0

    def __init__(self, graphicConfig, raceConfig):

        pystk.init(graphicConfig)
        self.graphicConfig = graphicConfig
        self.race = pystk.Race(raceConfig)
        self.state = pystk.WorldState()
        self.currentAction = pystk.Action()
        self.image = np.zeros((graphicConfig.screen_width, graphicConfig.screen_height, 3))

    def _get_position(self) -> int:
        overallDist = sorted([kart.overall_distance for kart in self.state.karts], reverse=True)
        return overallDist.index(self.playerKart.overall_distance) + 1

    def _get_finish_time(self) -> int:
        return int(self.playerKart.finish_time)

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

    def _get_overall_distance(self):
        return max(0, int(self.playerKart.overall_distance))

    def _get_attachment(self):
        attachment = self.playerKart.attachment
        if attachment == pystk.Attachment.Type.NOTHING:
            return None
        return attachment

    def _get_powerup(self):
        powerup = self.playerKart.powerup
        if powerup == pystk.Powerup.Type.NOTHING:
            return None
        return powerup

    def get_info(self) -> dict:
        info = {}

        info["done"] = self.done()
        info["position"] = self._get_position()
        info["nitro"] = self._check_nitro()
        info["finish_time"] = self._get_finish_time()
        info["overall_distance"] = self._get_overall_distance()
        info["powerup"] = self._get_powerup()
        info["attachment"] = self._get_attachment()

        return info

    def _update_actions(self, actions: list):
        self.currentAction.acceleration = int('w' in actions)
        self.currentAction.brake = 's' in actions
        self.currentAction.steer = int('d' in actions) - int('a' in actions)
        self.currentAction.fire = ' ' in actions
        self.currentAction.drift = 'm' in actions
        self.currentAction.nitro = 'n' in actions
        self.currentAction.rescue = 'r' in actions

    def done(self):
        return int(self.playerKart.finish_time) != 0

    def reset(self):
        # TODO: where should i start?
        self.race.start()
        self.race.step()
        self.state.update()
        self.playerKart = self.state.players[0].kart

    def step(self, actions:list):
        self._update_actions(actions)
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

    def __init__(self, env: STKPlayer):
        super(STKEnv, self).__init__()
        self.env = env
        self.observation_shape = (600, 400, 3)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                    high=np.full(self.observation_shape, 255,
                                    dtype=np.float16))

        # TODO: call rescue bird action?
        # movement(up/down), steer(left/right), drift, fire, nitro
        self.action_space = MultiDiscrete([2, 2, 1, 1, 1])

    def step(self, actions):
        image, reward, done, info = self.env.step(actions)
        return image, reward, done, info

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

class STKReward(Wrapper):

    FINISH = 10
    POSITION = 5
    POWERUP = 3
    DRIFT = 3
    NITRO = 3
    USE_POWERUP = 1     # TODO: modify
    SPEED = 1           # TODO: implement

    def __init__(self, env: STKEnv):
        super(STKReward, self).__init__(env)
        self.reward = 0
        self.env = env
        self.prevInfo = None

    def _get_reward(self, actions, info):

        if self.prevInfo is None:
            self.prevInfo = info
        reward = 0

        # TODO: rewards for benificial attachments
        # TODO: rewards for jumping
        # TODO: rewards based on prev finish time?
        # TODO: rewards based on speed
        # action rewards
        if 'n' in actions and info["nitro"]:
            reward += STKReward.NITRO
        if 'm' in actions:
            reward += STKReward.DRIFT
        if ' ' in actions and info["powerup"]:
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
            reward += STKReward.POWERUP

        self.prevInfo = info
        return reward

    def step(self, actions):

        state, reward, done, info = self.env.step(actions)
        reward += self._get_reward(actions, info)
        return state, reward, done, info


def create_env(track):
    env = STKPlayer(STKPlayer.make_graphic_config("hd"), STKPlayer.make_race_config(track))
    env = STKEnv(env)
    env = STKReward(env)
    env.reset()
    return env


def test_env():

    # TODO: hook up human agent to the env
    # TODO: create a config (prolly yaml?) file for all params
    track = choice(STKPlayer.tracks)
    env = create_env(track)

    actions = ['m', 'w']
    for _ in range(1000):
        image, reward, _, info = env.step(actions)
        plt.imshow(image)
        plt.pause(0.1)
        print(reward, info)
        print()

    env.close()


if __name__ == "__main__":
    test_env()
