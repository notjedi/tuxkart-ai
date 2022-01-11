import pystk
import numpy as np

from random import choice
from gym import Env, Wrapper
from sympy import Point3D, Line3D
from gym.spaces import Box, MultiDiscrete


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
        self.node_idx = 0
        self.started = False
        self.observation_shape = (graphicConfig.screen_height, graphicConfig.screen_width, 3)
        self.graphicConfig = graphicConfig
        self.race = pystk.Race(raceConfig)
        self.track = pystk.Track()
        self.state = pystk.WorldState()
        self.currentAction = pystk.Action()
        self.image = np.zeros(self.observation_shape, dtype=np.float32)

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

    def _compute_lines(self, nodes):
        return [Line3D(*node) for node in nodes]

    def _update_node_idx(self):
        dist_down_track = self.playerKart.distance_down_track
        path_dist = self.path_distance[self.node_idx]
        if not (path_dist[0] <= dist_down_track <= path_dist[1]):
            while not (path_dist[0] <= dist_down_track <= path_dist[1]):
                if dist_down_track < path_dist[0]:
                    self.node_idx -= 1
                elif dist_down_track > path_dist[1]:
                    self.node_idx += 1
                path_dist = self.path_distance[self.node_idx]

    def _get_jumping(self) -> bool:
        return self.playerKart.jumping

    def _get_powerup(self):
        powerup = self.playerKart.powerup
        if powerup.type == pystk.Powerup.Type.NOTHING:
            return None
        return powerup

    def _get_position(self) -> int:
        overallDist = sorted([kart.overall_distance for kart in self.state.karts], reverse=True)
        return overallDist.index(self.playerKart.overall_distance) + 1

    def _get_attachment(self):
        attachment = self.playerKart.attachment
        if attachment.type == pystk.Attachment.Type.NOTHING:
            return None
        return attachment

    def _get_finish_time(self) -> int:
        return int(self.playerKart.finish_time)

    def _get_overall_distance(self) -> int:
        return max(0, int(self.playerKart.overall_distance))

    def _get_kart_dist_from_center(self):
        # compute the dist b/w the kart and the center of the track
        # should have called self._update_node_idx() before calling this to avoid errors
        location = self.playerKart.location
        path_node = self.path_nodes[self.node_idx]
        return path_node.distance(Point3D(location)).evalf()

    def _get_is_inside_track(self):
        # should i call this inside step?
        # divide path_width by 2 because it's the width of the current path node
        # and the dist of kart is from the center line
        self._update_node_idx()
        curr_path_width = self.path_width[self.node_idx][0]
        kart_dist = self._get_kart_dist_from_center()
        return kart_dist <= curr_path_width/2

    def _get_velocity(self):
        # returns the magnitude of velocity
        return np.sqrt(np.sum(np.array(self.playerKart.velocity) ** 2))

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
        info["velocity"] = self._get_velocity()
        info["position"] = self._get_position()
        info["attachment"] = self._get_attachment()
        info["finish_time"] = self._get_finish_time()
        info["is_inside_track"] = self._get_is_inside_track()
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
        self.track.update()
        self.path_width = np.array(self.track.path_width)
        self.path_distance = np.array(self.track.path_distance)
        self.path_nodes = np.array(self._compute_lines(self.track.path_nodes))
        self.playerKart = self.state.players[0].kart

    def step(self, action:list):
        if not self.started:
            self.reset()
            self.started = True
        self._update_action(action)
        self.race.step(self.currentAction)
        self.state.update()
        self.track.update()
        self.image = np.array(self.race.render_data[0].image, dtype=np.float32)

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
                                    dtype=np.float32))

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

    def get_env_info(self):
        return self.env.get_env_info()

    def close(self):
        self.env.close()

class STKReward(Wrapper):

    FINISH          = 30
    POSITION        = 5
    DRIFT           = 3
    NITRO           = 3
    COLLECT_POWERUP = 3
    VELOCITY        = 2
    USE_POWERUP     = 1
    JUMP            = -3
    OUT_OF_TRACK    = -3
    RESCUE          = -5

    def __init__(self, env: STKEnv):
        # TODO: add terminal states
        # TODO: use RewardWrapper instead?
        # TODO: sanity check env while using it in model
        # TODO: intelligently handle rewards for attachments
        # TODO: rewards for using powerup - only if it hits other karts
        # TODO: change value of USE_POWERUP when accounted for hitting other karts
        super(STKReward, self).__init__(env)
        self.reward = 0
        self.prevInfo = None

    def _get_reward(self, action, info):

        early_end, reward = False, 0
        if self.prevInfo is None:
            self.prevInfo = info

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
            reward += STKReward.USE_POWERUP

        # finish rewards
        if info["done"]:
            reward += STKReward.FINISH

        # position rewards
        if info["position"] < self.prevInfo["position"]:
            reward += STKReward.POSITION
        elif info["position"] > self.prevInfo["position"]:
            reward -= STKReward.POSITION

        # rewards for velocity
        if info["velocity"] > (self.prevInfo["velocity"] + 1):
            reward += STKReward.VELOCITY

        if not info["is_inside_track"]:
            reward += STKReward.OUT_OF_TRACK
            early_end = True

        # don't go backwards
        reward += (info["overall_distance"] - self.prevInfo["overall_distance"])

        # rewards for collecting powerups
        if info["powerup"] is not None and self.prevInfo["powerup"] is None:
            reward += STKReward.COLLECT_POWERUP

        # misc rewards
        if info["jumping"] and not prevInfo["jumping"]:
            reward += STKReward.JUMP

        self.prevInfo = info
        return early_end, reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        early_end, reward = self._get_reward(action, info)
        # done = early_end or done
        return state, reward, done, info
