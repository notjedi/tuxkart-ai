import gym
import torch
import pystk
import numpy as np

from random import choice
from sympy import Point3D, Line3D
from torchvision import transforms as T
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
        self.AI = True
        for player in raceConfig.players:
            if player.controller == pystk.PlayerConfig.Controller.PLAYER_CONTROL:
                self.AI = False

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
        return self.playerKart.powerup.type

    def _get_position(self) -> int:
        overallDist = sorted([kart.overall_distance for kart in self.state.karts], reverse=True)
        return overallDist.index(self.playerKart.overall_distance) + 1

    def _get_attachment(self):
        return self.playerKart.attachment.type

    def _get_finish_time(self) -> int:
        return int(self.playerKart.finish_time)

    def _get_overall_distance(self) -> int:
        return max(0, self.playerKart.overall_distance)

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
        self.currentAction.brake = bool(max(0, action[1] - action[0])) # only True when acc is not 1
        self.currentAction.steer = action[2] - 1
        self.currentAction.fire = bool(action[3])
        self.currentAction.drift = bool(action[4])
        self.currentAction.nitro = bool(action[5])
        # self.currentAction.rescue = bool(action[6])
        self.currentAction.rescue = False

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
        if not self.started:
            self.race.start()
            self._update_action([1, 0, 1, 0, 0, 0, 0])
            for _ in range(20):
                self.race.step(self.currentAction)
                self.state.update()
                self.track.update()
                self.image = np.array(self.race.render_data[0].image, dtype=np.float32)
            # compute only if it's player controlled
            self.path_width = np.array(self.track.path_width)
            self.path_distance = np.array(self.track.path_distance)
            self.path_nodes = np.array(self._compute_lines(self.track.path_nodes))
            self.playerKart = self.state.players[0].kart
        self.started = True
        return self.image

    def step(self, action = None):
        if self.AI:
            self.race.step()
            info = None
        else:
            self._update_action(action)
            self.race.step(self.currentAction)
            info = self.get_info()

        self.state.update()
        self.track.update()
        self.image = np.array(self.race.render_data[0].image, dtype=np.float32)
        done = self.done()
        return self.image, 0, done, info

    def close(self):
        self.race.stop()
        del self.race
        pystk.clean()

class STKEnv(gym.Env):
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
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(low=np.zeros(self.env.observation_shape),
                                    high=np.full(self.env.observation_shape, 255,
                                    dtype=np.float32))

        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        self.action_space = MultiDiscrete([2, 2, 3, 2, 2, 2])

    def step(self, action):
        if action:
            assert self.action_space.contains(action), f'Invalid Action {action}'
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode: str = 'human'):
        return self.env.image

    def get_env_info(self):
        return self.env.get_env_info()

    def close(self):
        self.env.close()

class STKReward(gym.Wrapper):

    FINISH          = 30
    POSITION        = 10
    DRIFT           = 3
    NITRO           = 3
    COLLECT_POWERUP = 5
    VELOCITY        = 3
    USE_POWERUP     = 2
    JUMP            = -3
    BACKWARDS       = -10
    OUT_OF_TRACK    = -10
    EARLY_END       = -20

    def __init__(self, env: STKEnv):
        # TODO: should i add fps?
        # TODO: handle rewards for attachments
        # TODO: add reasons on why env is terminated
        # TODO: rewards for using powerup - only if it hits other karts
        # TODO: change value of USE_POWERUP when accounted for hitting other karts
        super(STKReward, self).__init__(env)
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                    high=np.full(self.observation_shape, 255,
                                    dtype=np.float32))
        self.reward = 0
        self.prevInfo = None
        self.total_jumps = 0
        self.no_movement = 0
        self.jump_threshold = 10
        self.out_of_track_count = 0
        self.no_movement_threshold = 30
        self.out_of_track_threshold = 15

    def _get_reward(self, action, info):

        reward = -1
        if self.prevInfo is None:
            self.prevInfo = info

        #  0             1      2      3     4      5      6
        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        # [2,            2,     3,     2,    2,     2,     2]   # action_space
        if action[5] and info["nitro"]:
            reward += STKReward.NITRO
        if action[4] and info["velocity"] > 10:
            reward += STKReward.DRIFT
        # if action[6]:
        #     reward += STKReward.RESCUE
        if action[3] and info["powerup"].value:
            reward += STKReward.USE_POWERUP

        if info["done"]:
            reward += STKReward.FINISH

        if info["position"] < self.prevInfo["position"]:
            reward += STKReward.POSITION
        elif info["position"] > self.prevInfo["position"]:
            reward -= STKReward.POSITION

        if info["velocity"] > (self.prevInfo["velocity"] + 1):
            reward += STKReward.VELOCITY

        if not info["is_inside_track"]:
            reward += STKReward.OUT_OF_TRACK
            self.out_of_track_count += 1
            if self.out_of_track_count > self.out_of_track_threshold:
                info["early_end"] = True
                info["early_end_reason"] = "Outside track"

        # don't go backwards - note that this can also implicitly add -ve rewards
        reward += (info["overall_distance"] - self.prevInfo["overall_distance"])
        if info["overall_distance"] <= self.prevInfo["overall_distance"]:
            reward += STKReward.BACKWARDS
            self.no_movement += 1

        if self.no_movement >= self.no_movement_threshold:
            info["early_end"] = True
            info["early_end_reason"] = "No movement"

        if info["powerup"].value and not self.prevInfo["powerup"].value:
            reward += STKReward.COLLECT_POWERUP

        if info["jumping"] and not self.prevInfo["jumping"]:
            reward += STKReward.JUMP
            self.total_jumps += 1

        if self.total_jumps > self.jump_threshold:
            info["early_end"] = True
            info["early_end_reason"] = "Jump threshold reached"

        if info.get("early_end", False):
            reward += STKReward.EARLY_END

        self.prevInfo = info
        return reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if info is not None:
            reward = self._get_reward(action, info)
            if info.get("early_end", False):
                done = True
                print(f'env_id: {self.env.env.id} - {info.get("early_end_reason", "None")}')
        return state, reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_shape = self.env.observation_shape[:2]
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                    high=np.full(self.observation_shape, 255,
                                    dtype=np.float32))
        self.transform = T.Grayscale()

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        return torch.from_numpy(observation)

    def observation(self, obs):
        return self.transform(self.permute_orientation(obs)).squeeze(dim=0)


class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                    high=np.full(self.observation_shape, 255,
                                    dtype=np.float32))

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
