import pystk
import numpy as np

from time import time, sleep, perf_counter
from matplotlib import pyplot as plt
from random import choice
from config import DEBUG

class STKPlayer():

    tracks = ['abyss', 'black_forest', 'candela_city', 'cocoa_temple', 'cornfield_crossing', 'fortmagma',
              'gran_paradiso_island', 'hacienda', 'lighthouse', 'mines', 'minigolf', 'olivermath',
              'overworld', 'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 'snowtuxpeak',
              'stk_enterprise', 'tutorial', 'volcano_island', 'xr591', 'zengarden']

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
    def make_race_config(track, numKarts=2, laps=1, reverse=False, difficulty=1, stepSize=0.07):

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

        for _ in range(1, numKarts):
            config.players.append(
                    pystk.PlayerConfig('', pystk.PlayerConfig.Controller.AI_CONTROL, 1))

        return config

    @property
    def is_race_finished(self):
        return int(self.state.players[0].kart.finish_time) != 0

    def __init__(self, graphicConfig, raceConfig):
        pystk.init(graphicConfig)
        self.maxTime = 200
        self.race = pystk.Race(raceConfig)
        self.state = pystk.WorldState()
        self.buffer = []
        self.rewards = []

    # TODO: parallelize
    def play(self):

        start = perf_counter()
        stepTimer = time()
        reward = Reward()
        n = 0

        self.race.start()
        self.race.step()
        self.state.update()
        self.buffer.append(self.race.render_data[0].image)
        self.rewards.append(0)

        # max 20000 frames per player
        # TODO: check if 20000 is enough for a game

        while not self.is_race_finished and n < 20000:
            if perf_counter() - start > self.maxTime:
                break

            # TOOD: get action from NN
            # self.race.step(uis[0].current_action)
            self.race.step()
            self.state.update()

            self.buffer.append(self.race.render_data[0].image)
            self.rewards.append(reward.get_reward(self.buffer[-2], self.buffer[-1]))

            if DEBUG:
                print(len(self.buffer))
                print(self.buffer[-1].shape)
                plt.imshow(self.buffer[-1])
                plt.pause(0.01)

            n += 1
            delta_d = n * self.race.config.step_size - (time() - stepTimer)
            if delta_d > 0:
                sleep(delta_d)

    def clean(self):
        self.race.stop()
        del self.race
        pystk.clean()


class Reward():

    FINISH_POSITION = 10
    POSITION = 5
    POWERUPS = 3
    DRIFT = 3
    NITRO = 3
    TOTAL_DISTANCE = 1
    USE_POWERUP = 1     # only if it has been used correctly, btw should i change this to 2?
    SPEED = 1
    BRAKE = -1          # should i do this?

    def __init__(self):
        pass

    def get_reward(self, prevKartState, kartState):
        # TODO: what is the max speed for a kart?
        pass


def main():

    track = choice(STKPlayer.tracks)
    player = STKPlayer(STKPlayer.make_graphic_config("hd"), STKPlayer.make_race_config(track))
    player.play()
    player.clean()


if __name__ == "__main__":
    main()
