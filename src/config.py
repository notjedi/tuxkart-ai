from random import choice

import torch
import pystk


class STK:

    TRACKS = ['abyss', 'black_forest', 'candela_city', 'cocoa_temple', 'cornfield_crossing', 'fortmagma',
          'gran_paradiso_island', 'hacienda', 'lighthouse', 'minigolf', 'olivermath', 'overworld',
          'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 'snowtuxpeak',
          'stk_enterprise', 'volcano_island', 'xr591', 'zengarden']

    GRAPHICS = { "hd": pystk.GraphicsConfig.hd,
            "sd": pystk.GraphicsConfig.sd,
            "ld": pystk.GraphicsConfig.ld,
            "none": pystk.GraphicsConfig.none }



class STKGraphicConfig:

    WIDTH = 600
    HEIGHT = 400

    def __init__(self, quality="hd"):
        self.config = STK.GRAPHICS[quality]()
        self.config.screen_width = STKGraphicConfig.WIDTH
        self.config.screen_height = STKGraphicConfig.HEIGHT

    def get_config(self):
        return self.config


class STKRaceConfig():

    def __init__(self, track=None, numKarts=5, laps=1, reverse=False, difficulty=1, stepSize=0.07):

        if track is None:
            track = choice(STK.TRACKS)

        self.config = pystk.RaceConfig()
        self.config.difficulty = difficulty
        self.config.step_size = stepSize
        self.config.num_kart = numKarts
        self.config.reverse = reverse
        self.config.track = track
        self.config.laps = laps
        self.config.players[0].team = 0
        self.config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

    def get_config(self):
        return self.config


DEBUG = 0
GAMMA = 0.9
NUM_ENVS = 1
LAMBDA = 0.95
BATCH_SIZE = 1
NUM_FRAMES = 5
BUFFER_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
