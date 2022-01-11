import pystk

from random import choice


class STK:

    TRACKS = ['abyss', 'black_forest', 'candela_city', 'cocoa_temple', 'cornfield_crossing',
            'fortmagma', 'gran_paradiso_island', 'hacienda', 'lighthouse', 'minigolf', 'olivermath',
            'overworld', 'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain',
            'snowtuxpeak', 'stk_enterprise', 'volcano_island', 'xr591', 'zengarden']

    KARTS = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi',
            'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
            'wilber', 'xue']

    GRAPHICS = { "hd": pystk.GraphicsConfig.hd,
            "sd": pystk.GraphicsConfig.sd,
            "ld": pystk.GraphicsConfig.ld,
            "none": pystk.GraphicsConfig.none }

    WIDTH = 600
    HEIGHT = 400

    @staticmethod
    def get_graphic_config(quality='hd'):
        config = STK.GRAPHICS[quality]()
        config.screen_width = STK.WIDTH
        config.screen_height = STK.HEIGHT
        return config

    @staticmethod
    def get_race_config(track=None, kart=None, numKarts=5, laps=1, reverse=False, difficulty=1):
        if track is None:
            track = choice(STK.TRACKS)
        if kart is None:
            kart = choice(STK.KARTS)

        config = pystk.RaceConfig()
        config.difficulty = difficulty
        config.num_kart = numKarts
        config.reverse = reverse
        config.track = track
        config.laps = laps
        config.players[0].team = 0
        config.players[0].kart = kart
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        return config


def make_env(id: int, quality='hd', race_config_args={}):
    """
    Utility function to create an env.

    :param env_id: (str) the environment ID
    :return: (Callable)
    """

    import gym
    def _init() -> gym.Env:
        from src.env import STKAgent, STKEnv, STKReward
        env = STKAgent(STK.get_graphic_config(quality), STK.get_race_config(**race_config_args), id)
        env = STKEnv(env)
        env = STKReward(env)
        return env
    return _init


def calc_params(model):
   return sum([np.prod(param.shape) for param in model.parameters()])
