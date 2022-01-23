import pystk

from random import choice


class STK:

    TRACKS = ['abyss', 'black_forest', 'candela_city', 'cocoa_temple', 'cornfield_crossing',
            'fortmagma', 'gran_paradiso_island', 'hacienda', 'lighthouse', 'minigolf', 'olivermath',
            'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 'snowtuxpeak',
            'stk_enterprise', 'volcano_island', 'xr591', 'zengarden']

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
        config.step_size = 0.045
        config.track = track
        config.laps = laps
        config.players[0].team = 0
        config.players[0].kart = kart
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        return config


class Logger():

    def __init__(self, writer):
        self.writer = writer
        self.train_step = 0
        self.eval_step = 0

    def log_train(self, step, actor_loss, critic_loss, entropy_loss, loss):
        self.writer.add_scalar("train/entropy_loss", entropy_loss.item(), self.train_step)
        self.writer.add_scalar("train/policy_loss", actor_loss.item(), self.train_step)
        self.writer.add_scalar("train/value_loss", critic_loss.item(), self.train_step)
        self.writer.add_scalar("train/loss", loss.item(), self.train_step)
        self.writer.flush()
        self.train_step += 1

    def log_eval(self, reward, value, tot_reward, image):
        self.writer.add_scalar('eval/rewards', reward, self.eval_step)
        self.writer.add_scalar('eval/values', value.item(), self.eval_step)
        self.writer.add_scalar('eval/total_rewards', tot_reward, self.eval_step)
        self.writer.add_image('eval/image', image, self.eval_step, dataformats='WH')
        self.writer.flush()
        self.eval_step += 1


def make_env(id: int, quality='hd', race_config_args={}):
    """
    Utility function to create an env.

    :param env_id: (str) the environment ID
    :return: (Callable)
    """

    import gym
    def _init() -> gym.Env:
        from src.env import STKAgent, STKEnv, STKReward, SkipFrame, GrayScaleObservation
        env = STKAgent(STK.get_graphic_config(quality), STK.get_race_config(**race_config_args), id)
        env = STKEnv(env)
        env = STKReward(env)
        env = SkipFrame(env, 2)
        env = GrayScaleObservation(env)
        return env
    return _init


def get_encoder(obs_shape):
    # encodes for channel-last layout
    import numpy as np

    num_info = 4
    idx_array = np.array_split(np.arange(obs_shape[0]), num_info)

    def encode(infos):
        info_image  = np.zeros((len(infos), ) + obs_shape[:-1], dtype=np.float32)

        for i, info in enumerate(infos):
            if info is None:
                continue
            info_image[i, idx_array[0], :] = info["nitro"]
            info_image[i, idx_array[1], :] = info["position"]
            info_image[i, idx_array[2], :] = info["powerup"].value # val of NOTHING = 0
            info_image[i, idx_array[3], :] = info["attachment"].value # val of NOTHING = 9
        return info_image

    return encode


def action_to_dict(action):
    # {acceleration, brake, steer, fire, drift, nitro, rescue}
    # action_space = [2, 2, 3, 2, 2, 2, 2]
    action_dict  = {}
    action_dict ["acceleration"] = action[0]
    action_dict ["brake"] = bool(action[1])
    action_dict ["steer"] = action[2] - 1
    action_dict ["fire "] = bool(action[3])
    action_dict ["drift"] = bool(action[4])
    action_dict ["nitro"] = bool(action[5])
    # action_dict["rescue"] = bool(action[6])
    return action_dict


def calc_params(model):
   return sum([np.prod(param.shape) for param in model.parameters()])
