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
    def get_race_config(track=None, kart=None, numKarts=5, laps=1, reverse=False, difficulty=1,
            vae=False):
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
        config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL if vae else \
            pystk.PlayerConfig.Controller.PLAYER_CONTROL
        return config


class Logger():

    def __init__(self, writer):
        self.writer = writer
        self.train_step = 0
        self.eval_step = 0
        self.vae_train_step = 0
        self.vae_eval_step = 0

    def log_train(self, step, actor_loss, critic_loss, entropy_loss, loss):
        self.writer.add_scalar("train/entropy_loss", entropy_loss, self.train_step)
        self.writer.add_scalar("train/policy_loss", actor_loss, self.train_step)
        self.writer.add_scalar("train/value_loss", critic_loss, self.train_step)
        self.writer.add_scalar("train/loss", loss, self.train_step)
        self.train_step += 1

    def log_eval(self, reward, value, tot_reward, image):
        self.writer.add_scalar('eval/rewards', reward, self.eval_step)
        self.writer.add_scalar('eval/values', value, self.eval_step)
        self.writer.add_scalar('eval/total_rewards', tot_reward, self.eval_step)
        self.writer.add_image('eval/image', image, self.eval_step, dataformats='HW')
        self.eval_step += 1

    def log_vae_train(self, recon_loss, kl_loss, tot_loss, beta):
        self.writer.add_scalar('train_vae/loss', recon_loss, self.vae_train_step)
        self.writer.add_scalar('train_vae/kl_loss', kl_loss, self.vae_train_step)
        self.writer.add_scalar('train_vae/tot_loss', tot_loss, self.vae_train_step)
        self.writer.add_scalar('train_vae/beta', beta, self.vae_train_step)
        self.vae_train_step += 1

    def log_vae_eval(self, recon_loss, kl_loss, tot_loss, images, recon_images, beta):
        self.writer.add_scalar('eval_vae/loss', recon_loss, self.vae_eval_step)
        self.writer.add_scalar('eval_vae/kl_loss', kl_loss, self.vae_eval_step)
        self.writer.add_scalar('eval_vae/tot_loss', tot_loss, self.vae_eval_step)
        self.writer.add_scalar('eval_vae/beta', beta, self.vae_eval_step)
        self.writer.add_images('eval_vae/images', images, self.vae_eval_step, dataformats='NCHW')
        self.writer.add_images('eval_vae/recon_images', recon_images, self.vae_eval_step,
                dataformats='NCHW')
        self.vae_eval_step += 1



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


def get_encoder():
    # encodes for channel-last layout
    import numpy as np
    num_infos = 4

    def encode(infos):
        encoded_infos = np.zeros((len(infos), num_infos), dtype=np.float32)

        for i, info in enumerate(infos):
            if info is None:
                continue
            encoded_infos[i, 0] = info["nitro"]
            encoded_infos[i, 1] = info["position"]
            encoded_infos[i, 2] = info["powerup"].value # val of NOTHING = 0
            encoded_infos[i, 3] = info["attachment"].value # val of NOTHING = 9
        return encoded_infos

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
