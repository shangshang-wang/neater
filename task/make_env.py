import gym

from .backprop_gym import BackpropClassifyEnv
from .bipedal_walker import BipedalWalkerHardcore
from .bipedal_walker import BipedalWalker
from .bipedal_walker import BipedalWalker
from .cartpole_swingup import CartPoleSwingUpEnv
from .classify_gym import ClassifyEnv
from .classify_gym import digit_raw
from .classify_gym import mnist_256


def make_env(env_name, seed=42):
    if "Bullet" in env_name:
        import pybullet as p # pip install pybullet
        import pybullet_envs
        import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

    if env_name.startswith("BipedalWalker"):
        if env_name.startswith("BipedalWalkerHardcore"):
            import Box2D
            env = BipedalWalkerHardcore()
        elif env_name.startswith("BipedalWalkerMedium"):
            env = BipedalWalker()
            env.accel = 3
        else:
            env = BipedalWalker()

    elif env_name.startswith("Classify"):
        if env_name.endswith("digits"):
            trainSet, target = digit_raw()
        if env_name.endswith("mnist256"):
            trainSet, target = mnist_256()
        env = ClassifyEnv(trainSet, target)

    elif env_name.startswith("CartPoleSwingUp"):
        env = CartPoleSwingUpEnv()
        if env_name.startswith("CartPoleSwingUp_Hard"):
            env.dt = 0.01
            env.t_limit = 200

    elif env_name.startswith("Backprop"):
        if env_name.endswith("XOR"):
            env = BackpropClassifyEnv(type="XOR", seed=seed)
        elif env_name.endswith("Spiral"):
            env = BackpropClassifyEnv(type="spiral", seed=seed)
        elif env_name.endswith("Circle"):
            env = BackpropClassifyEnv(type="circle", seed=seed)
        elif env_name.endswith("Gaussian"):
            env = BackpropClassifyEnv(type="gaussian", seed=seed)
        else:
            env = BackpropClassifyEnv()
    else:
        env = gym.make(env_name)

    return env