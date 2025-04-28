import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import namedtuple
import numpy as np


Game = namedtuple('Game', [
    'env_name',
    'time_factor',
    'actionSelect',
    'input_size',
    'output_size',
    'layers',
    'i_act',
    'h_act',
    'o_act',
    'weightCap',
    'noise_bias',
    'output_noise',
    'max_episode_length',
    'in_out_labels'])


games = {}

# -- Digit Classification ------------------------------------------------ -- #

# > Scikit learn digits data set
classify = Game(
    env_name='Classify_digits',
    actionSelect='softmax', # all, soft, hard
    input_size=64,
    output_size=10,
    time_factor=0,
    layers=[128,9],
    i_act=np.full(64,1),
    h_act=[1,3,4,5,6,7,8,9,10], # No step function
    o_act=np.full(10,1),
    weightCap = 2.0,
    noise_bias=0.0,
    output_noise=[False, False, False],
    max_episode_length = 0,
    in_out_labels = []
)
L = [list(range(1, classify.input_size)), \
     list(range(0, classify.output_size))]
label = [item for sublist in L for item in sublist]
classify = classify._replace(in_out_labels=label)
games['digits'] = classify


# > MNIST [28x28] data set
mnist784 = classify._replace( \
    env_name='Classify_mnist784', input_size=784, i_act=np.full(784,1))
L = [list(range(1, mnist784.input_size)), \
     list(range(0, mnist784.output_size))]
label = [item for sublist in L for item in sublist]
mnist784 = mnist784._replace(in_out_labels=label)
games['mnist784'] = mnist784

# > MNIST [16x16] data set
mnist256 = classify._replace( \
    env_name='Classify_mnist256', input_size=256, i_act=np.full(256,1))
L = [list(range(1, mnist256.input_size)), \
     list(range(0, mnist256.output_size))]
label = [item for sublist in L for item in sublist]
mnist256 = mnist256._replace(in_out_labels=label)
games['mnist256'] = mnist256


# -- Cart-pole Swingup --------------------------------------------------- -- #

# > Slower reaction speed
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
                        actionSelect='all', # all, soft, hard
                        input_size=5,
                        output_size=1,
                        time_factor=0,
                        layers=[5, 5],
                        i_act=np.full(5,1),
                        h_act=[1,2,3,4,5,6,7,8,9,10],
                        o_act=np.full(1,1),
                        weightCap = 2.0,
                        noise_bias=0.0,
                        output_noise=[False, False, False],
                        max_episode_length = 200,
                        in_out_labels = ['x','x_dot','cos(theta)','sin(theta)','theta_dot',
                                         'force']
                        )
games['swingup_hard'] = cartpole_swingup

# > Normal reaction speed
cartpole_swingup = cartpole_swingup._replace( \
    env_name='CartPoleSwingUp', max_episode_length=1000)
games['swingup'] = cartpole_swingup


# -- Bipedal Walker ------------------------------------------------------ -- #

# > Flat terrain
biped = Game(env_name='BipedalWalker-v2',
             actionSelect='all', # all, soft, hard
             input_size=24,
             output_size=4,
             time_factor=0,
             layers=[40, 40],
             i_act=np.full(24,1),
             h_act=[1,2,3,4,5,6,7,8,9,10],
             o_act=np.full(4,1),
             weightCap = 2.0,
             noise_bias=0.0,
             output_noise=[False, False, False],
             max_episode_length = 400,
             in_out_labels = [
                 'hull_angle','hull_vel_angle','vel_x','vel_y',
                 'hip1_angle','hip1_speed','knee1_angle','knee1_speed','leg1_contact',
                 'hip2_angle','hip2_speed','knee2_angle','knee2_speed','leg2_contact',
                 'lidar_0','lidar_1','lidar_2','lidar_3','lidar_4',
                 'lidar_5','lidar_6','lidar_7','lidar_8','lidar_9',
                 'hip_1','knee_1','hip_2','knee_2']
             )
games['biped'] = biped

# > Hilly Terrain
bipedmed = biped._replace(env_name='BipedalWalkerMedium-v2')
games['bipedmedium'] = bipedmed

# > Obstacles, hills, and pits
bipedhard = biped._replace(env_name='BipedalWalkerHardcore-v2')
games['bipedhard'] = bipedhard


# -- Bullet -------------------------------------------------------------- -- #

# > Quadruped ant
bullet_ant = Game(env_name='AntBulletEnv-v0',
                  actionSelect='all', # all, soft, hard
                  input_size=28,
                  output_size=8,
                  layers=[64, 32],
                  time_factor=1000,
                  i_act=np.full(28,1),
                  h_act=[1,2,3,4,5,6,7,8,9,10],
                  o_act=np.full(8,1),
                  weightCap = 2.0,
                  noise_bias=0.0,
                  output_noise=[False, False, True],
                  max_episode_length = 1000,
                  in_out_labels = []
                  )
games['bullet_ant'] = bullet_ant


# -- slimevolleygym ------------------------------------------------------ -- #
slime_volley = Game(env_name='SlimeVolley-v0',
                    actionSelect='all', # all, soft, hard
                    input_size=12,
                    output_size=3,
                    layers=[64, 32],
                    time_factor=0,
                    i_act=np.full(12,1),
                    h_act=[1,2,3,4,5,6,7,8,9,10],
                    o_act=np.full(3,5),
                    weightCap = 2.0,
                    noise_bias=0.0,
                    output_noise=[False, False, False],
                    max_episode_length = 3000,
                    in_out_labels = ["agent1_x","agent1_y","agent1_x_vel","agent1_y_vel", \
                                     "ball_x","ball_y","ball_x_vel","ball_y_vel", \
                                     "agent2_x","agent2_y","agent2_x_vel","agent2_y_vel", \
                                     "forward", "jump", "backward"]
                    )
games['slimevolley'] = slime_volley


# -- backprop classification ----------------------------------------------------- -- #
backprop_cls = Game(env_name='BackpropClassify',
                    actionSelect='sigmoid', # all, soft, hard
                    input_size=2,
                    output_size=1,
                    time_factor=0,
                    layers=[20, 20],
                    i_act=np.full(2,1),
                    h_act=[1,3,4,5,6,7,8,9,11],
                    o_act=np.full(1,1),
                    weightCap = 50.0,
                    noise_bias=0.0,
                    output_noise=[False, False, False],
                    max_episode_length = 0,
                    in_out_labels = ['x', 'y', 'prob']
                    )
games['backprop_XOR'] = backprop_cls._replace(env_name='BackpropXOR')
games['backprop_spiral'] = backprop_cls._replace(env_name='BackpropSpiral')
games['backprop_gaussian'] = backprop_cls._replace(env_name='BackpropGaussian')
games['backprop_circle'] = backprop_cls._replace(env_name='BackpropCircle')
