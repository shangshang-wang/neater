import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functools import partial
import gc
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import device_get, device_put
from jax.lax import stop_gradient
import numpy as np
import random
import sys

from task.make_env import make_env
from pretty_neat import *


class GymTask():
    def __init__(self, game,
                 param_only=False, num_fitness_sample=1, seed=42):

        self.seed = seed
        self.num_fitness_sample = num_fitness_sample
        self.need_closed = game.env_name.startswith("CartPoleSwingUp")

        self.nInput   = game.input_size
        self.nOutput  = game.output_size
        self.actRange = game.h_act
        self.absWCap  = game.weightCap
        self.layers   = game.layers
        self.num_epoch = game.max_episode_length
        self.actSelect = game.actionSelect
        self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]

        if not param_only:
            self.env = make_env(game.env_name)


    def get_individual_fitness(self, weight_vec, act_vec, hyp=None, view=False, backprop_eval=False):
        """
        Args:
          weight_vec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          act_vec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)
        Returns:
          fitness - (float)    - mean reward over all trials
        """
        backprop = hyp['backprop'] if hyp and 'backprop' in hyp else False

        if not backprop:
            weight_vec[np.isnan(weight_vec)] = 0
            reward = [ self.eval_individual(weight_vec, act_vec, view=view, seed=self.seed+i) for i in range(self.num_fitness_sample) ]
            fitness = np.mean(reward)
            return fitness
        else:
            grad_mask = np.where(np.isnan(weight_vec), 0, 1)
            num_conn = np.sum(~np.isnan(weight_vec))
            weight_vec = np.where(np.isnan(weight_vec), 0, weight_vec)
            if not backprop_eval:
                init_loss = self.cross_entropy_loss(weight_vec, act_vec)
                very_init_loss = init_loss
                weight_vec_ori = weight_vec.copy()
                weight_vec_prev = weight_vec.copy()
                for i in range(self.num_fitness_sample):
                    final_error, weight_vec = self.eval_individual(weight_vec, act_vec,
                                                                   view=view, seed=self.seed+i, hyp=hyp,
                                                                   backprop_eval=backprop_eval, grad_mask=grad_mask, first_epoch=i==0)
                    if final_error > init_loss:
                        weight_vec = weight_vec_prev
                        break
                    else:
                        init_loss = final_error
                        weight_vec_prev = weight_vec.copy()
                loss = self.cross_entropy_loss(weight_vec, act_vec)
                if loss > init_loss:
                    loss = init_loss
                    weight_vec = weight_vec_prev
                if loss > very_init_loss:
                    loss = very_init_loss
                    weight_vec = weight_vec_ori
                conn_penalty = hyp['connPenalty'] if 'connPenalty' in hyp else 0.03
                reward = -loss * (1 + conn_penalty * np.sqrt(num_conn))
                return reward, weight_vec
            else:
                reward = np.empty(self.num_fitness_sample)
                for i in range(self.num_fitness_sample):
                    reward[i] = self.eval_individual(weight_vec, act_vec, view=view, seed=self.seed+i, hyp=hyp, backprop_eval=backprop_eval, num_conn=num_conn)
                return np.mean(reward)


    def eval_individual(self, weight_vec, act_vec,
                        view=False, hyp=None, seed=42, backprop_eval=False, grad_mask=None, num_conn=None, first_epoch=False):
        """Evaluate individual on task
        Args:
          weight_vec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          act_vec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)
        Returns:
          fitness - (float)    - reward earned in trial
        """
        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        backprop = hyp['backprop'] if hyp and 'backprop' in hyp else False
        if not backprop:
            state = self.env.reset()
            self.env.t = 0
            annOut = act(weight_vec, act_vec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)

            state, reward, done, info = self.env.step(action)

            if self.num_epoch == 0:
                if view:
                    if self.need_closed:
                        self.env.render(close=done)
                    else:
                        self.env.render()
                return reward
            else:
                totalReward = reward

            for tStep in range(self.num_epoch):
                annOut = act(weight_vec, act_vec, self.nInput, self.nOutput, state)
                action = selectAct(annOut,self.actSelect)
                state, reward, done, info = self.env.step(action)
                totalReward += reward
                if view:
                    if self.need_closed:
                        self.env.render(close=done)
                    else:
                        self.env.render()
                if done:
                    break
            return totalReward
        else:
            if backprop_eval:
                conn_penalty = hyp['connPenalty'] if 'connPenalty' in hyp else 0.03
                loss = self.cross_entropy_loss(weight_vec, act_vec)
                totalReward = -loss * (1 + conn_penalty * np.sqrt(num_conn))
                return totalReward

            else:
                self.env.batch = hyp['batch_size'] if 'batch_size' in hyp else 10
                self.env.t = 0
                state = self.env.reset()
                y = self.env.get_labels()

                if jnp.ndim(weight_vec) < 2:
                    nNodes = int(jnp.sqrt(jnp.shape(weight_vec)[0]))
                else:
                    nNodes = int(jnp.shape(weight_vec)[0])

                def forward(weight_vec, act_vec, input, output, state, y, actSelect, backprop, nNodes, grad_mask):
                    annOut = act(weight_vec, act_vec, input, output, state, backprop, nNodes, grad_mask)
                    action = selectAct(annOut, actSelect, backprop)
                    action = jnp.clip(action, 1e-8, 1 - 1e-8)
                    loss = -jnp.mean(y * jnp.log(action) + (1 - y) * jnp.log(1 - action))
                    return loss

                loss = partial(forward, act_vec=act_vec, input=self.nInput, output=self.nOutput, actSelect=self.actSelect, backprop=backprop, nNodes=nNodes, grad_mask=grad_mask)
                loss = jit(loss)

                done = False
                step_size = hyp['step_size'] if 'step_size' in hyp else 0.01
                grad_clip = hyp['ann_absWCap'] / 10.0
                weight_decay = hyp['weight_decay'] if 'weight_decay' in hyp else 0.001
                alpha = hyp['alpha'] if 'alpha' in hyp else 0.99
                self.avg_vel = 0 if first_epoch else self.avg_vel
                eps = 1e-8
                while not done:
                    grads = grad(loss)(weight_vec, state=state, y=y)
                    grads = jnp.where(jnp.isnan(grads), 0, grads)
                    self.avg_vel = alpha * self.avg_vel + (1 - alpha) * jnp.square(grads)
                    grads = jnp.clip(grads, -grad_clip, grad_clip)
                    weight_vec = weight_vec - step_size * (grads / (jnp.sqrt(jnp.maximum(jnp.square(self.avg_vel), eps)))) - weight_decay * weight_vec
                    weight_vec = jnp.clip(weight_vec, -hyp['ann_absWCap'], hyp['ann_absWCap'])
                    state, _, done, _ = self.env.step(None)
                    y = self.env.get_labels()

                    if done:
                        state = self.env.trainSet
                        y = self.env.target
                        weight_vec_np = device_get(weight_vec).copy()
                        loss = self.cross_entropy_loss(weight_vec_np, act_vec)
                        break
                jax.clear_caches()
                return loss, weight_vec_np


    def cross_entropy_loss(self, weight_vec, act_vec):
        '''Compute cross entropy loss of the network
        Args:
          weight_vec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          act_vec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)
        Returns:
          loss - (float)    - loss got in trial
        '''
        state = self.env.trainSet
        y = self.env.target
        annOut = act(weight_vec, act_vec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)
        pred = np.where(action > 0.5, 1, 0)
        assert pred.shape == y.shape, "Prediction and target shape mismatch"
        # loss = np.mean(np.abs(pred - y))
        action = np.clip(action, 1e-8, 1 - 1e-8)
        loss = -np.mean(y * np.log(action) + (1 - y) * np.log(1 - action))
        return loss