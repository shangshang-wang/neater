import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gym.wrappers
import slimevolleygym

import matplotlib.pyplot as plt
import multiprocessing
import neat
import numpy as np
import pickle
import random
import time

import utils.visualize as visualize


NUM_CORES = multiprocessing.cpu_count()
env = gym.make('SlimeVolley-v0')


class SlimeGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)

    def __str__(self):
        return f"Slime:\n{super().__str__()}"

class PooledErrorCompute(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes = 3000
        self.generation = 0
        self.num_trials = 16

    def compute_fitness(self, genome, net):
        scores = []
        data = np.zeros(self.num_trials)
        for i in range(self.num_trials):
            observation = env.reset()
            step = 0
            while step < self.test_episodes:
                step += 1
                action = net.activate(observation)
                observation, reward, terminated, info = env.step(action)
                data[i] += reward

                if terminated:
                    break

        score = np.mean(data)
        scores.append(score)
        return score

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        print("Evaluating {0} test episodes".format(self.test_episodes))
        if self.num_workers < 2:
            for genome, net in nets:
                reward = self.compute_fitness(genome, net)
                genome.fitness = reward
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for genome, net in nets:
                    jobs.append(pool.apply_async(self.compute_fitness,
                                                 (genome, net)))

                for job, (genome_id, genome) in zip(jobs, genomes):
                    reward = job.get(timeout=None)
                    genome.fitness = reward

        print("final fitness compute time {0}\n".format(time.time() - t0))


def main():
    config_path = os.path.join("task", 'config_slime')
    config = neat.Config(SlimeGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(100, None))

    ec = PooledErrorCompute(NUM_CORES)
    while 1:
        try:
            gen_best = pop.run(ec.evaluate_genomes, 5)
            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            best_genomes = stats.best_unique_genomes(1)[0]
            best_networks = neat.nn.FeedForwardNetwork.create(best_genomes, config)

            solved = True
            best_scores = []
            for k in range(100):
                observation = env.reset()
                score = 0
                step = 0
                while step < 3000:
                    step += 1
                    best_action = best_networks.activate(observation)
                    observation, reward, terminated, info = env.step(best_action)
                    score += reward
                    # env.render()
                    if terminated:
                        break

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                if avg_score < 0:
                    solved = False
                    break

            if solved:
                print("Solved.")

                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                    visualize.draw_net(config, g, view=False, filename=name + "-net-pruned.gv", prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()


if __name__ == '__main__':
    main()