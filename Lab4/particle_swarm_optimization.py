import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        delta = upper_bound - lower_bound

        self.position = np.array([random.uniform(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))])
        self.best_position = self.position
        self.best_value = -inf
        self.speed = np.array([random.uniform(-delta[i], delta[i]) for i in range(len(lower_bound))])


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.particles = [Particle(lower_bound, upper_bound) for i in range(hyperparams.num_particles)]
        self.hyperparams = hyperparams

        self.best_iteration_value = -inf
        self.best_iteration_position = None

        self.best_global = -inf
        self.best_position = self.particles[0].position
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.curr = 0

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return self.best_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        return self.best_global

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        particle = self.particles[self.curr]
        rp = random.random()
        rg = random.random()
        particle.speed = self.hyperparams.inertia_weight * particle.speed + self.hyperparams.cognitive_parameter * rp * (particle.best_position - particle.position) + self.hyperparams.social_parameter * rg * (self.best_position - particle.position)
        
        particle.position = particle.position + particle.speed
        return particle.position



    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        if self.curr >= self.hyperparams.num_particles:
            if self.best_iteration_value > self.best_global:
                self.best_global = self.best_iteration_value
                self.best_position = self.best_iteration_position
            self.best_iteration_value = -inf
            self.best_iteration_position = None
            self.curr = 0

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        if value > self.particles[self.curr].best_value:
            self.particles[self.curr].best_value = value
            self.particles[self.curr].best_position = self.particles[self.curr].position
        if value > self.best_iteration_value:
            self.best_iteration_value = value
            self.best_iteration_position = self.particles[self.curr].position
        
        self.curr += 1
        self.advance_generation()

        return

    


