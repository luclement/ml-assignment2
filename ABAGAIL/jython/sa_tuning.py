import sys
import os
import time
import json

sys.path.append("../ABAGAIL.jar")

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FlipFlopEvaluationFunction as FlipFlopEvaluationFunction
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import shared.ConvergenceTrainer as ConvergenceTrainer

from array import array



"""
Commandline parameter(s):
   none
"""

N=100
temps = [1, 10, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9, 1E10, 1E11]
problems = ['flipflop', 'fourpeaks', 'knapsack']
curves = {problem: {temp: [] for temp in temps} for problem in problems}
seeds = 5
converge = False

for i in range(seeds):
    fill = [2] * N

    for problem in problems:
        ef = FlipFlopEvaluationFunction()
        if problem == 'knapsack':
            random = Random()
            # The number of items
            NUM_ITEMS = N
            # The number of copies each
            COPIES_EACH = 1
            # The maximum weight for a single element
            MAX_WEIGHT = 1000
            # The maximum volume for a single element
            MAX_VOLUME = 1000
            # create copies
            fill = [COPIES_EACH] * NUM_ITEMS
            copies = array('i', fill)
            # create weights and volumes
            fill = [0] * NUM_ITEMS
            weights = array('d', fill)
            volumes = array('d', fill)
            for i in range(0, NUM_ITEMS):
                weights[i] = random.nextDouble() * MAX_WEIGHT
                volumes[i] = random.nextDouble() * MAX_VOLUME
            MAX_CAPACITY = sum(weights) * 0.35
            ef = KnapsackEvaluationFunction(weights, volumes, MAX_CAPACITY, copies)
            fill = [COPIES_EACH + 1] * NUM_ITEMS
        elif problem == 'fourpeaks':
            ef = FourPeaksEvaluationFunction(N/10)

        ranges = array('i', fill)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)

        for temp in temps:
            print "Problem: %s, Temp: %s" % (problem, str(temp))
            sa_curve = []
            # Train SA
            sa = SimulatedAnnealing(1E11, .95, hcp)
            fit = ConvergenceTrainer(sa, N-1, 100) if (converge) else FixedIterationTrainer(sa, 1000000)
            t0 = time.time()
            fit.train()
            t1 = time.time()
            calls = ef.functionCalls
            sa_score=ef.value(sa.getOptimal())
            sa_curve.append(sa_score)
            curves[problem][temp].append((sa_score, calls, t1 - t0))
            ef.functionCalls = 0 # reset function calls

avg_scores = {}
avg_function_calls = {}
avg_wall_time = {}

for problem in problems:
    avg_scores[problem] = [(sum(fitness for fitness, _, _ in curves[problem][temp]) / seeds) for temp in temps]
    avg_function_calls[problem] = [(sum(functionCalls for _, functionCalls, _ in curves[problem][temp]) / seeds) for temp in temps]
    avg_wall_time[problem] = [(sum(wall_time for _, _, wall_time in curves[problem][temp]) / seeds) for temp in temps]
print(avg_scores)
print(avg_function_calls)
print(avg_wall_time)

sa = {
    'scores': avg_scores,
    'calls': avg_function_calls,
    'time': avg_wall_time
}

with open('../../sa.json', 'w') as fp:
    json.dump(sa, fp)
