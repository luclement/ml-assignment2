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
populations = [10, 100, 200, 500, 1000, 1500, 2500]
problems = ['flipflop', 'fourpeaks', 'knapsack']
curves = {problem: {population: [] for population in populations} for problem in problems}
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
        df = DiscreteDependencyTree(.1, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

        for population in populations:
            mimic_curve = []
            # Train GA
            print "Problem: %s, Population: %s" % (problem, str(population))
            mimic = MIMIC(population, int(population*0.10), pop)
            fit = ConvergenceTrainer(mimic, int(N+(N-(T+1))), sys.maxint) if (converge) else FixedIterationTrainer(mimic, 1000)
            t0 = time.time()
            fit.train()
            t1 = time.time()
            calls = ef.functionCalls
            mimic_score=ef.value(mimic.getOptimal())
            mimic_curve.append(mimic_score)
            curves[problem][population].append((mimic_score, calls, t1 - t0))
            ef.functionCalls = 0 # reset function calls

avg_scores = {}
avg_function_calls = {}
avg_wall_time = {}

for problem in problems:
    avg_scores[problem] = [(sum(fitness for fitness, _, _ in curves[problem][population]) / seeds) for population in populations]
    avg_function_calls[problem] = [(sum(functionCalls for _, functionCalls, _ in curves[problem][population]) / seeds) for population in populations]
    avg_wall_time[problem] = [(sum(wall_time for _, _, wall_time in curves[problem][population]) / seeds) for population in populations]
print(avg_scores)
print(avg_function_calls)
print(avg_wall_time)

mimic = {
    'scores': avg_scores,
    'calls': avg_function_calls,
    'time': avg_wall_time
}

with open('../../mimic.json', 'w') as fp:
    json.dump(mimic, fp)
