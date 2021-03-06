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
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
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

runs=[10, 20, 40, 60, 80, 100]
algo_list = ['RHC', 'SA', 'GA', 'MIMIC']
curves = {algo: {size: [] for size in runs} for algo in algo_list}
seeds = 10
converge = False

for i in range(seeds):
    for N in runs:
        rhc_curve=[]
        sa_curve=[]
        ga_curve=[]
        mimic_curve=[]

        T=N/10
        fill = [2] * N
        ranges = array('i', fill)

        ef = FourPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

        print("N: %d, Max Score: %d" % (N, int(N+(N-(T+1)))))
        # Train RHC
        rhc = RandomizedHillClimbing(hcp)
        fit = ConvergenceTrainer(rhc, int(N+(N-(T+1))), sys.maxint) if (converge) else FixedIterationTrainer(rhc, 1500000)
        t0 = time.time()
        fit.train()
        t1 = time.time()
        calls = ef.functionCalls
        rhc_score=ef.value(rhc.getOptimal())
        print "RHC: " + str(rhc_score)
        rhc_curve.append(rhc_score)
        curves['RHC'][N].append((rhc_score, calls, t1 - t0))
        ef.functionCalls = 0 # reset function calls

        # Train SA
        sa = SimulatedAnnealing(1E11, .95, hcp)
        fit = ConvergenceTrainer(sa, int(N+(N-(T+1))), sys.maxint) if (converge) else FixedIterationTrainer(sa, 1500000)
        t0 = time.time()
        fit.train()
        t1 = time.time()
        calls = ef.functionCalls
        sa_score=ef.value(sa.getOptimal())
        print "SA: " + str(sa_score)
        sa_curve.append(sa_score)
        curves['SA'][N].append((sa_score, calls, t1 - t0))
        ef.functionCalls = 0 # reset function calls

        population = 1500
        # Train GA
        ga = StandardGeneticAlgorithm(population, int(population*0.9), int(population*0.15), gap)
        fit = ConvergenceTrainer(ga, int(N+(N-(T+1))), sys.maxint) if (converge) else FixedIterationTrainer(ga, 1095)
        t0 = time.time()
        fit.train()
        t1 = time.time()
        calls = ef.functionCalls
        ga_score=ef.value(ga.getOptimal())
        print "GA: " + str(ga_score)
        ga_curve.append(ga_score)
        curves['GA'][N].append((ga_score, calls, t1 - t0))
        ef.functionCalls = 0 # reset function calls

        population = 1500
        # Train MIMIC
        mimic = MIMIC(population, int(population*0.01), pop)
        fit = ConvergenceTrainer(mimic, int(N+(N-(T+1))), sys.maxint) if (converge) else FixedIterationTrainer(mimic, 1000)
        t0 = time.time()
        fit.train()
        t1 = time.time()
        calls = ef.functionCalls
        mimic_score=ef.value(mimic.getOptimal())
        print "MIMIC: " + str(mimic_score)
        mimic_curve.append(mimic_score)
        curves['MIMIC'][N].append((mimic_score, calls, t1 - t0))
        ef.functionCalls = 0 # reset function calls

avg_scores = {}
avg_function_calls = {}
avg_wall_time = {}

for algo in algo_list:
    avg_scores[algo] = [(sum(fitness for fitness, _, _ in curves[algo][size]) / seeds) for size in runs]
    avg_function_calls[algo] = [(sum(functionCalls for _, functionCalls, _ in curves[algo][size]) / seeds) for size in runs]
    avg_wall_time[algo] = [(sum(wall_time for _, _, wall_time in curves[algo][size]) / seeds) for size in runs]
print(avg_scores)
print(avg_function_calls)
print(avg_wall_time)

fourpeaks = {
    'scores': avg_scores,
    'calls': avg_function_calls,
    'time': avg_wall_time
}

with open('../../fourpeaks.json', 'w') as fp:
    json.dump(fourpeaks, fp)
