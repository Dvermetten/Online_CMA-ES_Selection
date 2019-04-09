#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
import timeit
import pandas as pd
from datetime import datetime
from functools import partial
from copy import copy
from bbob import bbobbenchmarks, fgeneric
from itertools import product

from src import Config
from src.Algorithms import _MIES
from EvolvingES import reorganiseBBOBOutput, ensureFullLengthRepresentation, evaluateCustomizedESs, _displayDuration, evaluateCustomizedSingleSplitESs
from src.Individual import MixedIntIndividual
from src.Parameters import Parameters
from src.Utils import ESFitness, getOpts, options, num_options_per_module, \
    getBitString, getPrintName, create_bounds, guaranteeFolderExists, chunkListByLength, intToRepr, reprToString
from src.local import non_bbob_datapath
import random
# Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs
guaranteeFolderExists(non_bbob_datapath)


opts = {'algid': None,
        'comments': '<comments>',
        'inputformat': 'col'}  # 'row' or 'col'

def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def _testEachOption():
    # Test all individual options
    n = len(options)
    fid = 1
    ndim = 10
    representation = [0] * n
    lambda_mu = [None, None]
    representation.extend(lambda_mu)
    ensureFullLengthRepresentation(representation)
    evaluateCustomizedESs(representation, fid=fid, ndim=ndim, iids=range(Config.ES_num_runs))
    for i in range(n):
        for j in range(1, num_options_per_module[i]):
            representation = [0] * n
            representation[i] = j
            representation.extend(lambda_mu)
            ensureFullLengthRepresentation(representation)
            evaluateCustomizedESs(representation, fid=fid, ndim=ndim, iids=range(Config.ES_num_runs))

    print("\n\n")


def _problemCases():
    fid = 1
    ndim = 10
    iids = range(Config.ES_num_runs)

    # Known problems
    print("Combinations known to cause problems:")

    rep = ensureFullLengthRepresentation(getBitString({'sequential': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'tpa': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'tpa': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    # these are the actual failures
    rep = ensureFullLengthRepresentation(getBitString({'sequential': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'sequential': True, 'tpa': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)

    rep = ensureFullLengthRepresentation([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 113, 0.18770573922911427])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 107, 0.37768142336353183])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation([0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 2, None, None])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 27, 0.9383818903266666])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)

    rep = ensureFullLengthRepresentation([0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 2, 3, 0.923162952008686])
    print(getPrintName(getOpts(rep[:-2])))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)


def _exampleRuns(use_threshold = False):
    fid = 1
    ndim = 10
    iids = range(Config.ES_num_runs)

    print("Mirrored vs Mirrored-pairwise")
    rep = ensureFullLengthRepresentation(getBitString({'mirrored': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    rep = ensureFullLengthRepresentation(getBitString({'mirrored': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    #
    # print("Regular vs Active")
    # rep = ensureFullLengthRepresentation(getBitString({'active': False}))
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    # rep = ensureFullLengthRepresentation(getBitString({'active': True}))
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    #
    # print("No restart vs local restart")
    # rep = ensureFullLengthRepresentation(getBitString({'ipop': None}))
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    # rep = ensureFullLengthRepresentation(getBitString({'ipop': True}))
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    # rep = ensureFullLengthRepresentation(getBitString({'ipop': 'IPOP'}))
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)
    # rep = ensureFullLengthRepresentation(getBitString({'ipop': 'BIPOP'}))
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim, use_threahold=use_threshold)


def thresholdExampleRun(use_threshold = False):
    fid = 10
    ndim = 5
    iids = range(5)
    rep1 = ensureFullLengthRepresentation([0,0,1,1,0,0,0,0,0,0,0])
    np.random.seed(42)
    evaluateCustomizedESs(rep1, iids=iids, fid=fid, ndim=ndim, use_threshold=True)
    np.random.seed(42)
    evaluateCustomizedESs(rep1, iids=iids, fid=fid, ndim=ndim, use_threshold=True)



def _singleSplitExampleRuns():
    fid = 10
    ndim =5
    iids = [0,1,2]
    n_reps = 10
    # iids = range(Config.ES_num_runs)
    print(iids)
    # print("Single swich")
    # rep1 = ensureFullLengthRepresentation([0,1,0,0,1,1,1,0,1,2,0])
    # rep2 = ensureFullLengthRepresentation([0,1,1,0,1,0,0,0,0,1,2])
    # # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    # threshold = pow(10,0.2)
    # evaluateCustomizedSingleSplitESs(rep1, rep2, iids=iids, fid=fid, ndim=ndim, threshold = threshold)
    #
    # rep0 = ensureFullLengthRepresentation([1,1,0,0,0,1,1,0,0,2,2])
    # evaluateCustomizedESs(rep0, iids=iids, fid=fid, ndim=ndim)
    rep1 = ensureFullLengthRepresentation(intToRepr(0))
    rep2 = ensureFullLengthRepresentation(intToRepr(1152))
    print(rep1)
    print(rep2)
    # evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    threshold = 10e-0
    evaluateCustomizedSingleSplitESs(rep1, rep2, iids=iids, fid=fid, ndim=ndim, split = threshold,num_reps=n_reps)

    # rep0 = ensureFullLengthRepresentation([1,1,0,0,0,1,1,0,0,2,2])
    # evaluateCustomizedESs(rep1, iids=iids, fid=fid, ndim=ndim,use_threshold=True,num_reps=n_reps, split = threshold)
    # evaluateCustomizedESs(rep2, iids=iids, fid=fid, ndim=ndim,use_threshold=True,num_reps=n_reps,split = threshold)
    # evaluateCustomizedESs(rep0, iids=iids, fid=fid, ndim=ndim)


def _bruteForce(ndim, fid, parallel=1, part=0):
    # Exhaustive/brute-force search over *all* possible combinations
    # NB: This assumes options are sorted ascending by number of possible values per option
    num_combinations = np.product(num_options_per_module)
    print("F{} in {} dimensions:".format(fid, ndim))
    print("Brute-force exhaustive search of *all* available ES-combinations.")
    print("Number of possible ES-combinations currently available: {}".format(num_combinations))
    from collections import Counter
    from itertools import product
    from datetime import datetime
    import cPickle

    best_ES = None
    best_result = ESFitness()

    progress_fname = non_bbob_datapath + '{}_f{}.prog'.format(ndim, fid)
    try:
        with open(progress_fname) as progress_file:
            start_at = cPickle.load(progress_file)
    except:
        start_at = 0

    if start_at >= np.product(num_options_per_module):
        return
    if part == 1 and start_at >= num_combinations // 2:  # Been there, done that
        return
    elif part == 2 and start_at < num_combinations // 2:
        raise ValueError("Unexpected value for 'start_at' in part 2: {}".format(start_at))

    products = []
    # count how often there is a choice of x options
    counts = Counter(num_options_per_module)
    for num, count in sorted(counts.items(), key=lambda x: x[0]):
        products.append(product(range(num), repeat=count))

    if Config.write_output:
        storage_file = '{}bruteforce_{}_f{}.tdat'.format(non_bbob_datapath, ndim, fid)
    else:
        storage_file = None

    all_combos = []
    for combo in list(product(*products)):
        all_combos.append(list(sum(combo, ())))

    x = datetime.now()
    for combinations in chunkListByLength(all_combos[start_at:], parallel):
        bitstrings = [ensureFullLengthRepresentation(bitstring) for bitstring in combinations]
        results = evaluateCustomizedESs(bitstrings, fid=fid, ndim=ndim, num_reps=10,
                                        iids=range(Config.ES_num_runs), storage_file=storage_file)

        start_at += parallel
        with open(progress_fname, 'w') as progress_file:
            cPickle.dump(start_at, progress_file)

        for result, bitstring in zip(results, bitstrings):
            if result < best_result:
                best_result = result
                best_ES = bitstring

    y = datetime.now()

    print("Best ES found:       {}\n"
          "With fitness: {}\n".format(best_ES, best_result))

    _displayDuration(x, y)


def _runGA(ndim=5, fid=1, run=1):
    x = datetime.now()

    # Where to store genotype-fitness information
    # storage_file = '{}GA_results_{}dim_f{}.tdat'.format(non_bbob_datapath, ndim, fid)
    storage_file = '{}MIES_results_{}dim_f{}run_{}.tdat'.format(non_bbob_datapath, ndim, fid, run)

    # Fitness function to be passed on to the baseAlgorithm
    fitnessFunction = partial(evaluateCustomizedESs, fid=fid, ndim=ndim,
                              iids=range(Config.ES_num_runs), storage_file=storage_file)

    parameters = Parameters(len(options) + 15, Config.GA_budget, mu=Config.GA_mu, lambda_=Config.GA_lambda)
    parameters.l_bound[len(options):] = np.array([  2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(15)
    parameters.u_bound[len(options):] = np.array([200, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]).reshape(15)

    # Initialize the first individual in the population
    discrete_part = [np.random.randint(len(x[1])) for x in options]
    lamb = int(4 + np.floor(3 * np.log(parameters.n)))
    int_part = [lamb]
    float_part = [
        parameters.mu,
        parameters.alpha_mu, parameters.c_sigma, parameters.damps, parameters.c_c, parameters.c_1,
        parameters.c_mu,
        0.2, 0.955,
        0.5, 0, 0.3, 0.5,
        2
    ]

    population = [
        MixedIntIndividual(len(discrete_part) + len(int_part) + len(float_part),
                           num_discrete=len(num_options_per_module),
                           num_ints=len(int_part))
    ]
    population[0].genotype = np.array(discrete_part + int_part + float_part)
    population[0].fitness = ESFitness()

    while len(population) < Config.GA_mu:
        population.append(copy(population[0]))

    u_bound, l_bound = create_bounds(float_part, 0.3)
    parameters.u_bound[len(options) + 1:] = np.array(u_bound)
    parameters.l_bound[len(options) + 1:] = np.array(l_bound)

    gen_sizes, sigmas, fitness, best = _MIES(n=ndim, fitnessFunction=fitnessFunction, budget=Config.GA_budget,
                                             mu=Config.GA_mu, lambda_=Config.GA_lambda, parameters=parameters,
                                             population=population)  # This line does all the work!
    y = datetime.now()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.genotype, best.fitness, fitness))

    z = _displayDuration(x, y)

    if Config.write_output:
        np.savez("{}final_GA_results_{}dim_f{}_run{}".format(non_bbob_datapath, ndim, fid, run),
                 sigma=sigmas, best_fitness=fitness, best_result=best.genotype,
                 generation_sizes=gen_sizes, time_spent=z)


def _runExperiments():
    for ndim in Config.experiment_dims:
        for fid in Config.experiment_funcs:

            # Initialize the first individual in the population
            discrete_part = [np.random.randint(len(x[1])) for x in options]
            lamb = int(4 + np.floor(3 * np.log(parameters.n)))
            int_part = [lamb]
            float_part = [
                parameters.mu,
                parameters.alpha_mu, parameters.c_sigma, parameters.damps, parameters.c_c, parameters.c_1,
                parameters.c_mu,
                0.2, 0.955,
                0.5, 0, 0.3, 0.5,
                2
            ]

            population = [
                MixedIntIndividual(len(discrete_part) + len(int_part) + len(float_part),
                                   num_discrete=len(num_options_per_module),
                                   num_ints=len(int_part))
            ]
            population[0].genotype = np.array(discrete_part + int_part + float_part)
            population[0].fitness = ESFitness()

            while len(population) < Config.GA_mu:
                population.append(copy(population[0]))

            parameters = Parameters(len(options) + 15, Config.GA_budget, mu=Config.GA_mu, lambda_=Config.GA_lambda)
            parameters.l_bound[len(options):] = np.array([  2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(15)
            parameters.u_bound[len(options):] = np.array([200, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]).reshape(15)
            u_bound, l_bound = create_bounds(float_part, 0.3)
            parameters.u_bound[len(options) + 1:] = np.array(u_bound)
            parameters.l_bound[len(options) + 1:] = np.array(l_bound)

            print("Optimizing for function ID {} in {}-dimensional space:".format(fid, ndim))
            x = datetime.now()
            gen_sizes, sigmas, fitness, best = _MIES(n=ndim, fitnessFunction=fid, budget=Config.GA_budget,
                                                     mu=Config.GA_mu, lambda_=Config.GA_lambda, parameters=parameters,
                                                     population=population)
            y = datetime.now()

            z = y - x
            np.savez("{}final_GA_results_{}dim_f{}".format(non_bbob_datapath, ndim, fid),
                     sigma=sigmas, best_fitness=fitness, best_result=best.genotype,
                     generation_sizes=gen_sizes, time_spent=z)


def runComparativeTableTestSimple(config_idx, use_stored_hyperparameters=False):
    overview_table = pd.read_csv("data/overview_table_extended.csv", index_col=0)
    
    relevant_table = [[overview_table["fid"][idx],
                       overview_table["split"][idx],
                       overview_table["target"][idx],
                       overview_table["Static index"][idx],
                       overview_table["$C_1$ index"][idx],
                       overview_table["$C_{\Gamma}$ index"][idx]] for idx in range(22)]

    iids = range(5)
    num_reps = 50

    config = relevant_table[config_idx]
    fid = config[0]
    ndim = 5
    rep0 = ensureFullLengthRepresentation(intToRepr(config[3]))
    rep1 = ensureFullLengthRepresentation(intToRepr(config[4]))
    rep2 = ensureFullLengthRepresentation(intToRepr(config[5]))
    if type(config[1]) == type("string"):
        split_exp = float(config[1][3:])
    else:
        split_exp = float(config[1])
    split = 10**split_exp
    print(fid,ndim)
    print("Part 1 only")
    evaluateCustomizedESs(rep1,split=split, iids=iids, fid=fid, ndim=ndim, use_threshold=True, num_reps=num_reps)
    # print("Part 2 only")
    if (rep2 != rep1):
        evaluateCustomizedESs(rep2,split=split, iids=iids, fid=fid, ndim=ndim, use_threshold=True, num_reps=num_reps,save_hyperparams=use_stored_hyperparameters)
    # print("Best static")
    if (rep0 != rep1 and rep0 != rep2):
        evaluateCustomizedESs(rep0,split=split, iids=iids, fid=fid, ndim=ndim, use_threshold=True, num_reps=num_reps)
    # print("Single split")
    evaluateCustomizedSingleSplitESs(rep1, rep2, iids=iids, fid=fid, ndim=ndim, split=split, num_reps=num_reps,load_hyperparameters=use_stored_hyperparameters)




def instance_based_experiment(idx):
    overview_table = pd.read_csv("instance_based_WW_cleaned.csv", index_col=0)

    relevant_table = [[overview_table["fid"][i],
                       overview_table["split"][i],
                       overview_table["target"][i],
                       overview_table["Static index"][i],
                       overview_table["$C_1$ index"][i],
                       overview_table["$C_{\Gamma}$ index"][i],
                       overview_table["iid"][i]] for i in range(110)]

    num_reps = 50
    config = relevant_table[idx]
    iids = [config[-1]]
    fid = config[0]
    ndim = 5
    rep0 = ensureFullLengthRepresentation(intToRepr(config[3]))
    rep1 = ensureFullLengthRepresentation(intToRepr(config[4]))
    rep2 = ensureFullLengthRepresentation(intToRepr(config[5]))
    split_exp = float(config[1])
    split = 10 ** split_exp
    print(fid, ndim)
    # print("Part 1 only")
    evaluateCustomizedESs(rep1,split=split, iids=iids, fid=fid, ndim=ndim, use_threshold=True, num_reps=num_reps)
    # print("Part 2 only")
    if (rep2 != rep1):
    # if use_stored_hyperparameters:
        evaluateCustomizedESs(rep2,split=split, iids=iids, fid=fid, ndim=ndim, use_threshold=True, num_reps=num_reps)
    # print("Best static")
    if (rep0 != rep1 and rep0 != rep2):
        evaluateCustomizedESs(rep0,split=split, iids=iids, fid=fid, ndim=ndim, use_threshold=True, num_reps=num_reps)
    # print("Single split")
    print(evaluateCustomizedSingleSplitESs(rep1, rep2, iids=iids, fid=fid, ndim=ndim,
                                           split=split, num_reps=num_reps))


def multiSplitExp(fid,idx):
    confs = np.load("F{0}_bestsplits.npy".format(fid))
    config = confs[idx]
    rep1 = ensureFullLengthRepresentation(intToRepr(int(config[1])))
    rep2 = ensureFullLengthRepresentation(intToRepr(int(config[2])))
    powers = [ 2. ,  1.8,  1.6,  1.4,  1.2,  1. ,  0.8,  0.6,  0.4,  0.2,  0. , -0.2, -0.4, -0.6, -0.8, -1. , -1.2, -1.4, -1.6, -1.8, -2. , -2.2, -2.4, -2.6, -2.8, -3. , -3.2, -3.4, -3.6, -3.8, -4. , -4.2, -4.4, -4.6, -4.8, -5. , -5.2, -5.4, -5.6, -5.8, -6. , -6.2, -6.4, -6.6, -6.8, -7. , -7.2, -7.4, -7.6, -7.8, -8. ]
    split = 10**powers[int(config[0])]
    num_reps = 50
    iids = range(5)
    ndim = 5
    evaluateCustomizedSingleSplitESs(rep1, rep2, iids=iids, fid=fid, ndim=ndim,
                                     split=split, num_reps=num_reps)

def get_bbob_contour(fid,iid):
    f = fgeneric.LoggingFunction("D:/Research_project/data/trash")
    f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=iid))
    f_target = f.ftarget
    bests = np.load("D:/Research_project/data/contours/mean{0}.npy".format(fid))
    best = bests[iid]
    for i in range(5):
        for j in range(5):
            if i!=j:
                valslist = []
                vecs = product(np.arange(-5,5,0.1),np.arange(-5,5,0.1))
                for x4,x5 in vecs:
                    vec = best
                    vec[i] = x4
                    vec[j] = x5
                    valslist.append(f(vec) - f_target)
                np.save("D:/Research_project/data/contours/{0}_{1}_{2}_{3}".format(fid,iid,i,j),np.array(valslist))

def runDefault():
    # _runGA()
    # _testEachOption()
    # _problemCases()
    # _exampleRuns()
    # _bruteForce(ndim=10, fid=1)
    # _runExperiments()
    # reorganizeBbobcleanup()
    # runComparativeTableTestSimple(5,use_stored_hyperparameters=True)
    # runComparativeTableTestSimple(12)
    # runComparativeTableTestSimple(7)
    # runComparativeTableTestSimple(3)
    # runComparativeTableTestSimple(5,use_stored_hyperparameters=False)
    # data_to_rerun = np.load("to_run_mean_calcd.npy")
    # for fid,c1,c2,split,st in data_to_rerun:
    #     evaluateCustomizedSingleSplitESs(ensureFullLengthRepresentation((intToRepr(c1))),
    #                                        ensureFullLengthRepresentation(intToRepr(c2)),
    #                                        iids=range(5), fid=fid, ndim=5,
    #                                        split=10**(2-split/5), num_reps=50)
    #     evaluateCustomizedESs(ensureFullLengthRepresentation(intToRepr(st)),
    #                            iids=range(5), fid=fid, ndim=5,
    #                            split=10**(2-split/5), num_reps=50)
    # for fid in [5,8,10,15]:
    #     for iid in range(5):
    #
    #         get_bbob_contour(fid,iid)
    # evaluateCustomizedESs(ensureFullLengthRepresentation(intToRepr(111)),iids=[0],fid=2,ndim=5,use_threshold=True,num_reps=50)
    from bbob import bbobbenchmarks, fgeneric
    bbob_opts = {'algid': None,
                 'comments': '<comments>',
                 'inputformat': 'col'}
    bbob_opts['algid'] = 'dsfaf'
    datapath_ext = 'test'
    guaranteeFolderExists('test')

    f = fgeneric.LoggingFunction('test', **bbob_opts)
    targets = np.zeros((24,6))
    for fid in [i+1 for i in range(24)]:
        for iid in range(6):
            f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=iid))
            targets[fid-1,iid] = f.ftarget
    np.save('targets_bbob',targets)
    pass

def run5050_exp(fid, idx):
    items = np.load("Splits_F{0}_new.npy".format(fid))
    item = items[idx]
    rep1 = ensureFullLengthRepresentation(intToRepr(int(item[1])))
    rep2 = ensureFullLengthRepresentation(intToRepr(int(item[2])))
    iids = range(5)
    ndim = 5
    split = 10 ** (2 - int(item[0]) / 5)
    num_reps = 50
    evaluateCustomizedSingleSplitESs(rep1, rep2, iids=iids, fid=fid, ndim=ndim,
                                     split=split, num_reps=num_reps)


def run_statics(fid,idx):
    items = np.load("Statics_F{0}.npy".format(fid))
    item = items[idx]
    rep = ensureFullLengthRepresentation(intToRepr(int(item)))
    # rep2 = ensureFullLengthRepresentation(intToRepr(int(item[1])))
    iids = range(5)
    ndim = 5
    split = 0 #10 ** (2 - int(item[0]) / 5)
    num_reps = 50
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim,
                                     split=split, num_reps=num_reps)


def testRepresentations():
    rep1 = ensureFullLengthRepresentation(intToRepr(0))
    rep2 = ensureFullLengthRepresentation(intToRepr(9))
    print(rep1)
    print(rep2)

def reorganizeBbobcleanup():
    path = "D:/Research_Project/data/test_results"
    fids = [2,3,4,5,6,8,9,11,12,13,14,16,17,18,19,21,22,23,24]
    ndim = 5
    iids = range(5)
    num_reps = 5
    for fid in fids:
        for i in range(4608):
            reorganiseBBOBOutput(path + '/' + reprToString(intToRepr(i))+ '/', fid, ndim, iids, num_reps)

def main():
    # np.seterr(all='raise')
    np.set_printoptions(linewidth=1000, precision=3)
    if len(sys.argv) == 2:
        print("running index: {0}".format(sys.argv[1]))
        instance_based_experiment(int(sys.argv[1]))
    elif len(sys.argv) == 3:
        # ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        run5050_exp(13, fid)
        # run_statics(int(sys.argv[1]), int(sys.argv[2]))
        # _runGA(ndim, fid)
    elif len(sys.argv) == 4:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        run = int(sys.argv[3])
        _runGA(ndim, fid, run)
    elif len(sys.argv) == 5:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        parallel = int(sys.argv[3])
        part = int(sys.argv[4])
        _bruteForce(ndim, fid, parallel, part)
    else:
        runDefault()


if __name__ == '__main__':
    main()
