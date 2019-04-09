from scipy import stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter, defaultdict
from matplotlib import rc
from functools import partial
from itertools import product
from multiprocessing import Pool


num_options_per_module = [2]*9        # Binary part
num_options_per_module.extend([3]*2)  # Ternary part
max_length = 11
factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]


data_location = 'D:/Research_project/data/anytime_convergence/data/'
file_name = 'interpolated_ART_data_{ndim}D-f{fid}.csv'


#TODO: Retrieve this information from the files instead?
ndims = [5]
fids = [2,3,4,5,6,8,9,10,11,12,13,14, 15,16,17,18,19,20,21,22,23,24]

num_steps = 51
powers = np.round(np.linspace(2, -8, num_steps), decimals=1)
target_values = np.power([10]*num_steps, powers)

def reprToString(representation):
    """ Function that converts the structure parameters of a given ES-structure representation to a string

        >>> reprToInt([0,0,0,0,0,1,0,1,0,1,0])
        >>> '00000101010'
    """
    return ''.join([str(i) for i in representation[:max_length]])

def intToRepr(integer):
    """
        Dencode the ES-structure from a single integer back to the mixed base-2 and 3 representation.
        Reverse of :func:`~reprToInt`

        >>> intToRepr(93)
        >>> [0,0,0,0,0,1,0,1,0,1,0]

        :param integer: Integer (e.g. outoutput from reprToInt() )
        :returns:       String consisting of all structure choices concatenated,
    """
    max_length = 11
    factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]
    representation = []
    for i in range(max_length):
        if integer >= factors[i]:
            gene = integer // factors[i]
            integer -= gene * factors[i]
        else:
            gene = 0
        representation.append(gene)

    return representation

def reprToInt(representation):
    """
        Encode the ES-structure representation to a single integer by converting it to base-10 as if it is a
        mixed base-2 or 3 number. Reverse of :func:`~intToRepr`

        >>> reprToInt([0,0,0,0,0,1,0,1,0,1,0])
        >>> 93

        :param representation:  Iterable; the genotype of the ES-structure
        :returns:               String consisting of all structure choices concatenated,
    """
    # TODO FIXME Hardcoded
    max_length = 11
    factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]
    integer = 0
    for i in range(max_length):
        integer += representation[i] * factors[i]

    return integer

def get_data(ndim, fid):
    return pd.read_csv(data_location + file_name.format(ndim=ndim, fid=fid), index_col=0)


def runPool(runFunction, arguments):
    """
        Small overhead-function to handle multi-processing using Python's built-in multiprocessing.Pool

        :param runFunction: The (``partial``) function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    p = Pool( len(arguments))

    local_func = partial(func_star, func=runFunction)
    results = p.map(local_func, arguments)
    return results


configs_to_consider = [i * 3 for i in range(int(4608 / 3))]


def calculatesplitbasedrecord_worstcase(df, sliding_window=False):
    dfa = np.array(df)
    best_target = 5
    #     configs_to_consider = dist2list #[3*i for i in range(int(1002/3))]

    best_worstcase = np.inf
    bestidx1 = -1
    bestidx2 = -1
    bestsplit = -1

    for idx1 in configs_to_consider:
        #         print(idx1)
        for idx2 in configs_to_consider:
            vals1 = [dfa[i][5:] for i in range(idx1 * 25, (1 + idx1) * 25)]
            vals2 = [dfa[i][5:] for i in range(idx2 * 25, (1 + idx2) * 25)]
            #         valsstatic = dfa[25*idxstatic:25*(idxstatic+1)]
            #         splitidx = [i for i in range(-51,0) if columns[i] == table_item[1]][0]
            #         targetidx = [i for i in range(-51,0) if columns[i] == table_item[2]][0]

            targets_hit = [i for i in range(51) if
                           (np.sum(vals1, axis=0)[i] < np.inf and np.sum(vals2, axis=0)[i] < np.inf)]
            #             print(targets_hit, idx1, idx2)
            if len(targets_hit) == 0:
                #             print(idx1,idx2)
                continue
            targetidx = max(targets_hit)

            if targetidx > best_target:
                best_worstcase = np.inf
                bestidx1 = -1
                bestidx2 = -1
                bestsplit = -1
                best_target = targetidx

            if targetidx == best_target:
                value_split = []
                for split in range(targetidx):
                    p1 = [vals1[i][split] for i in range(25)]
                    p2 = [vals2[i][targetidx] - vals2[i][split] for i in range(25)]

                    combined = []
                    for iid in range(5):
                        worstp1 = max(p1[5 * iid:5 * (iid + 1)])
                        worstp2 = max(p2[5 * iid:5 * (iid + 1)])
                        combined.append(worstp1 + worstp2)

                    worstcase_mean = np.mean(combined)
                    value_split.append(worstcase_mean)
                if not sliding_window:
                    split = np.argmin(value_split)
                    worstcase_mean = np.min(value_split)
                else:
                    summed_window = [np.mean(value_split[i - 2:i + 3]) for i in range(2, targetidx - 2)]
                    window_split = np.argmin(np.array(summed_window))
                    split = window_split + 2
                    worstcase_mean = summed_window[window_split]
                if worstcase_mean < best_worstcase:
                    bestsplit = split
                    bestidx1 = idx1
                    bestidx2 = idx2
                    best_worstcase = worstcase_mean
    return (best_target, bestidx1, bestidx2, bestsplit, best_worstcase)


def calcworstcasestatic(df, target):
    dfa = np.array(df)
    worsts = []
    for idx in configs_to_consider:
        vals = [dfa[i][5 + target] for i in range(idx * 25, (1 + idx) * 25)]
        #         print(idx,vals)
        worstiids = [max(vals[5 * i:5 * (i + 1)]) for i in range(5)]
        worsts.append(np.mean(worstiids))
    # print(combined)
    #     print(worsts)
    #     print(worsts == np.nan)
    #     print(len([i for i in worsts if i == i]))
    #     print([i for i in worsts if i == i])
    if len([i for i in worsts if i == i]) == 0:
        return (np.inf, -1)
    bestst = np.nanmin(worsts)

    return (bestst, configs_to_consider[np.nanargmin(worsts)])


def calcMeanImprovement(fid, idx1, idx2, idxst, target,split):
    df = get_data(5, fid)
    dfa = np.array(df)
    vals1 = [dfa[i][5:] for i in range(idx1 * 25, (idx1 + 1) * 25)]
    vals2 = [dfa[i][5:] for i in range(idx2 * 25, (idx2 + 1) * 25)]
    meanst = np.mean([dfa[i][5 + target] for i in range(idxst, idxst + 25)])
    p1 = [vals1[i][split] for i in range(25)]
    p2 = [vals2[i][target] - vals2[i][split] for i in range(25)]

    combined = []
    for iid in range(5):
        meanp1 = np.mean(p1[5 * iid:5 * (iid + 1)])
        meanp2 = np.mean(p2[5 * iid:5 * (iid + 1)])
        combined.append(meanp1 + meanp2)

    meanmean = np.mean(combined)
    print(fid, meanmean, meanst)
    return 1 - (meanmean / meanst)


def calculatesplitbasedoverview_worstcase(cases, sliding_window=False):
    records = []
    for ndim, fid in cases:
        print(fid)
        df = get_data(ndim, fid)
        record = calculatesplitbasedrecord_worstcase(df, sliding_window)
        recordst = calcworstcasestatic(df, record[0])
        improvement = 1 - (record[-1] / recordst[0])
        improvementmean = calcMeanImprovement(fid, record[1], record[2], recordst[1], record[3],record[3])
        records.append((ndim, fid, *record, *recordst, improvement, improvementmean))
    #         records.append(recordst[0])
    #         records.append(recordst[1])
    #     print(records)
    labels = ['ndim', 'fid', 'target', 'Idx1', 'Idx2',
              'split', 'split_value', 'static_value', 'Static idx', 'improvement possible (worstcase)',
              'improvement possible (mean)']
    results = pd.DataFrame.from_records(records, columns=labels)
    return results

worstcase_overview = calculatesplitbasedoverview_worstcase(cases = product([5],[2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]),sliding_window=True)
worstcase_overview.to_csv("Worstcase_overview.csv")