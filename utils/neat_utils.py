import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import warnings

from task import *


def listXor(b,c):
    """Returns elements in lists b and c they don't share
    """
    A = [a for a in b+c if (a not in b) or (a not in c)]
    return A

def rankArray(X):
    """Returns ranking of a list, with ties resolved by first-found first-order
    NOTE: Sorts descending to follow numpy conventions
    """
    tmp = np.argsort(X)
    rank = np.empty_like(tmp)
    rank[tmp] = np.arange(len(X))
    return rank

def tiedRank(X):
    """Returns ranking of a list, with ties recieving and averaged rank
    # Modified from: github.com/cmoscardi/ox_ml_practical/blob/master/util.py
    """
    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort(reverse=True)
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0

    return np.asarray(Rx)

def bestIntSplit(ratio, total):
    """Divides a total into integer shares that best reflects ratio
      Args:
        share      - [1 X N ] - Percentage in each pile
        total      - [int   ] - Integer total to split

      Returns:
        intSplit   - [1 x N ] - Number in each pile
    """
    # Handle poorly defined ratio
    if sum(ratio) is not 1:
        ratio = np.asarray(ratio)/sum(ratio)

    # Get share in real and integer values
    floatSplit = np.multiply(ratio,total)
    intSplit   = np.floor(floatSplit)
    remainder  = int(total - sum(intSplit))

    # Rank piles by most cheated by rounding
    deserving = np.argsort(-(floatSplit-intSplit),axis=0)

    # Distribute remained to most deserving
    intSplit[deserving[:remainder]] = intSplit[deserving[:remainder]] + 1
    return intSplit

def quickINTersect(A,B):
    """ Faster set intersect: only valid for vectors of positive integers.
    (useful for matching indices)

      Example:
      A = np.array([0,1,2,3,5],dtype=np.int16)
      B = np.array([0,1,6,5],dtype=np.int16)
      C = np.array([0],dtype=np.int16)
      D = np.array([],dtype=np.int16)

      print(quickINTersect(A,B))
      print(quickINTersect(B,C))
      print(quickINTersect(B,D))
    """
    if (len(A) is 0) or (len(B) is 0):
        return [],[]
    P = np.zeros((1+max(max(A),max(B))),dtype=bool)
    P[A] = True
    IB = P[B]
    P[A] = False # Reset
    P[B] = True
    IA = P[A]

    return IA, IB


def nsga_sort(obj_vals, ret_fronts=False):
    """Returns ranking of objective values based on non-dominated sorting.
    Optionally returns fronts (useful for visualization).

    NOTE: Assumes maximization of objective function

    Args:
        obj_vals - (np_array) - Objective values of each individual [nInds X nObjectives]

    Returns:
        rank    - (np_array) - Rank in population of each individual
                int([nIndividuals X 1])
        front   - (np_array) - Pareto front of each individual
                int([nIndividuals X 1])
    """

    fronts = get_fronts(obj_vals)

    # Rank each individual in each front by crowding distance
    for f in range(len(fronts)):
        x1 = obj_vals[fronts[f], 0]
        x2 = obj_vals[fronts[f], 1]
        crowdDist = get_crowd_dist(x1) + get_crowd_dist(x2)
        frontRank = np.argsort(-crowdDist)
        fronts[f] = [fronts[f][i] for i in frontRank]

    # Convert to ranking
    tmp = [ind for front in fronts for ind in front]
    rank = np.empty_like(tmp)
    rank[tmp] = np.arange(len(tmp))

    if ret_fronts is True:
        return rank, fronts
    else:
        return rank


def get_fronts(obj_vals):
    """Fast non-dominated sort.

    Args:
        obj_vals - (np_array) - Objective values of each individual
                  [nInds X nObjectives]

    Returns:
        front   - [list of lists] - One list for each front:
                                    list of indices of individuals in front

    [adapted from: https://github.com/haris989/NSGA-II]
    """

    values1 = obj_vals[:, 0]
    values2 = obj_vals[:, 1]

    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    # Get domination relations
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0

        for q in range(0, len(values1)):
            if ((values1[p] > values1[q] and values2[p] > values2[q])
                    or (values1[p] >= values1[q] and values2[p] > values2[q])
                    or (values1[p] > values1[q] and values2[p] >= values2[q])):
                if q not in S[p]:
                    S[p].append(q)
            elif ((values1[q] > values1[p] and values2[q] > values2[p])
                  or (values1[q] >= values1[p] and values2[q] > values2[p])
                  or (values1[q] > values1[p] and values2[q] >= values2[p])):
                n[p] = n[p] + 1

        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    # Assign fronts
    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)

    del front[len(front) - 1]

    return front


def get_crowd_dist(objVector):
    """Returns crowding distance of a vector of values, used once on each front.

    Note: Crowding distance of individuals at each end of front is infinite, as they don't have a neighbor.

    Args:
        objVector - (np_array) - Objective values of each individual
                    [nInds X nObjectives]

    Returns:
        dist      - (np_array) - Crowding distance of each individual
                    [nIndividuals X 1]
    """
    # Order by objective value

    key = np.argsort(objVector)
    sortedObj = objVector[key]

    # Distance from values on either side
    shiftVec = np.r_[np.inf, sortedObj, np.inf]  # Edges have infinite distance
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # inf on purpose
    prevDist = np.abs(sortedObj - shiftVec[:-2])
    nextDist = np.abs(sortedObj - shiftVec[2:])
    crowd = prevDist + nextDist
    if (sortedObj[-1] - sortedObj[0]) > 0:
        crowd *= abs((1 / sortedObj[-1] - sortedObj[0]))  # Normalize by fitness range

    # Restore original order
    dist = np.empty(len(key))
    dist[key] = crowd[:]

    return dist


def loadHyp(pFileName, printHyp=False):
    """Loads hyperparameters from .json file
    Args:
        pFileName - (string) - file name of hyperparameter file
        printHyp  - (bool)   - print contents of hyperparameter file to terminal?

    Note: see p/hypkey.txt for detailed hyperparameter description
    """
    with open(pFileName) as data_file:
        hyp = json.load(data_file)

    # Task hyper parameters
    task = GymTask(games[hyp['task']], param_only=True)
    hyp['ann_nInput'] = task.nInput
    hyp['ann_nOutput'] = task.nOutput
    hyp['ann_initAct'] = task.activations[0]
    hyp['ann_absWCap'] = task.absWCap
    hyp['ann_mutSigma'] = task.absWCap * 0.2
    hyp['ann_layers'] = task.layers  # if fixed toplogy is used

    if hyp['alg_act'] == 0:
        hyp['ann_actRange'] = task.actRange
    else:
        hyp['ann_actRange'] = np.full_like(task.actRange, hyp['alg_act'])

    if printHyp is True:
        print(json.dumps(hyp, indent=4, sort_keys=True))
    return hyp


def updateHyp(hyp, pFileName=None):
    """Overwrites default hyperparameters with those from second .json file
    """
    if pFileName != None:
        print('\t*** Running with hyperparameters: ', pFileName, '\t***')
        with open(pFileName) as data_file:
            update = json.load(data_file)
        hyp.update(update)

        # Task hyper parameters
        task = GymTask(games[hyp['task']], param_only=True)
        hyp['ann_nInput'] = task.nInput
        hyp['ann_nOutput'] = task.nOutput
        hyp['ann_initAct'] = task.activations[0]
        hyp['ann_absWCap'] = task.absWCap
        hyp['ann_mutSigma'] = task.absWCap * 0.1
        hyp['ann_layers'] = task.layers  # if fixed toplogy is used

        if hyp['alg_act'] == 0:
            hyp['ann_actRange'] = task.actRange
        else:
            hyp['ann_actRange'] = np.full_like(task.actRange, hyp['alg_act'])
