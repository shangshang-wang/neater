import glob
import warnings
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

from task import *
from task.task_config import games
from pretty_neat import *
from .visualize_net import *
from .plot import *


def view_classifier(ind, taskName, seed=42):
    task = GymTask(games[taskName])
    env = games[taskName]
    if isinstance(ind, str):
        ind = np.loadtxt(ind, delimiter=',')
        wMat = ind[:, :-1]
        aVec = ind[:, -1]
    else:
        wMat = ind.wMat
        aVec = np.zeros((np.shape(wMat)[0]))

    # Create Graph
    nIn = env.input_size + 1  # bias
    nOut = env.output_size
    G, layer = ind2graph(wMat, nIn, nOut)
    pos = getNodeCoord(G, layer, taskName)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
    drawEdge(G, pos, wMat, layer, ax1)
    nx.draw_networkx_nodes(G, pos, \
                           node_color='lightblue', node_shape='o', \
                           cmap='terrain', vmin=0, vmax=6, ax=ax1)
    drawNodeLabels(G, pos, aVec, ax1)
    labelInOut(pos, env, ax1)

    ax1.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    task.env._generate_data(type=task.env.type, seed=seed)
    X, y = task.env.trainSet, task.env.target
    # Predict logits
    annOut = act(wMat, aVec, task.nInput, task.nOutput, X)
    action = selectAct(annOut, task.actSelect)
    pred = np.where(action > 0.5, 1, 0).reshape(-1, 1)
    test_acc = np.mean(pred == y)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), num=1000),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), num=1000))
    pred_contour = selectAct(act(wMat, aVec, task.nInput, task.nOutput, np.c_[xx.ravel(), yy.ravel()]), task.actSelect)
    pred_contour = np.where(pred_contour > 0.5, 1, 0).reshape(xx.shape)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Plot setup as before
    ax2.contourf(xx, yy, pred_contour, alpha=0.8, levels=np.linspace(0, 1, 11), cmap=plt.cm.coolwarm)
    ax2.scatter(X[pos_idx, 0], X[pos_idx, 1], c='r', marker='o', edgecolors='k')
    ax2.scatter(X[neg_idx, 0], X[neg_idx, 1], c='b', marker='o', edgecolors='k')
    ax2.text(xx.min() + 0.3, yy.min() + 0.7, f'Test accuracy = {test_acc * 100:.1f}%', fontsize=12)
    ax2.set_title('2D Classification with Decision Boundary')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())

    fig.savefig(f'assets/outputs/{taskName}_classifier.png', dpi=fig.dpi)

    return fig, ax1, ax2


def viewReps(prefix,label=[],val='Fit', title='Fitness', \
             axis=False, getBest=False):
    fig, ax = getAxis(axis)
    fig.dpi=100
    bestRun = []
    for pref in prefix:
        statFile = sorted(glob.glob(pref + '*stats.out'))
        if len(statFile) == 0:
            print('ERROR: No files with that prefix found (it is a list?)')
            return False

        for i in range(len(statFile)):
            tmp = lload(statFile[i])
            if i == 0:
                x = tmp[:,0]
                if val == 'Conn':
                    fitVal = tmp[:,5]
                else: # Fitness
                    fitVal = tmp[:,3]
                    bestVal = fitVal[-1]
                    bestRun.append(statFile[i])
            else:
                if np.shape(tmp)[0] != np.shape(fitVal)[0]:
                    print("Incomplete file found, ignoring ", statFile[i], ' and later.')
                    break

                if val == 'Conn':
                    fitVal = np.c_[fitVal,tmp[:,5]]
                else: # Fitness
                    fitVal = np.c_[fitVal,tmp[:,3]]
                    if fitVal[-1,-1] > bestVal:
                        bestVal = fitVal[-1,-1]
                        bestRun[-1] = statFile[i]

        x = np.arange(len(x))
        lquart(x,fitVal,axis=ax) # Display Quartiles

    # Legend
    if len(label) > 0:
        newLeg = []
        for i in range(len(label)):
            newLeg.append(label[i])
            newLeg.append('_nolegend_')
            newLeg.append('_nolegend_')
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.gca().legend((newLeg))
    plt.title(title)
    plt.xlabel('Evaluations')
    plt.xlabel('Generations')
    if val == 'Conn':
        plt.ylabel('Median Connections')
    else: # Fitness
        plt.ylabel('Best Fitness Found')

    if getBest is True:
        return fig,ax,bestRun
    else:
        return fig,ax
# -- ------------ -- ----------------------------------------------#

def getAxis(axis):
    if axis is not False:
        ax = axis
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots()

    return fig,ax





def viewInd(ind, taskName):
    env = games[taskName]
    if isinstance(ind, str):
        ind = np.loadtxt(ind, delimiter=',')
        wMat = ind[:, :-1]
        aVec = ind[:, -1]
    else:
        wMat = ind.wMat
        aVec = np.zeros((np.shape(wMat)[0]))
        # print('# of Connections in ANN: ', np.sum(wMat!=0))
    print('# of Connections in ANN: ', np.sum(~np.isnan(wMat)))

    # Create Graph
    nIn = env.input_size + 1  # bias
    nOut = env.output_size
    G, layer = ind2graph(wMat, nIn, nOut)
    pos = getNodeCoord(G, layer, taskName)

    # Draw Graph
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    drawEdge(G, pos, wMat, layer)
    nx.draw_networkx_nodes(G, pos, \
                           node_color='lightblue', node_shape='o', \
                           cmap='terrain', vmin=0, vmax=6)
    drawNodeLabels(G, pos, aVec)
    labelInOut(pos, env)

    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    return fig, ax


def ind2graph(wMat, nIn, nOut):
    hMat = wMat[nIn:-nOut, nIn:-nOut]
    hLay = getLayer(hMat) + 1

    if len(hLay) > 0:
        lastLayer = max(hLay) + 1
    else:
        lastLayer = 1
    L = np.r_[np.zeros(nIn), hLay, np.full((nOut), lastLayer)]

    layer = L
    order = layer.argsort()
    layer = layer[order]

    wMat = wMat[np.ix_(order, order)]
    nLayer = layer[-1]

    # Convert wMat to Full Network Graph
    # rows, cols = np.where(wMat != 0)
    rows, cols = np.where(~np.isnan(wMat))
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G, layer


def getNodeCoord(G, layer, taskName):
    env = games[taskName]

    # Calculate positions of input and output
    nIn = env.input_size + 1
    nOut = env.output_size
    nNode = len(G.nodes)
    fixed_pos = np.empty((nNode, 2))
    fixed_nodes = np.r_[np.arange(0, nIn), np.arange(nNode - nOut, nNode)]

    # Set Figure dimensions
    fig_wide = 10
    fig_long = 5

    # Assign x and y coordinates per layer
    x = np.ones((1, nNode)) * layer  # Assign x coord by layer
    x = (x / np.max(x)) * fig_wide  # Normalize

    _, nPerLayer = np.unique(layer, return_counts=True)

    y = cLinspace(-2, fig_long + 2, nPerLayer[0])
    for i in range(1, len(nPerLayer)):
        if i % 2 == 0:
            y = np.r_[y, cLinspace(0, fig_long, nPerLayer[i])]
        else:
            y = np.r_[y, cLinspace(-1, fig_long + 1, nPerLayer[i])]

    fixed_pos = np.c_[x.T, y.T]
    pos = dict(enumerate(fixed_pos.tolist()))

    return pos


def labelInOut(pos, env, ax=None):
    nIn = env.input_size + 1
    nOut = env.output_size
    nNode = len(pos)
    fixed_nodes = np.r_[np.arange(0, nIn), np.arange(nNode - nOut, nNode)]

    if len(env.in_out_labels) > 0:
        stateLabels = ['bias'] + env.in_out_labels
        labelDict = {}
    for i in range(len(stateLabels)):
        labelDict[fixed_nodes[i]] = stateLabels[i]

    for i in range(nIn):
        if not ax:
            plt.annotate(labelDict[i], xy=(pos[i][0] - 0.5, pos[i][1]), xytext=(pos[i][0] - 2.5, pos[i][1] - 0.5), \
                         arrowprops=dict(arrowstyle="->", color='k', connectionstyle="angle"))
        else:
            ax.annotate(labelDict[i], xy=(pos[i][0] - 0.5, pos[i][1]), xytext=(pos[i][0] - 2.5, pos[i][1] - 0.5), \
                        arrowprops=dict(arrowstyle="->", color='k', connectionstyle="angle"))

    for i in range(nNode - nOut, nNode):
        if not ax:
            plt.annotate(labelDict[i], xy=(pos[i][0] + 0.1, pos[i][1]), xytext=(pos[i][0] + 1.5, pos[i][1] + 1.0), \
                         arrowprops=dict(arrowstyle="<-", color='k', connectionstyle="angle"))
        else:
            ax.annotate(labelDict[i], xy=(pos[i][0] + 0.1, pos[i][1]), xytext=(pos[i][0] + 1.5, pos[i][1] + 1.0), \
                        arrowprops=dict(arrowstyle="<-", color='k', connectionstyle="angle"))


def drawNodeLabels(G, pos, aVec, ax=None):
    actLabel = np.array((['', '( + )', '(0/1)', '(sin)', '(gau)', '(tanh)', \
                          '(sig)', '( - )', '(abs)', '(relu)', '(cos)', '(sqr)']))
    listLabel = actLabel[aVec.astype(int)]
    label = dict(enumerate(listLabel))
    nx.draw_networkx_labels(G, pos, labels=label, ax=ax)


def drawEdge(G, pos, wMat, layer, ax=None):
    # wMat[np.isnan(wMat)]=0
    # Organize edges by layer
    _, nPerLayer = np.unique(layer, return_counts=True)
    edgeLayer = []
    layBord = np.cumsum(nPerLayer)
    for i in range(0, len(layBord)):
        tmpMat = np.copy(wMat)
        start = layBord[-i]
        end = layBord[-i + 1]
        # tmpMat[:,:start] *= 0
        # tmpMat[:,end:] *= 0
        # rows, cols = np.where(tmpMat != 0)
        tmpMat[:, :start] = np.nan
        tmpMat[:, end:] = np.nan
        rows, cols = np.where(~np.isnan(tmpMat))
        edges = zip(rows.tolist(), cols.tolist())
        edgeLayer.append(nx.DiGraph())
        edgeLayer[-1].add_edges_from(edges)
    edgeLayer.append(edgeLayer.pop(0))  # move first layer to correct position

    # Layer Colors
    for i in range(len(edgeLayer)):
        C = [i / len(edgeLayer)] * len(edgeLayer[i].edges)
        nx.draw_networkx_edges(G, pos, edgelist=edgeLayer[i].edges, \
                               alpha=.75, width=1.0, edge_color=C, edge_cmap=plt.cm.viridis, \
                               edge_vmin=0.0, edge_vmax=1.0, arrowsize=8, ax=ax)


def getLayer(wMat):
    '''
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1
    '''
    wMat = wMat.copy()
    wMat[~np.isnan(wMat)] = 1
    wMat[np.isnan(wMat)] = 0
    # wMat[wMat!=0]=1
    nNode = np.shape(wMat)[0]
    layer = np.zeros((nNode))
    while (True):  # Loop until sorting doesn't help any more
        prevOrder = np.copy(layer)
        for curr in range(nNode):
            srcLayer = np.zeros((nNode))
            for src in range(nNode):
                srcLayer[src] = layer[src] * wMat[src, curr]
            layer[curr] = np.max(srcLayer) + 1
        if all(prevOrder == layer):
            break
    return layer - 1


def cLinspace(start, end, N):
    if N == 1:
        return np.mean([start, end])
    else:
        return np.linspace(start, end, N)


def lload(fileName):
    return np.loadtxt(fileName, delimiter=',')