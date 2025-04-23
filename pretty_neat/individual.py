import numpy as np
from .agnostic_net import getLayer, getNodeOrder


def listXor(b,c):
    """Returns elements in lists b and c they don't share
    """
    A = [a for a in b+c if (a not in b) or (a not in c)]
    return A


class Ind():
    """Individual class: genes, network, and fitness
    """
    def __init__(self, conn, node):
        """Intialize individual with given genes
        Args:
          conn - [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == Weight
                 [4,:] == Enabled?
          node - [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)

        Attributes:
          node    - (np_array) - node genes (see args)
          conn    - (np_array) - conn genes (see args)
          nInput  - (int)      - number of inputs
          nOutput - (int)      - number of outputs
          wMat    - (np_array) - weight matrix, one row and column for each node
                    [N X N]    - rows: connection from; cols: connection to
          wVec    - (np_array) - wMat as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node (as int)
                    [N X 1]
          nConn   - (int)      - number of connections
          fitness - (double)   - fitness averaged over all trials (higher better)
          X fitMax  - (double)   - best fitness over all trials (higher better)
          rank    - (int)      - rank in population (lower better)
          birth   - (int)      - generation born
          species - (int)      - ID of species
        """
        self.node    = np.copy(node)
        self.conn    = np.copy(conn)
        self.nInput  = sum(node[1,:]==1)
        self.nOutput = sum(node[1,:]==2)
        self.wMat    = []
        self.wVec    = []
        self.aVec    = []
        self.nConn   = []
        self.fitness = [] # Mean fitness over trials
        #self.fitMax  = [] # Best fitness over trials
        self.rank    = []
        self.birth   = []
        self.species = []

    def nConns(self):
        """Returns number of active connections
        """
        return int(np.sum(self.conn[4,:]))

    def express(self):
        """Converts genes to weight matrix and activation vector
        """
        order, wMat = getNodeOrder(self.node, self.conn)
        assert order is not False, 'Topological sort failed'
        self.wMat = wMat
        self.aVec = self.node[2,order]
        wVec = self.wMat.flatten()
        self.nConn = np.sum(~np.isnan(wVec))
        wVec[np.isnan(wVec)] = 0
        self.wVec  = wVec

    def impress(self, wVec):
        order, _ = getNodeOrder(self.node, self.conn)
        assert order is not False, 'Topological sort failed'
        wMat = wVec.reshape(np.shape(self.wMat))
        node_perm = self.node[0,order]
        for i in range(len(self.conn[0])):
            value = wMat[np.where(node_perm==self.conn[1,i])[0][0],np.where(node_perm==self.conn[2,i])[0][0]]
            if self.conn[4,i] == 1:
                self.conn[3,i] = value
        self.express()

    def createChild(self, p, innov, gen=0, mate=None):
        """Create new individual with this individual as a parent

          Args:
            p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
            innov  - (np_array) - innovation record
               [5 X nUniqueGenes]
               [0,:] == Innovation Number
               [1,:] == Source
               [2,:] == Destination
               [3,:] == New Node?
               [4,:] == Generation evolved
            gen    - (int)      - (optional) generation (for innovation recording)
            mate   - (Ind)      - (optional) second for individual for crossover


        Returns:
            child  - (Ind)      - newly created individual
            innov  - (np_array) - updated innovation record

        """
        if mate is not None:
            child = self.crossover(mate)
        else:
            child = Ind(self.conn, self.node)

        child, innov = child.mutate(p,innov,gen)

        return child, innov

    # -- Canonical NEAT recombination operators ------------------------------ -- #

    def crossover(self,mate):
        """Combine genes of two individuals to produce new individual

          Procedure:
          ) Inherit all nodes and connections from most fit parent
          ) Identify matching connection genes in parentA and parentB
          ) Replace weights with parentB weights with some probability

          Args:
            parentA  - (Ind) - Fittest parent
              .conns - (np_array) - connection genes
                       [5 X nUniqueGenes]
                       [0,:] == Innovation Number (unique Id)
                       [1,:] == Source Node Id
                       [2,:] == Destination Node Id
                       [3,:] == Weight Value
                       [4,:] == Enabled?
            parentB - (Ind) - Less fit parent

        Returns:
            child   - (Ind) - newly created individual

        """
        parentA = self
        parentB = mate

        # # Inherit all nodes and connections from most fit parent
        # child = Ind(parentA.conn, parentA.node)

        # # Identify matching connection genes in ParentA and ParentB
        # aConn = np.copy(parentA.conn[0,:])
        # bConn = np.copy(parentB.conn[0,:])
        # matching, IA, IB = np.intersect1d(aConn,bConn,return_indices=True)

        # # Replace weights with parentB weights with some probability
        # bProb = 0.5
        # bGenes = np.random.rand(1,len(matching))<bProb
        # child.conn[3,IA[bGenes[0]]] = parentB.conn[3,IB[bGenes[0]]]

        # return child

        # < ---- True NEAT Crossover ---- >
        connA, nodeA = parentA.conn, parentA.node
        connB, nodeB = parentB.conn, parentB.node

        connChild = np.empty((5,0))
        nodeChild = np.empty((3,0))

        nodeAId, nodeBId = nodeA[0,:], nodeB[0,:]
        overlapNode = np.intersect1d(nodeAId,nodeBId)
        diffNode = np.setxor1d(nodeAId,nodeBId)
        allNode = np.concatenate((overlapNode,diffNode)) # no need to sort, np.intersect1d already sorted and nIns and nOuts are always the smallest
        overlapNode_len = len(overlapNode)
        for i in range(len(allNode)):
            id = allNode[i]
            aInd = np.where(nodeAId==id)[0]
            bInd = np.where(nodeBId==id)[0]
            if i < overlapNode_len:
                assert len(aInd) == len(bInd) == 1, f'Innovation record corrupted {aInd} {bInd}'
                if np.random.rand() < 0.5:
                    nodeChild = np.hstack((nodeChild,nodeA[:,aInd]))
                else:
                    nodeChild = np.hstack((nodeChild,nodeB[:,bInd]))
            else:
                if len(aInd) > 0:
                    assert len(bInd) == 0, f'Innovation record corrupted {aInd} {bInd}'
                    nodeChild = np.hstack((nodeChild,nodeA[:,aInd]))
                else:
                    assert len(aInd) == 0, f'Innovation record corrupted {aInd} {bInd}'
                    nodeChild = np.hstack((nodeChild,nodeB[:,bInd]))

        connAId, connBId = connA[0,:], connB[0,:]
        overlapConn = np.intersect1d(connAId,connBId)
        diffConnA = np.setdiff1d(connAId,connBId) # setdiffid only return different elements in first argument
        diffConnB = np.setdiff1d(connBId,connAId) # setdiffid only return different elements in first argument
        allConn = np.concatenate((overlapConn,diffConnA,diffConnB))
        overlapConn_len, diffConnA_len = len(overlapConn), len(diffConnA)
        for i in range(len(allConn)):
            id = allConn[i]
            aInd = np.where(connAId==id)[0]
            bInd = np.where(connBId==id)[0]
            if i < overlapConn_len:
                assert len(aInd) == len(bInd) == 1, f'Innovation record corrupted {aInd} {bInd}'
                if np.random.rand() < 0.5:
                    connChild = np.hstack((connChild,connA[:,aInd]))
                else:
                    connChild = np.hstack((connChild,connB[:,bInd]))
                if (connA[4,aInd[0]] == 0) and (connB[4,bInd[0]] == 0):
                    connChild[4,-1] = 0
                else:
                    connChild[4,-1] = 1
            elif i < overlapConn_len + diffConnA_len:
                assert len(aInd) == 1 and len(bInd) == 0, f'Innovation record corrupted {aInd} {bInd}'
                connChild = np.hstack((connChild,connA[:,aInd]))
            else:
                assert len(aInd) == 0 and len(bInd) == 1, f'Innovation record corrupted {aInd} {bInd}'
                connChild = np.hstack((connChild,connB[:,bInd]))

        order, _ = getNodeOrder(nodeChild, connChild)

        if order is False:
            connChild, nodeChild = connA, nodeA

        return Ind(connChild, nodeChild)

    def mutate(self,p,innov=None,gen=None):
        """Randomly alter topology and weights of individual

        Args:
          p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
          child    - (Ind) - individual to be mutated
            .conns - (np_array) - connection genes
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number (unique Id)
                     [1,:] == Source Node Id
                     [2,:] == Destination Node Id
                     [3,:] == Weight Value
                     [4,:] == Enabled?
            .nodes - (np_array) - node genes
                     [3 X nUniqueGenes]
                     [0,:] == Node Id
                     [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                     [2,:] == Activation function (as int)
          innov    - (np_array) - innovation record
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number
                     [1,:] == Source
                     [2,:] == Destination
                     [3,:] == New Node?
                     [4,:] == Generation evolved

        Returns:
            child   - (Ind)      - newly created individual
            innov   - (np_array) - innovation record

        """
        # Readability
        nConn = np.shape(self.conn)[1]
        connG = np.copy(self.conn)
        nodeG = np.copy(self.node)

        # - Weight mutation
        # [Canonical NEAT: 10% of weights are fully random...but seriously?]
        mutatedWeights = np.random.rand(1,nConn) < p['prob_mutConn'] # Choose weights to mutate
        weightChange = mutatedWeights * np.random.randn(1,nConn) * p['ann_mutSigma']
        connG[3,:] += weightChange[0]

        # - Re-enable connections
        # disabled  = np.where(connG[4,:] == 0)[0]
        # reenabled = np.random.rand(1,len(disabled)) < p['prob_enable']
        # connG[4,disabled] = reenabled

        # Clamp weight strength [ Warning given for nan comparisons ]
        connG[3, (connG[3,:] >  p['ann_absWCap'])] =  p['ann_absWCap']
        connG[3, (connG[3,:] < -p['ann_absWCap'])] = -p['ann_absWCap']

        if (np.random.rand() < p['prob_mutAct']):
            nodeG, innov = self.mutAct(connG, nodeG, innov, gen, p)

        if (np.random.rand() < p['prob_addNode']) and np.any(connG[4,:]==1):
            connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)

        if (np.random.rand() < p['prob_addConn']):
            connG, innov = self.mutAddConn(connG, nodeG, innov, gen, p)

        disabled = np.where(connG[4,:] == 0)[0]
        if len(disabled) > 0:
            selected = np.random.choice(disabled)
            connG[4,selected] = 1 if np.random.rand() < p['prob_enable'] else 0

        child = Ind(connG, nodeG)
        child.birth = gen

        return child, innov

    def mutAddNode(self, connG, nodeG, innov, gen, p):
        """Add new node to genome

        Args:
          connG    - (np_array) - connection genes
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number (unique Id)
                     [1,:] == Source Node Id
                     [2,:] == Destination Node Id
                     [3,:] == Weight Value
                     [4,:] == Enabled?
          nodeG    - (np_array) - node genes
                     [3 X nUniqueGenes]
                     [0,:] == Node Id
                     [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                     [2,:] == Activation function (as int)
          innov    - (np_array) - innovation record
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number
                     [1,:] == Source
                     [2,:] == Destination
                     [3,:] == New Node?
                     [4,:] == Generation evolved
          gen      - (int) - current generation
          p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


        Returns:
          connG    - (np_array) - updated connection genes
          nodeG    - (np_array) - updated node genes
          innov    - (np_array) - updated innovation record

        """
        assert innov is not None, 'Innovation record required for addNode mutation'

        newNodeId = int(max(innov[2,:])+1) # next node id is a running counter
        newConnId = innov[0,-1]+1

        # Choose connection to split
        connActive = np.where(connG[4,:] == 1)[0]
        if len(connActive) < 1:
            return connG, nodeG, innov # No active connections, nothing to split
        connSplit  = connActive[np.random.randint(len(connActive))]

        # Create new node
        newActivation = p['ann_actRange'][np.random.randint(len(p['ann_actRange']))]
        newNode = np.array([[newNodeId, 3, newActivation]]).T

        # Add connections to and from new node
        # -- Effort is taken to minimize disruption from node addition:
        # The 'weight to' the node is set to 1, the 'weight from' is set to the
        # original  weight. With a near linear activation function the change in performance should be minimal.

        connTo    = connG[:,connSplit].copy()
        connTo[0] = newConnId
        connTo[2] = newNodeId
        connTo[3] = 1 # weight set to 1

        connFrom    = connG[:,connSplit].copy()
        connFrom[0] = newConnId + 1
        connFrom[1] = newNodeId
        connFrom[3] = connG[3,connSplit] # weight set to previous weight value

        newConns = np.vstack((connTo,connFrom)).T

        # Disable original connection
        connG[4,connSplit] = 0

        # Record innovations
        newInnov = np.empty((5,2))
        newInnov[:,0] = np.hstack((connTo[0:3], newNodeId, gen))
        newInnov[:,1] = np.hstack((connFrom[0:3], -1, gen))
        innov = np.hstack((innov,newInnov))

        # Add new structures to genome
        nodeG = np.hstack((nodeG,newNode))
        connG = np.hstack((connG,newConns))

        return connG, nodeG, innov

    def mutAddConn(self, connG, nodeG, innov, gen, p):
        """Add new connection to genome.
        To avoid creating recurrent connections all nodes are first sorted into
        layers, connections are then only created from nodes to nodes of the same or
        later layers.


        Todo: check for preexisting innovations to avoid duplicates in same gen

        Args:
          connG    - (np_array) - connection genes
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number (unique Id)
                     [1,:] == Source Node Id
                     [2,:] == Destination Node Id
                     [3,:] == Weight Value
                     [4,:] == Enabled?
          nodeG    - (np_array) - node genes
                     [3 X nUniqueGenes]
                     [0,:] == Node Id
                     [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                     [2,:] == Activation function (as int)
          innov    - (np_array) - innovation record
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number
                     [1,:] == Source
                     [2,:] == Destination
                     [3,:] == New Node?
                     [4,:] == Generation evolved
          gen      - (int)      - current generation
          p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


        Returns:
          connG    - (np_array) - updated connection genes
          innov    - (np_array) - updated innovation record

        """
        assert innov is not None, 'Innovation record required for addConn mutation'

        newConnId = innov[0,-1]+1

        nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
        nOuts = len(nodeG[0,nodeG[1,:] == 2])
        order, wMat = getNodeOrder(nodeG, connG)   # Topological Sort of Network
        assert order is not False, 'Topological sort failed'
        hMat = wMat[nIns:-nOuts,nIns:-nOuts]
        hLay = getLayer(hMat)+1

        # To avoid recurrent connections nodes are sorted into layers, and connections are only allowed from lower to higher layers
        if len(hLay) > 0:
            lastLayer = max(hLay)+1
        else:
            lastLayer = 1
        L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer) ]
        nodeKey = np.c_[nodeG[0,order], L] # Assign Layers

        sources = np.random.permutation(len(nodeKey))
        for src in sources:
            srcLayer = nodeKey[src,1]
            if srcLayer == lastLayer:
                continue
            elif srcLayer == 0:
                dest = np.where(nodeKey[:,1] > srcLayer)[0]
            else:
                dest = np.where((nodeKey[:,1] >= srcLayer) & (nodeKey[:,0] != nodeKey[src,0]))[0]

                # Finding already existing connections:
            #   ) take all connection genes with this source (connG[1,:])
            #   ) take the destination of those genes (connG[2,:])
            #   ) convert to nodeKey index (Gotta be a better numpy way...)
            srcIndx = np.where(connG[1,:]==nodeKey[src,0])[0]
            exist = connG[2,srcIndx]
            existKey = []
            for iExist in exist:
                existKey.append(np.where(nodeKey[:,0]==iExist)[0])
            dest = np.setdiff1d(dest,existKey) # Remove existing connections

            # Add a random valid connection
            np.random.shuffle(dest)
            if len(dest)>0:  # (there is a valid connection)
                connNew = np.empty((5,1))
                connNew[0] = newConnId
                connNew[1] = nodeKey[src,0]
                connNew[2] = nodeKey[dest[0],0]
                connNew[3] = (np.random.rand()-0.5)*2*p['ann_absWCap']
                connNew[4] = 1
                # connG = np.c_[connG,connNew]

                # deduplicate innovations
                dup = False
                connInnov = innov[:,(innov[1,:]==connNew[1]) & (innov[2,:]==connNew[2])]
                if connInnov.shape[1] != 0:
                    assert connInnov.shape[1] == 1, 'Innovation record corrupted'
                    connNew[0] = connInnov[0,0] # connInnov could be smaller than connNew
                    assert connG[:,connG[0,:]==connNew[0]].shape[1] == 0, 'Innovation record corrupted'
                    dup = True

                connG = np.c_[connG,connNew]
                # Record innovation
                if not dup:
                    newInnov = np.hstack((connNew[0:3].flatten(), -1, gen))
                    innov = np.hstack((innov,newInnov[:,None]))
                break

        return connG, innov

    def mutAct(self, connG, nodeG, innov, gen, p):
        """Randomly alter activation function of a node

        Args:
          child    - (Ind) - individual to be mutated
            .conns - (np_array) - connection genes
                    [5 X nUniqueGenes]
                    [0,:] == Innovation Number (unique Id)
                    [1,:] == Source Node Id
                    [2,:] == Destination Node Id
                    [3,:] == Weight Value
                    [4,:] == Enabled?
            .nodes - (np_array) - node genes
                    [3 X nUniqueGenes]
                    [0,:] == Node Id
                    [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                    [2,:] == Activation function (as int)
          innov    - (np_array) - innovation record
                    [5 X nUniqueGenes]
                    [0,:] == Innovation Number
                    [1,:] == Source
                    [2,:] == Destination
                    [3,:] == New Node?
                    [4,:] == Generation evolved

        Returns:
            child   - (Ind)      - newly created individual
            innov   - (np_array) - innovation record

        """

        nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
        nOuts = len(nodeG[0,nodeG[1,:] == 2])

        # Mutate Activation
        start = nIns+nOuts
        end = nodeG.shape[1]
        if start != end:
            mutNode = np.random.randint(start,end)
            newActPool = listXor([int(nodeG[2,mutNode])], list(p['ann_actRange']))
            nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])

        return nodeG, innov