import copy
import json
import itertools
import math
import numpy as np

from task import *
from utils.neat_utils import *
from .individual import Ind


class Species():
    """Species class, only contains fields: all methods belong to the NEAT class.
    Note: All 'species' related functions are part of the Neat class, though defined in this file.
    """

    def __init__(self,seed):
        """Intialize species around a seed
        Args:
          seed - (Ind) - individual which anchors seed in compatibility space

        Attributes:
          seed       - (Ind)   - individual who acts center of species
          members    - [Ind]   - individuals in species
          bestInd    - (Ind)   - highest fitness individual ever found in species
          bestFit    - (float) - highest fitness ever found in species
          lastImp    - (int)   - generations since a new best individual was found
          num_offspring - (int)   - new individuals to create this generation
        """
        self.seed = seed      # Seed is type Ind
        self.members = [seed] # All inds in species
        self.bestFit = seed.fitness
        self.lastImp = 0
        self.num_offspring = []


class Neat():
    """NEAT main class. Evolves population given fitness values of individuals.
    """
    def __init__(self, hyp):
        """Intialize NEAT algorithm with hyperparameters
        Args:
          hyp - (dict) - algorithm hyperparameters

        Attributes:
          p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
          population     - (Ind)      - Current population
          species - (Species)  - Current species
          innov   - (np_array) - innovation record
                    [5 X nUniqueGenes]
                    [0,:] == Innovation Number
                    [1,:] == Source
                    [2,:] == Destination
                    [3,:] == New Node?
                    [4,:] == Generation evolved
          gen     - (int)      - Current generation
        """
        self.p       = hyp
        self.population     = []
        self.species = []
        self.innov   = []
        self.gen     = 0

    def ask(self):
        """Returns newly evolved population
        """
        if len(self.population) == 0:
            self.initPop()      # Initialize population
        else:
            self.gen += 1
            self.probMoo()      # Rank population according to objectivess
            self.speciate()     # Divide population into species
            self.evolvePop()    # Create child population

        return self.population       # Send child population for evaluation

    def tell(self,reward, wVec=None):
        """Assigns fitness to current population

        Args:
          reward - (np_array) - fitness value of each individual
                   [nInd X 1]

        """
        for i in range(np.shape(reward)[0]):
            self.population[i].fitness = reward[i]
            self.population[i].nConn   = self.population[i].nConn
            if wVec is not None:
                self.population[i].impress(wVec[i])

    def initPop(self):
        """Initialize population with a list of random individuals
        """
        ##  Create base individual
        p = self.p # readability

        # - Create Nodes -
        nodeId = np.arange(0,p['ann_nInput']+ p['ann_nOutput']+1,1)
        node = np.empty((3,len(nodeId)))
        node[0,:] = nodeId

        # Node types: [1:input, 2:output, 3:hidden, 4:bias]
        node[1,0]             = 4 # Bias
        node[1,1:p['ann_nInput']+1] = 1 # Input Nodes
        node[1,(p['ann_nInput']+1): \
               (p['ann_nInput']+p['ann_nOutput']+1)]  = 2 # Output Nodes

        # Node Activations
        node[2,:] = p['ann_initAct']
        # - Create Conns -
        nConn = (p['ann_nInput']+1) * p['ann_nOutput']
        ins   = np.arange(0,p['ann_nInput']+1,1)            # Input and Bias Ids
        outs  = (p['ann_nInput']+1) + np.arange(0,p['ann_nOutput']) # Output Ids

        conn = np.empty((5,nConn,))
        conn[0,:] = np.arange(0,nConn,1)      # Connection Id
        conn[1,:] = np.tile(ins, len(outs))   # Source Nodes
        conn[2,:] = np.repeat(outs,len(ins) ) # Destination Nodes
        conn[3,:] = np.nan                    # Weight Values
        conn[4,:] = 1                         # Enabled?

        # Create population of individuals with varied weights
        population = []
        for i in range(p['popSize']):
            newInd = Ind(conn, node)
            newInd.conn[3,:] = 2*(np.random.rand(1,nConn)-0.5) * p['ann_mutSigma']
            newInd.conn[4,:] = np.random.rand(1,nConn) < p['prob_initEnable']
            newInd.express()
            newInd.birth = 0
            population.append(copy.deepcopy(newInd))

        # - Create Innovation Record -
        innov = np.zeros([5,nConn])
        innov[0:3,:] = population[0].conn[0:3,:]
        innov[3,:] = -1

        self.population = population
        self.innov = innov

    def evolvePop(self):
        """ Evolves new population from existing species.
        Wrapper which calls 'recombine' on every species and combines all offspring into a new population. When speciation is not used, the entire population is treated as a single species.
        """
        new_population = []
        for i in range(len(self.species)):
            children, self.innov = self.recombine(self.species[i], self.innov, self.gen)
            new_population.append(children)
        self.population = list(itertools.chain.from_iterable(new_population))



    def probMoo(self):
        """Rank population according to Pareto dominance.
        """
        # Compile objectives
        meanFit = np.asarray([ind.fitness for ind in self.population])
        nConns  = np.asarray([ind.nConn   for ind in self.population])
        nConns[nConns==0] = 1 # No connections is pareto optimal but boring...
        objVals = np.c_[meanFit,1/nConns] # Maximize

        # Alternate between two objectives and single objective
        if self.p['alg_probMoo'] < np.random.rand():
            rank = nsga_sort(objVals[:,[0,1]])
        else: # Single objective
            rank = rankArray(-objVals[:,0])

        # Assign ranks
        for i in range(len(self.population)):
            self.population[i].rank = rank[i]


    def recombine(self, species, innov, gen):
        """ Creates next generation of child solutions from a species

        Procedure:
          ) Sort all individuals by rank
          ) Eliminate lower percentage of individuals from breeding pool
          ) Pass upper percentage of individuals to child population unchanged
          ) Select parents by tournament selection
          ) Produce new population through crossover and mutation

        Args:
            species - (Species) -
              .members    - [Ind] - parent population
              .num_offspring - (int) - number of children to produce
            innov   - (np_array)  - innovation record
                      [5 X nUniqueGenes]
                      [0,:] == Innovation Number
                      [1,:] == Source
                      [2,:] == Destination
                      [3,:] == New Node?
                      [4,:] == Generation evolved
            gen     - (int) - current generation

        Returns:
            children - [Ind]      - newly created population
            innov   - (np_array)  - updated innovation record

        """
        p = self.p
        num_offspring = int(species.num_offspring)
        population = species.members
        children = []

        # Sort by rank
        population.sort(key=lambda x: x.rank)

        # Cull  - eliminate worst individuals from breeding pool
        numberToCull = int(np.floor(p['select_cullRatio'] * len(population)))
        if numberToCull > 0:
            population[-numberToCull:] = []

            # Elitism - keep best individuals unchanged
        # nElites = int(np.floor(len(population)*p['select_eliteRatio']))
        nElites = int(np.ceil(len(population)*p['select_eliteRatio']))
        for i in range(nElites):
            children.append(population[i])
            num_offspring -= 1

        if num_offspring > 0: #TODO: why is this necessary?
            # Get parent pairs via tournament selection
            # -- As individuals are sorted by fitness, index comparison is
            # enough. In the case of ties the first individual wins

            parentA = np.random.randint(len(population),size=(num_offspring,p['select_tournSize']))
            parentB = np.random.randint(len(population),size=(num_offspring,p['select_tournSize']))
            parents = np.vstack( (np.min(parentA,1), np.min(parentB,1) ) )
            parents = np.sort(parents,axis=0) # Higher fitness parent first

            # Breed child population
            for i in range(num_offspring):
                if np.random.rand() > p['prob_crossover']:
                    # Mutation only: take only highest fit parent
                    child, innov = population[parents[0,i]].createChild(p,innov,gen)
                else:
                    # Crossover
                    child, innov = population[parents[0,i]].createChild(p,innov,gen, \
                                                                 mate=population[parents[1,i]])

                child.express()
                children.append(child)

        return children, innov

    def speciate(self):
        """Divides population into species and assigns each a number of offspring/
        """
        # Readbility
        p = self.p
        pop = self.pop
        species = self.species

        if p['alg_speciate'] == 'neat':
            # Adjust species threshold to track desired number of species
            if len(species) > p['spec_target']:
                p['spec_thresh'] += p['spec_compatMod']

            if len(species) < p['spec_target']:
                p['spec_thresh'] -= p['spec_compatMod']

            if p['spec_thresh'] < p['spec_threshMin']:
                p['spec_thresh'] = p['spec_threshMin']

            species, pop = self.assignSpecies  (species, pop, p)
            species      = self.assignOffspring(species, pop, p)

        elif p['alg_speciate'] == "none":
            # Recombination takes a species, when there is no species we dump the whole population into one species that is awarded all offspring
            species = [Species(pop[0])]
            species[0].num_offspring = p['popSize']
            for ind in pop:
                ind.species = 0
            species[0].members = pop

        # Update
        self.p = p
        self.pop = pop
        self.species = species

    def assignSpecies(self, species, pop, p):
        """Assigns each member of the population to a species.
        Fills each species class with nearests members, assigns a species Id to each
        individual for record keeping

        Args:
          species - (Species) - Previous generation's species
            .seed       - (Ind) - center of species
          pop     - [Ind]     - unassigned individuals
          p       - (Dict)    - algorithm hyperparameters

        Returns:
          species - (Species) - This generation's species
            .seed       - (Ind) - center of species
            .members    - [Ind] - parent population
          pop     - [Ind]     - individuals with species ID assigned

        """

        # Get Previous Seeds
        if len(species) == 0:
            # Create new species if none exist
            species = [Species(pop[0])]
            species[0].num_offspring = p['popSize']
            species[0].members = []
        else:
            unspeciated = set(range(len(pop)))
            for iSpec in range(len(species)):
                candidates = []
                for gid in unspeciated:
                    g = pop[gid]
                    d = self.compatDist(species[iSpec].seed.conn, g.conn)
                    candidates.append((d, gid))
                # The new representative is the genome closest to the current representative.
                _, new_seed_id = min(candidates, key=lambda x: x[0])
                new_seed = pop[new_seed_id]
                species[iSpec].seed = new_seed
                species[iSpec].members = []
                unspeciated.remove(new_seed_id)

        assert p['spec_thresh'] > 0, "ERROR: Species threshold must be positive"
        # Assign members of population to first species within compat distance
        beyond_dist = 0
        for i in range(len(pop)):
            candidates = []
            iSpec = 0
            while iSpec < len(species):
                ref = np.copy(species[iSpec].seed.conn)
                ind = np.copy(pop[i].conn)
                cDist = self.compatDist(ref,ind)
                candidates.append((cDist, iSpec))
                iSpec += 1
            # find best species to assign to
            min_cDist, best_iSpec = min(candidates, key=lambda x: x[0])
            if min_cDist < p['spec_thresh']:
                pop[i].species = best_iSpec
                species[best_iSpec].members.append(pop[i])
            elif len(species) >= p['spec_target']:
                pop[i].species = best_iSpec
                species[best_iSpec].members.append(pop[i])
                beyond_dist += min_cDist - p['spec_thresh']
            # If no seed is close enough, start your own species
            else:
                pop[i].species = iSpec
                species.append(Species(pop[i]))

        p['spec_thresh'] += beyond_dist / p['popSize']

        return species, pop

    def assignOffspring(self, species, pop, p):
        """Assigns number of offspring to each species based on fitness sharing.
        NOTE: Ordinal rather than the cardinal fitness of canonical NEAT is used.

        Args:
          species - (Species) - this generation's species
            .members    - [Ind]   - individuals in species
          pop     - [Ind]     - individuals with species assigned
            .fitness    - (float) - performance on task (higher is better)
          p       - (Dict)    - algorithm hyperparameters

        Returns:
          species - (Species) - This generation's species
            .num_offspring - (int) - number of children to produce
        """

        nSpecies = len(species)
        if nSpecies == 1:
            species[0].num_offspring = p['popSize']
        else:
            # -- Fitness Sharing
            # Rank all individuals
            popFit = np.asarray([ind.fitness for ind in pop])
            popRank = tiedRank(popFit)
            smoothing = p['spec_smooth'] if 'spec_smooth' in p and p['spec_smooth'] >= 0 else 1
            if p['select_rankWeight'] == 'exp':
                # rankScore = 1/popRank
                rankScore = np.exp(-popRank*smoothing)
            elif p['select_rankWeight'] == 'lin':
                rankScore = 1+abs(popRank-len(popRank))*smoothing
            else:
                print("Invalid rank weighting (using linear)")
                rankScore = 1+abs(popRank-len(popRank))*smoothing
            specId = np.asarray([ind.species for ind in pop])

            # Best and Average Fitness of Each Species
            speciesFit = np.zeros((nSpecies,1))
            speciesTop = np.zeros((nSpecies,1))
            for iSpec in range(nSpecies):
                if not np.any(specId==iSpec):
                    speciesFit[iSpec] = 0
                else:
                    speciesFit[iSpec] = np.mean(rankScore[specId==iSpec])
                    bestId = np.argmax(popFit[specId==iSpec])
                    speciesTop[iSpec] = species[iSpec].members[bestId].fitness

                    # Did the species improve?
                    if speciesTop[iSpec] > species[iSpec].bestFit:
                        species[iSpec].bestFit = speciesTop[iSpec]
                        species[iSpec].lastImp = 0
                    else:
                        species[iSpec].lastImp += 1

                    # Stagnant species don't recieve species fitness
                    if species[iSpec].lastImp > p['spec_dropOffAge']:
                        speciesFit[iSpec] = 0

            # -- Assign Offspring
            if sum(speciesFit) == 0 or sum(speciesTop) == 0:
                speciesFit = np.ones((nSpecies,1))
                print("WARN: Entire population stagnant, continuing without extinction")

            offspring = bestIntSplit(speciesFit, p['popSize'])
            for iSpec in range(nSpecies):
                species[iSpec].num_offspring = offspring[iSpec]

        # Extinction
        species[:] = [s for s in species if s.num_offspring != 0]

        return species

    def compatDist(self, ref, ind):
        """Calculate 'compatiblity distance' between to genomes

        Args:
          ref - (np_array) -  reference genome connection genes
                [5 X nUniqueGenes]
                [0,:] == Innovation Number (unique Id)
                [3,:] == Weight Value
          ind - (np_array) -  genome being compared
                [5 X nUniqueGenes]
                [0,:] == Innovation Number (unique Id)
                [3,:] == Weight Value

        Returns:
          dist - (float) - compatibility distance between genomes
        """

        # Find matching genes
        IA, IB = quickINTersect(ind[0,:].astype(int),ref[0,:].astype(int))

        # Calculate raw genome distances
        ind[3,np.isnan(ind[3,:])] = 0
        ref[3,np.isnan(ref[3,:])] = 0
        weightDiff = abs(ind[3,IA] - ref[3,IB])
        geneDiff   = sum(np.invert(IA)) + sum(np.invert(IB))

        # Normalize and take weighted sum
        nInitial = self.p['ann_nInput'] + self.p['ann_nOutput']
        longestGenome = max(len(IA),len(IB)) - nInitial
        weightDiff = np.mean(weightDiff)
        geneDiff   = geneDiff   / (1+longestGenome) # this can be bigger than 1 but less than 2

        dist = geneDiff   * self.p['spec_geneCoef'] \
               + weightDiff * self.p['spec_weightCoef']
        return dist