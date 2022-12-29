import os
import argparse
import numpy as np
import scipy.sparse
import utilities
from itertools import combinations
import numpy as np

import scipy.stats
from numpy.random import default_rng
from numpy import meshgrid, array, random


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Er random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_indset(graph, filename):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")


def generate_indsetnew(graph, filename):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")


        
        
        
def generate_setcover(nrows, ncols, density, filename, rng, max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))

        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([f" +1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))


def generate_capacited_facility_location(random, filename, n_customers, n_facilities, ratio, param_fixed_cost_sample=90):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(5, 35+1, size=n_customers)
    capacities = rng.randint(10, 160+1, size=n_facilities)
    
    
    
    fixed_costs = rng.randint(param_fixed_cost_sample, param_fixed_cost_sample+5, size=n_facilities) * np.sqrt(capacities) \
    
    
   
    fixed_costs = fixed_costs.astype(int)
 

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))

        
        
        
def generate_capacited_facility_location_vary_demand_maxfacopen(random, filename, n_customers, n_facilities, ratio,dem_low, dem_high, cap_low=10,  cap_high=160, maxfacopen=None):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(dem_low, dem_high+1, size=n_customers)
    capacities = rng.randint(cap_low, cap_high+1, size=n_facilities)
    
    
    fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + rng.randint(90+1, size=n_facilities)
    
    
    
    
    fixed_costs = fixed_costs.astype(int)
    print('demands', demands)

    total_demand = demands.sum()
    total_capacity = capacities.sum()
    
    print('total_demand', total_demand)
    print('total_cap', total_capacity)
    print('capacities', capacities)
    

    
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    print('capacities after', capacities)
    print('total_capacity after', total_capacity)
    
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")

        file.write("\n maxopenfac: "+ "".join([f" +1 y_{j+1}" for j in range(n_facilities)]) + " <= " + str( maxfacopen)) 
                
        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

             
                
                
        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))        
        
def generate_capacited_facility_location_vary_demand(random, filename, n_customers, n_facilities, ratio,dem_low, dem_high, cap_low=10,  cap_high=160):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(dem_low, dem_high+1, size=n_customers)
#     
    capacities = rng.randint(cap_low, cap_high+1, size=n_facilities)
    
    
    fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + rng.randint(90+1, size=n_facilities)
    
  
    
    fixed_costs = fixed_costs.astype(int)
    print('demands', demands)

    total_demand = demands.sum()
    total_capacity = capacities.sum()
    
    print('total_demand', total_demand)
    print('total_cap', total_capacity)
    print('capacities', capacities)
    

    
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    print('capacities after', capacities)
    print('total_capacity after', total_capacity)
    
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))

def generate_capacited_facility_location_vary_demand_no_fixed_cost(random, filename, n_customers, n_facilities, ratio,dem_low, dem_high):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    # demands = rng.randint(5, 35+1, size=n_customers)
    demands = rng.randint(dem_low, dem_high+1, size=n_customers)
    capacities = rng.randint(10, 160+1, size=n_facilities)
    
  

    total_demand = demands.sum()
    total_capacity = capacities.sum()
    
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

  
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'facvers', 'faccost', 'tsp', 'setcover_densize','faccostbig', 'facdem','facdemnofix', 'facdemmaxfacopen', 'bmatch','indsetnewerdos','indsetnewba'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=utilities.valid_seed,
        default=0,
    )
    
    parser.add_argument(
        '-density', '--density',
        help='density',
        type=float,
        default=None,
    )
    
    parser.add_argument(
        '-affinity', '--affinity',
        help='affinity',
        type=int,
        default=None,
    )
    
    parser.add_argument(
        '-edge_prob', '--edge_prob',
        help='edge_prob',
        type=float,
        default=None,
    )
    
    parser.add_argument(
        '-indnodes', '--indnodes',
        help='indnodes',
        type=int,
        default=None,
    )
    parser.add_argument(
        '-faccost', '--faccost',
        help='faccost',
        type=int,
        default=None,
    )
    
    
    parser.add_argument(
        '-add_item_prob', '--add_item_prob',
        help='add_item_prob',
        type=float,
        default=None,
    )
    
    parser.add_argument(
        '-facdemlow', '--facdemlow',
        help='facdemlow',
        type=int,
        default=None,
    )


    parser.add_argument(
        '-facdemhigh', '--facdemhigh',
        help='facdemhigh',
        type=int,
        default=None,
    )


    parser.add_argument(
        '-facdemcaplow', '--facdemcaplow',
        help='facdemcaplow',
        type=int,
        default=None,
    )


    parser.add_argument(
        '-facdemcaphigh', '--facdemcaphigh',
        help='facdemcaphigh',
        type=int,
        default=None,
    )
    
    parser.add_argument(
        '-facmaxopen', '--facmaxopen',
        help='facmaxopen',
        type=int,
        default=None,
    )


    
    
    parser.add_argument(
        '-number_of_facilities', '--number_of_facilities',
        help='number_of_facilities',
        type=int,
        default=None,
    )
    
    
    
    
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)


    if args.problem == 'indsetnewba':
        # number_of_nodes = 750
        number_of_nodes = args.indnodes
        # affinity = 4
        affinity = args.affinity
        print('affinity', affinity)
        
    

        filenames = []
        nnodess = []

        # train instances
        n = 10000
        lp_dir = f'data/instances/indsetnewba_{affinity}/train_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/indsetnewba_{affinity}/valid_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # small transfer instances
        n = 100
        # number_of_nodes = 750
        lp_dir = f'data/instances/indsetnewba_{affinity}/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

      
        # test instances
        n = 100
        # number_of_nodes = 750
        lp_dir = f'data/instances/indsetnewba_{affinity}/test_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

   

        # actually generate the instances
        for filename, nnodes in zip(filenames, nnodess):
            print(f"  generating file {filename} ...")
            graph = Graph.barabasi_albert(nnodes, affinity, rng)
            generate_indset(graph, filename)
 

       
    if args.problem == 'facdem':
        number_of_customers = 100
        number_of_facilities = 100
        ratio = 5
        
        facdemlow = args.facdemlow
        facdemhigh = args.facdemhigh
        facdemcaplow = args.facdemcaplow
        facdemcaphigh = args.facdemcaphigh
        
        print('facdemlow = args.facdemlow', facdemlow )
        print('facdemhigh = args.facdemhigh', facdemhigh )
        
        print('facdemcaplow = args.facdemcaplow', facdemcaplow )
        print('facdemcaphigh = args.facdemcaphigh', facdemcaphigh )
        
        # faccost = args.faccost
        
        # number_of_customers = args.number_of_customers
        # number_of_facilities = args.number_of_facilities
        
        print('number_of_customers', number_of_customers)
        print('number_of_facilities', number_of_facilities)
        
        # ratio = args.ratio
        print('ratio', ratio)
        filenames = []
        ncustomerss = []
        nfacilitiess = []
        ratios = []

        # train instances
        # n = 10000
        n = 10000
        lp_dir = f'data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/train_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/valid_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # small transfer instances
        n = 100
        # number_of_customers = 100
        # number_of_facilities = 100
        lp_dir = f'data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # medium transfer instances
        n = 100
        number_of_customers = 200
        number_of_customers = 120
        
        lp_dir = f'data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # big transfer instances
        n = 100
        number_of_customers = 400
        
        number_of_customers = 200
        lp_dir = f'data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # test instances
        n = 2000
        number_of_customers = 100
        number_of_facilities = 100
        lp_dir = f'data/instances/facdem_{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}/test_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # actually generate the instances
        for filename, ncs, nfs, r in zip(filenames, ncustomerss, nfacilitiess, ratios):
            print(f"  generating file {filename} ...")
            generate_capacited_facility_location_vary_demand(rng, filename, n_customers=ncs, n_facilities=nfs, ratio=r, dem_low=facdemlow, dem_high=facdemhigh, cap_low=facdemcaplow, cap_high=facdemcaphigh)

        print("done.")
        
        
    if args.problem == 'facdemmaxfacopen':
        number_of_customers = 100
        number_of_facilities = 100
        ratio = 5
        
        facdemlow = args.facdemlow
        facdemhigh = args.facdemhigh
        facdemcaplow = args.facdemcaplow
        facdemcaphigh = args.facdemcaphigh
        
        facmaxopen = args.facmaxopen
        
        print('facdemlow = args.facdemlow', facdemlow )
        print('facdemhigh = args.facdemhigh', facdemhigh )
        
        print('facdemcaplow = args.facdemcaplow', facdemcaplow )
        print('facdemcaphigh = args.facdemcaphigh', facdemcaphigh )
        
        # faccost = args.faccost
        
        # number_of_customers = args.number_of_customers
        # number_of_facilities = args.number_of_facilities
        
        print('number_of_customers', number_of_customers)
        print('number_of_facilities', number_of_facilities)
        
        # ratio = args.ratio
        print('ratio', ratio)
        filenames = []
        ncustomerss = []
        nfacilitiess = []
        ratios = []

        # train instances
        # n = 10000
        n = 10000
        lp_dir = f'data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{facmaxopen}/train_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{facmaxopen}/valid_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # small transfer instances
        n = 100
        # number_of_customers = 100
        # number_of_facilities = 100
        lp_dir = f'data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{facmaxopen}/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # medium transfer instances
        n = 100
        number_of_customers = 200
        number_of_customers = 120
        
        lp_dir = f'data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{facmaxopen}/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # big transfer instances
        n = 100
        number_of_customers = 400
        
        number_of_customers = 200
        lp_dir = f'data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{facmaxopen}/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # test instances
        n = 2000
        number_of_customers = 100
        number_of_facilities = 100
        lp_dir = f'data/instances/facdem_maxopen{facdemlow}_{facdemhigh}_{facdemcaplow}_{facdemcaphigh}_{facmaxopen}/test_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # actually generate the instances
        for filename, ncs, nfs, r in zip(filenames, ncustomerss, nfacilitiess, ratios):
            print(f"  generating file {filename} ...")
            generate_capacited_facility_location_vary_demand_maxfacopen(rng, filename, n_customers=ncs, n_facilities=nfs, ratio=r, dem_low=facdemlow, dem_high=facdemhigh, cap_low=facdemcaplow, cap_high=facdemcaphigh, maxfacopen=facmaxopen)

        print("done.")
                 
            
    
    
    
    if args.problem == 'setcover_densize':
        nrows = 700
        ncols = 800
        # dens = 0.05
        max_coef = 100
        dens = args.density
        print('dens', dens)

        filenames = []
        nrowss = []
        ncolss = []
        denss = []

        # train instances
        n = 10000
        # n = 100
        lp_dir = f'data/instances/setcover_densize_{dens}/train_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # validation instances
        n = 200
        n = 100
        lp_dir = f'data/instances/setcover_densize_{dens}/valid_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # small transfer instances
        n = 100
        nrows = 400
        lp_dir = f'data/instances/setcover_densize_{dens}/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # medium transfer instances
        n = 100
        nrows = 800
        lp_dir = f'data/instances/setcover_densize_{dens}/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # big transfer instances
        n = 100
        nrows = 1000
        lp_dir = f'data/instances/setcover_densize_{dens}/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # test instances
        # n = 2000
        n = 100
        nrows = 700
        ncols = 800
        lp_dir = f'data/instances/setcover_densize_{dens}/test_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # actually generate the instances
        for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
            print(f'  generating file {filename} ...')
            generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

        print('done.')
        

        
 