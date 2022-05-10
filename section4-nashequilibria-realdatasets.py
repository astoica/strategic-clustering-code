''' This code implements the methodology described in Section 4, computing the Nash equilibria of different utility functions 
and comparing (1) conductance of Spectral Clustering with the equilibrium solutions and (2) average utility of Spectral Clustering as 
compared to the equilibrium solutions.'''

import networkx as nx
import csv 
import numpy as np
import random
import copy
import math 
from sklearn.cluster import SpectralClustering, KMeans

# the function compute_conductance() computes average conductance for a clustering assignment
def compute_conductance(G, list_of_nodes_G, cluster_assignment, no_of_clusters):
    clconductance = {}
    for i in range(no_of_clusters):
        icl = [list_of_nodes_G[x] for x in list(np.where(cluster_assignment == i)[0])]
        #print(icl)
        if len(icl) == 0 or len(icl) == len(list_of_nodes_G):
            clconductance[i] = 0
        else:
            clconductance[i] = nx.conductance(G,icl)
    if(len([v for v in clconductance.values() if v > 0]) == 0):
        return 0
    else:
        return 1 - sum(clconductance.values())/len([v for v in clconductance.values() if v > 0])

# the function cut_size() computes the cut size between a subgraph S and a graph G
def cut_size(G, S, T=None, weight=None):
    edges = nx.edge_boundary(G, S, T, data=weight, default=1)
    if G.is_directed():
        edges = chain(edges, nx.edge_boundary(G, T, S, data=weight, default=1))
    return sum(weight for u, v, weight in edges)

# the function incluster_degree() computes the incluster degree of a node, given a graph and a specified clustering partition (number of edges that are fully within the same cluster as the source node) 
def incluster_degree(G, list_nodes_G, cluster_assignment, node):
    degu = 0
    for nbr in G.neighbors(node):
        if cluster_assignment[list_nodes_G.index(node)] == cluster_assignment[list_nodes_G.index(nbr)]:
            degu += 1
    return degu

# the function utility_node_divided() computes the closeness utility of a node, given a graph and a clustering partition
def utility_node_divided(G, list_nodes_G, cluster_assignment, no_clusters, threshold, node):
    lengths_total = nx.single_source_shortest_path_length(G, node)
    lengths = {}
    for i in range(no_clusters):
        lengths[i] = 0 
        cl = np.where(cluster_assignment == i)[0]
        for j in cl: 
            lengths[i] += lengths_total[list_nodes_G[j]]
    #deg_u = incluster_degree(G, list_nodes_G, cluster_assignment, node)
    cluster_assignment_counterfactual = {}

    utility = {}
    for i in range(no_clusters):
        cluster_assignment_counterfactual[i] = copy.deepcopy(cluster_assignment)
        cluster_assignment_counterfactual[i][list_nodes_G.index(node)] = i
        deg_u = incluster_degree(G, list_nodes_G, cluster_assignment_counterfactual[i], node)
        if lengths[i] != 0:
            utility[i] = threshold * deg_u / lengths[i]
        else:
            utility[i] = 0
    return utility

# the utility function utility_node_mfu() computes the mfu of a node given a graph and a clustering partition
# the mfu is from Price of Pareto Optimality in hedonic games by elkind et al, namely, w_i(C) / |C| - 1 (or w_i(C)/|C|) where w_i(C) is the sum of utility of node i in cluster C
def utility_node_mfu(G, list_nodes_G, cluster_assignment, no_clusters, node):
    cluster_assignment_counterfactual = {}

    utility = {}
    for i in range(no_clusters):
        cluster_assignment_counterfactual[i] = copy.deepcopy(cluster_assignment)
        cluster_assignment_counterfactual[i][list_nodes_G.index(node)] = i
        deg_u = incluster_degree(G, list_nodes_G, cluster_assignment_counterfactual[i], node)
        if len(np.where(cluster_assignment_counterfactual[i] == i)[0]) != 0:
            utility[i] = threshold * deg_u / len(np.where(cluster_assignment_counterfactual[i] == i)[0])
        else:
            utility[i] = 0
    return utility

# the following function finds an equilibrium partition for the modified fractional utility 
def process_mfu(iter, k,prob_diffusion):

    conductance_random = {}
    util_all = {}


    for iter in range(no_iterations):
        print("Iteration: ", iter)
        util_all[iter] = 0
        conductance_random[iter] = 0

        list_nodes_copy = copy.deepcopy(list_nodes)
        random.shuffle(list_nodes_copy)
        assert(list_nodes_copy != list_nodes)
        part = [list_nodes_copy[i::k] for i in range(k)]

        # random initial partition
        y = np.zeros(len(list_nodes)) 
        for i in range(k):
            y[[list_nodes.index(xx) for xx in part[i]]] = i

        y_copy = copy.deepcopy(y)
        y_util = np.zeros(len(Gc.nodes()))
        mycounter = 0 
        myprob = 0.5

        for u in Gc.nodes():
            mycounter += 1
            x = utility_node_mfu(Gc, list_nodes, y_copy, k, u)
            y_util[list_nodes.index(u)] = max(x, key=x.get)

        y = copy.deepcopy(y_copy)

        counter =0 
        while list(y_util) != list(y):
            print("number of unmoved nodes: ", len(np.where(y==y_util)[0]))
            counter += 1
            if counter > 200:
                break

            y = y_util.copy()
            y_utiltest = np.zeros(len(Gc.nodes()))
            for u in Gc.nodes():
                dd = random.uniform(0,1)
                if dd > myprob:
                    x = utility_node_mfu(Gc, list_nodes, y_util, k, u)
                    y_utiltest[list_nodes.index(u)] = max(x, key=x.get)
                else:
                    y_utiltest[list_nodes.index(u)] = y_util[list_nodes.index(u)]
            y_util = y_utiltest.copy()
        for u in Gc.nodes():
            util_all[iter] += utility_node_mfu(Gc, list_nodes, y_util, k, u)[y_util[list_nodes.index(u)]]
        util_all[iter] /= len(list_nodes)
        conductance_random[iter] = compute_conductance(Gc,list_nodes,y_utiltest,k)
        
        conductance_random_avg = np.mean([x for x in conductance_random.values()])
        conductance_random_std = np.std([xx for xx in conductance_random.values()])
        util_all_avg = np.mean([x for x in util_all.values()])
        util_all_std = np.std([xx for xx in util_all.values()])
        return conductance_random_avg, conductance_random_std, util_all_avg, util_all_std



# the following function finds an equilibrium partition for the closeness utility 
def process_closeness(Gc,list_nodes,k, no_iterations):

    for iter in range(no_iterations):
        print("Iteration: ", iter)
        util_all[iter] = 0
        conductance_random[iter] = 0

        list_nodes_copy = copy.deepcopy(list_nodes)
        random.shuffle(list_nodes_copy)
        assert(list_nodes_copy != list_nodes)
        part = [list_nodes_copy[i::k] for i in range(k)]

        # random initial partition
        y = np.zeros(len(list_nodes)) 
        for i in range(k):
            y[[list_nodes.index(xx) for xx in part[i]]] = i

        y_copy = copy.deepcopy(y)
        y_util = np.zeros(len(Gc.nodes()))
        mycounter = 0
        myprob = 0.5

        for u in Gc.nodes():
            mycounter += 1
            x = utility_node_divided(Gc, list_nodes, y_copy, k, u)
            y_util[list_nodes.index(u)] = max(x, key=x.get)

        y = copy.deepcopy(y_copy)

        counter =0 
        while list(y_util) != list(y):
            print("number of unmoved nodes: ", len(np.where(y==y_util)[0]))

            counter += 1
            if counter > 100:
                break
            y = y_util.copy()
            y_utiltest = np.zeros(len(Gc.nodes()))

            for u in Gc.nodes():
                dd = random.uniform(0,1)
                if dd > myprob:
                    x = utility_node_divided(Gc, list_nodes, y_util, k, u)
                    y_utiltest[list_nodes.index(u)] = max(x, key=x.get)
                else:
                    y_utiltest[list_nodes.index(u)] = y_util[list_nodes.index(u)]
            y_util = y_utiltest.copy()
        for u in Gc.nodes():
            util_all[iter] += utility_node_divided(Gc, list_nodes, y_util, k, u)[y_util[list_nodes.index(u)]]
        util_all[iter] /= len(list_nodes)
        conductance_random[iter] = compute_conductance(Gc,list_nodes,y_utiltest,k)
        
    conductance_random_avg = np.mean([x for x in conductance_random.values()])
    conductance_random_std = np.std([xx for xx in conductance_random.values()])
    util_all_avg = np.mean([x for x in util_all.values()])
    util_all_std = np.std([xx for xx in util_all.values()])
    return conductance_random_avg, conductance_random_std, util_all_avg, util_all_std


### the following section reads in one of the datasets: APS, Facebook, Highschool; uncomment for the data desired to use
'''#APS dataset: 
filename = 'APS-clusteringgames-utilities-conductance.csv'

# read in the data as a graph
G_og = nx.read_gexf('Downloads/clustering_plotting/APS/sampled_APS_pacs052030.gexf')

# work with the largest connected compoenent
gg = sorted(nx.connected_components(G_og),key=len,reverse=True)[0]
Gc = G_og.subgraph(gg)

list_nodes=list(Gc.nodes())


# finding the spectrum of the graph
A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))
e, v = np.linalg.eig(L.todense())
'''

'''#Facebook dataset: 
filename = 'Facebook-clusteringgames-utilities-conductance.csv'

# read in the data as a graph
Gc = nx.read_edgelist('Facebook/facebook_combined.txt')

list_nodes=list(Gc.nodes())

# finding the spectrum of the graph
A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))
e, v = np.linalg.eig(L.todense())

gender = {}
egos = ['0', '107','348','414','686','698','1684','1912','3437','3980']
genderfeatfinder = {}

# find the sensitive feaures (anonymized gender), and place them in a dictionary
for u in egos: 
    genderfeatfinder[u] = {}
    filenamefeat = 'Facebook/' + u + '.featnames'
    ffeat = open(filenamefeat)
    readerfeat = csv.reader(ffeat)
    for rowfeat in readerfeat:
        myrowfeat = rowfeat[0].split()
        genderfeatfinder[u][myrowfeat[0]] = myrowfeat[1].split(';')[0]
    ffeat.close()
    gender_ind = [k for k,v in genderfeatfinder[u].items() if v == 'gender']
    filenameego= 'Facebook/' + u +'.egofeat'
    fego = open(filenameego)
    readerego =csv.reader(fego)
    for rowego in readerego:
        myrowego = rowego[0].split()
        gender[u] = myrowego[int(max(gender_ind))]
    fego.close()
    filename= 'Facebook/' + u +'.feat'
    f = open(filename)
    reader =csv.reader(f)
    for row in reader:
        myrow = row[0].split()
        user = myrow[0]
        gender[user] = myrow[int(max(gender_ind))+1]
    f.close()

# create a list, sensitive[], that encodes the anonymized gender in the data; it is not used in this section
sensitive = []
for u in list_nodes:
    if (gender[u] == '1'):
        sensitive.append(1)
    else:
        sensitive.append(0)
sensitive = np.array(sensitive)
sensitive
'''

'''#Highschool dataset:
filename = 'Highschool-clusteringgames-utilities-conductance.csv'

# read in the data as a graph
G_og = nx.read_edgelist('Downloads/Friendship-network_data_2013.csv')

# get the largest connected component of the graph
gg = sorted(nx.connected_components(G_og),key=len,reverse=True)[0]
Gbig = G_og.subgraph(gg)
Gc = Gbig.copy()

# k is the number of clusters for spectral clustering
#k = 4
# finding the spectrum of the graph

# find the sensitive features (unanonymized gender) and place it in a dictionary
gender = {}

filename_read = 'Downloads/metadata_2013.txt'
f = open(filename_read)
reader=csv.reader(f)

for row in reader:
    myrow = row[0].split('\t')
    gender[myrow[0]] = myrow[2]

list_init = list(Gc.nodes())
for u in list_init:
    if gender[u] == 'Unknown':
        Gc.remove_node(u)
        
# find the spectrum of the graph
list_nodes = list(Gc.nodes())
A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))
e, v = np.linalg.eig(L.todense())
'''

myfile = open(filename,'w')
writer= csv.writer(myfile, lineterminator="\n")
# header of the file containing the experimental data 
writer.writerow(['no_clusters', 'conductance_sc', 'util_sc_closeness', 'util_sc_mfu', 'conductance_closeness_avg', 'conductance_closeness_std', 'util_closeness_avg', 'util_closeness_std', 'conductance_mfu_avg', 'conductance_mfu_std', 'util_mfu_avg', 'util_mfu_std'])
# set the number of clusters
no_clusters = [2,4,6,8,10,12,14,16,18,20]

no_iterations = 50


for k in no_clusters:
    # compute the spectral clustering partition
    i = [list(e).index(j) for j in sorted(list(e))[1:k+1]]
    U = np.array(v[:, i])
    sc = SpectralClustering(n_clusters=k,affinity="precomputed",n_init=200)
    ysc = sc.fit_predict(A)
    # compute conductance of spectral clustering
    conductance_sc = compute_conductance(Gc,list_nodes,ysc,k)
    # compute utility of spectral clustering for the closeness utility and the modified fractional utility 
    util_sc_closeness = 0
    for u in Gc.nodes():
        util_sc_closeness += utility_node_divided(Gc, list_nodes, ysc, k, u)[ysc[list_nodes.index(u)]]
    util_sc_closeness /= len(list_nodes)

    util_sc_mfu = 0
    for u in Gc.nodes():
        util_sc_mfu += utility_node_mfu(Gc, list_nodes, ysc, k, u)[ysc[list_nodes.index(u)]]
    util_sc_mfu /= len(list_nodes)

    # compute the equilibrium partitions for the closeness utility, returning averaged results for conductance and utility 
    conductance_closeness_avg, conductance_closeness_std, util_closeness_avg, util_closeness_std = process_closeness(Gc,list_nodes,k, no_iterations)
    # compute the equilibrium partitions for the modified fractional utility, returning averaged results for conductance and utility 
    conductance_mfu_avg, conductance_mfu_std, util_mfu_avg, util_mfu_std = process_mfu(Gc,list_nodes,k, no_iterations)

# write data to file
writer.writerow([k, conductance_sc, util_sc_closeness, util_sc_mfu, conductance_closeness_avg, conductance_closeness_std, util_closeness_avg, util_closeness_std, conductance_mfu_avg, conductance_mfu_std, util_mfu_avg, util_mfu_std])
myfile.close()
