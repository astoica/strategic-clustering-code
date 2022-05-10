''' This code implements the methodology described in Section 6, computing the Nash equilibria of different utility functions 
and comparing the minimum and average balance of Spectral Clustering, Fair Spectral Clustering, and equilibrium solutions.'''

import networkx as nx
import csv 
import numpy as np
import random
import copy
import math 
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib.pyplot as plt
import scipy

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
def utility_node_divided(G, list_nodes_G, cluster_assignment, no_clusters, node):
    lengths_total = nx.single_source_shortest_path_length(G, node)
    lengths = {}
    for i in range(no_clusters):
        lengths[i] = 0 
        cl = np.where(cluster_assignment == i)[0]
        for j in cl: 
            lengths[i] += lengths_total[list_nodes_G[j]]
    cluster_assignment_counterfactual = {}

    utility = {}
    for i in range(no_clusters):
        cluster_assignment_counterfactual[i] = copy.deepcopy(cluster_assignment)
        cluster_assignment_counterfactual[i][list_nodes_G.index(node)] = i
        deg_u = incluster_degree(G, list_nodes_G, cluster_assignment_counterfactual[i], node)
        utility[i] = deg_u / lengths[i] 
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
        utility[i] = deg_u / (len(np.where(cluster_assignment_counterfactual[i] == i)[0]) - 1)
    return utility

# computing average balance of a graph and a clustering
def compute_balance(G,list_of_nodes_G,cluster_assignment,no_of_clusters, sensitive_info):
   balance_avg = 0
   for cl in range(no_of_clusters):
       
       clind = np.where(cluster_assignment == cl)[0]
       sens0cl = 0
       sens1cl = 0
       for j in clind:
           if sensitive_info[j] == 0:
               sens0cl += 1
           else:
               sens1cl +=1
       if sens0cl == 0 or sens1cl == 0:
           balance_cl = 0
       else:
           balance_cl = min(sens0cl/sens1cl,sens1cl/sens0cl)
       balance_avg += balance_cl
   return balance_avg/no_of_clusters

# computing min balance of a graph and a clustering, as per Cherichietti et al.
def compute_balancemin(G,list_of_nodes_G,cluster_assignment,no_of_clusters, sensitive_info):
    balance_min = []
    for cl in range(no_of_clusters):
        clind = np.where(cluster_assignment == cl)[0]
        sens0cl = 0
        sens1cl = 0
        for j in clind:
            if sensitive_info[j] == 0:
                sens0cl += 1
            else:
                sens1cl +=1
        if sens0cl == 0 or sens1cl == 0:
            balance_cl = 0
        else:
            balance_cl = min(sens0cl/sens1cl,sens1cl/sens0cl)
        balance_min.append(balance_cl)
    return min(balance_min)

'''This code implements (translates from matlab) fair clustering, where fairness is defined as statistical parity 
for the sensitive attribute [from https://github.com/matthklein/fair_spectral_clustering/blob/master/Fair_SC_normalized.m]'''

#function clusterLabels = Fair_SC_normalized(adj,k,sensitive)
#implementation of fair normalized SC as stated in Alg. 3 
#
#INPUT:
#adj ... (weighted) adjacency matrix of size n x n
#k ... number of clusters
#sensitive ... vector of length n encoding the sensitive attribute 
#
#OUTPUT:
#clusterLabels ... vector of length n comprising the cluster label for each
#                  data point

def Fair_SC_normalized(G, adj,no_clusters,sensitive):
    n = np.shape(adj)[1]
    sens_unique = np.unique(sensitive)    
    h = len(sens_unique)
    sensitiveNEW=sensitive.copy()
    
    temp = 0
    
    for ell in sens_unique:
        sensitiveNEW[np.where(sensitive==ell)[0]] = temp
        temp += 1

    F=np.zeros([n,h-1])
    for ell in range(h-1):
        temp = np.where(sensitiveNEW == ell)[0]
        F[temp,ell]=1
        groupSize = len(temp)
        F[:,ell] = F[:,ell]-groupSize/n

    L = nx.normalized_laplacian_matrix(G)
    L.todense()
    D = np.diag(np.sum(np.array(adj.todense()), axis=1))

    _,Z = null(F.transpose())
    zz = ((Z.transpose()).dot(D)).dot(Z)
    Q = scipy.linalg.sqrtm(zz)
    Q = Q.real
    Qinv = np.linalg.inv(Q)
    
    Msymm = ((((Qinv.transpose()).dot(Z.transpose())).dot(L.todense())).dot(Z)).dot(Qinv)
    Msymm = (Msymm+Msymm.transpose())/2
    e,v = np.linalg.eig(Msymm)
    
    i = [list(e).index(j) for j in sorted(list(e))[1:no_clusters]]
    Y = np.array(v[:, i])
    Y = Y.real

    H = (Z.dot(Qinv)).dot(Y)
    
    km_fair = KMeans(init='k-means++', n_clusters=no_clusters, max_iter=200, n_init=200, verbose=0, random_state=3425)
    km_fair.fit(H)
    clusterLabels = km_fair.labels_
    return clusterLabels

def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

# the following function finds an equilibrium partition for the closeness utility and computes average and min balance 
def process_closeness(Gc,list_nodes,k,no_iterations,sensitive):
    balance_eq_closeness = {}
    balance_eq_closenessmin = {}

    for iter in range(no_iterations):
        print("Iteration: ", iter)


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

        balance_eq_closeness[iter] = compute_balance(Gc,list_nodes,y_utiltest,k,sensitive)
        balance_eq_closenessmin[iter] = compute_balancemin(Gc,list_nodes,y_utiltest,k,sensitive)

    balance_eq_closeness_avg = np.mean([x for x in balance_eq_closeness.values()])
    balance_eq_closeness_std = np.std([x for x in balance_eq_closeness.values()])
    balance_eq_closenessmin_avg = np.mean([x for x in balance_eq_closenessmin.values()])
    balance_eq_closenessmin_std = np.std([x for x in balance_eq_closenessmin.values()])

    return balance_eq_closeness_avg, balance_eq_closeness_std, balance_eq_closenessmin_avg, balance_eq_closenessmin_std

# the following function finds an equilibrium partition for the modified fractional utility and computes average and min balance 
def process_mfu(Gc, list_nodes, k, no_iterations, sensitive):
    balance_eq_mfu = {}
    balance_eq_mfumin = {}

    for iter in range(no_iterations):
        print("Iteration: ", iter)

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

        balance_eq_mfu[iter] = compute_balance(Gc,list_nodes,y_utiltest,k,sensitive)
        balance_eq_mfumin[iter] = compute_balancemin(Gc,list_nodes,y_utiltest,k,sensitive)
    balance_eq_mfu_avg = np.mean([x for x in balance_eq_mfu.values()])
    balance_eq_mfu_std = np.std([x for x in balance_eq_mfu.values()])
    balance_eq_mfumin_avg = np.mean([x for x in balance_eq_mfumin.values()])
    balance_eq_mfumin_std = np.std([x for x in balance_eq_mfumin.values()])
    return balance_eq_mfu_avg, balance_eq_mfu_std, balance_eq_mfumin_avg, balance_eq_mfumin_std    


### the following section reads in one of the datasets: APS, Facebook, Highschool; uncomment for the data desired to use
'''#APS dataset: 
filename = 'APS-clusteringgames-balance_min.csv'
dataset = 'APS'
# read in the data as a graph
G_og = nx.read_gexf('APS/sampled_APS_pacs052030.gexf')

# work with the largest connected compoenent
gg = sorted(nx.connected_components(G_og),key=len,reverse=True)[0]
Gc = G_og.subgraph(gg)

list_nodes=list(Gc.nodes())
print("read the APS graph")


# finding the spectrum of the graph
A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))
e, v = np.linalg.eig(L.todense())

mm = 0 
ff = 0 

for u in Gc.nodes():
    if (Gc.nodes[u]['pacs'] == '05.30.-d'):
        mm += 1
    else:
        ff += 1
sensitive = []
for u in list_nodes:
    if (Gc.nodes[u]['pacs'] == '05.30.-d'):
        sensitive.append(1)
    else:
        sensitive.append(0)
sensitive = np.array(sensitive)
'''

'''#Facebook dataset: 
filename = 'Facebook-clusteringgames-utilities-k' + str(k) + '.csv'
dataset = 'Facebook'
# read in the data as a graph
Gc = nx.read_edgelist('Facebook/facebook_combined.txt')

list_nodes=list(Gc.nodes())
print("read the Facebook graph")

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
    filenamet= 'Facebook/' + u +'.feat'
    f = open(filenamet)
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
filename = 'Highschool-clusteringgames-utilities.csv'
dataset = 'Highschool'
# read in the data as a graph
G_og = nx.read_edgelist('Highschool/Friendship-network_data_2013.csv')

# get the largest connected component of the graph
gg = sorted(nx.connected_components(G_og),key=len,reverse=True)[0]
Gbig = G_og.subgraph(gg)
Gc = Gbig.copy()
print("read the Highschool graph")

# finding the spectrum of the graph

# find the sensitive features (unanonymized gender) and place it in a dictionary
gender = {}

filename_to_read = 'Highschool/metadata_2013.txt'
f = open(filename_to_read)
reader=csv.reader(f)

for row in reader:
    myrow = row[0].split('\t')
    gender[myrow[0]] = myrow[2]

list_init = list(Gc.nodes())
for u in list_init:
    if gender[u] == 'Unknown':
        Gc.remove_node(u)

mm = 0 
ff = 0
for u in Gc.nodes():
    if gender[u] == 'M':
        mm += 1
    else:
        ff += 1


# find the spectrum of the graph
list_nodes = list(Gc.nodes())
A = nx.adjacency_matrix(Gc)
L = nx.normalized_laplacian_matrix(Gc)
L.todense()
D = np.diag(np.sum(np.array(A.todense()), axis=1))
e, v = np.linalg.eig(L.todense())
sensitive = []
for u in list_nodes:
    if gender[u] == 'F':
        sensitive.append(1)
    else:
        sensitive.append(0)
sensitive = np.array(sensitive)
'''

data_balance = min(len(np.where(sensitive == 0)[0]) / len(np.where(sensitive == 1)[0]),len(np.where(sensitive == 1)[0]) / len(np.where(sensitive == 0)[0]))

myfile = open(filename,'w')
writer= csv.writer(myfile, lineterminator="\n")
# write header row for the data file 
writer.writerow(['no_clusters', 'conductance_SC', 'conductance_SC_fair', 'balance_SC_avg', 'balance_SC_avg_fair', 'balance_SC_min', 'balance_SC_min_fair', 'balance_eq_closeness_avg', 'balance_eq_closeness_std', 'balance_eq_closenessmin_avg', 'balance_eq_closenessmin_std', 'balance_eq_mfu_avg', 'balance_eq_mfu_std', 'balance_eq_mfumin_avg', 'balance_eq_mfumin_std'])

no_iterations = 50
no_clusters = [2,3,4,5,6,7,8]

balance_SC = {}

balance_eq_closeness_avg = []
balance_eq_closeness_std = []
balance_eq_mfu_avg = []
balance_eq_mfu_std = []
balancemin_eq_closeness_avg = []
balancemin_eq_closeness_std = []
balancemin_eq_mfu_avg = []
balancemin_eq_mfu_std = []

balance_SC_min = {}
balance_SC_min['sc'] = {}
balance_SC_min['scfair'] = {}

balance_SC_avg = {}
balance_SC_avg['sc'] = {}
balance_SC_avg['scfair'] = {}

conductance_scfair = {}
conductance_sc = {}

for k in no_clusters:
    print("no of clusters: ", k)
    # compute the balance and conductance of spectral clustering and fair spectral clustering; note that conductance for SC and equilibrium solutions has already been computed in section 4
    print("k: ", k)
    i = [list(e).index(j) for j in sorted(list(e))[1:k+1]]
    U = np.array(v[:, i])
    sc = SpectralClustering(n_clusters=k,affinity="precomputed",n_init=200)
    ysc = sc.fit_predict(A)
    y_scfair = Fair_SC_normalized(Gc,A,k,sensitive)

    conductance_scfair[k] = compute_conductance(Gc,list_nodes,ysc,k)
    conductance_sc[k] = compute_conductance(Gc,list_nodes,ysc_fair,k)

    balance_SC_min['sc'][k] = compute_balancemin(Gc,list_nodes,ysc,k, sensitive)
    balance_SC_min['scfair'][k] = compute_balancemin(Gc,list_nodes,y_scfair,k, sensitive)

    balance_SC_avg['sc'][k] = compute_balance(Gc,list_nodes,ysc,k, sensitive)
    balance_SC_avg['scfair'][k] = compute_balance(Gc,list_nodes,y_scfair,k, sensitive)

    # compute the min and avg balance of the closeness utility and the modified fractional utility 
    balance_eq_closeness_avg, balance_eq_closeness_std, balance_eq_closenessmin_avg, balance_eq_closenessmin_std = process_closeness(Gc,list_nodes,k,no_iterations,sensitive)
    balance_eq_mfu_avg, balance_eq_mfu_std, balance_eq_mfumin_avg, balance_eq_mfumin_std = process_mfu(Gc, list_nodes, k, no_iterations, sensitive)

    write.writerow([k, conductance_sc[k], conductance_scfair[k], balance_SC_avg['sc'][k], balance_SC_avg['scfair'][k], balance_SC_min['sc'][k], balance_SC_min['scfair'][k], balance_eq_closeness_avg, balance_eq_closeness_std, balance_eq_closenessmin_avg, balance_eq_closenessmin_std, balance_eq_mfu_avg, balance_eq_mfu_std, balance_eq_mfumin_avg, balance_eq_mfumin_std])

myfile.close()



# an example of plotting the min balance
# balance_SC_list = sorted(balance_SC_min['sc'].items())
# balance_SC_fair_list = sorted(balance_SC_min['scfair'].items())
# xb,yb = zip(*balance_SC_list)
# xbf,ybf = zip(*balance_SC_fair_list)
# plt.plot(xb,yb,color='k',label="Spectral Clustering")
# plt.plot(xbf,ybf,'k--',label="Fair Spectral Clustering")

# plt.errorbar(cluster_iter,balance_eq_closeness_avg,yerr=balance_eq_closeness_std,color='purple',label="Closeness")
# plt.errorbar(cluster_iter,balance_eq_mfu_avg,yerr=balance_eq_mfu_std,color='b',label="mfu")

# plt.hlines(data_balance,cluster_iter[0],cluster_iter[-1],linestyle='-.', color='green',label='Data Balance')
# plt.grid(linestyle = '--')

# plt.xlabel("Number of clusters")
# plt.ylabel("Min balance")
# plt.legend()
# file_fig = dataset + '_balance_SC_utilities_min.pdf'
# plt.savefig(file_fig)