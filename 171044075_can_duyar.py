import pandas as pd
import numpy as np
import networkx as nx
import metis
import matplotlib.pyplot as plt

# CAN DUYAR - 171044075


class IterableInt:
    def __init__(self, i):
        self.i = i
    
    def __iter__(self):
        return map(int, str(self.i))


# this method finds combination with n value of the list which is given as parameter
def combinationNth(listParam, nth):
    
    keep =[]
    
    if nth == 0:
        return [[]]
        
    for g in range(0, len(listParam)):
        index = listParam[g]
        list_collect = listParam[g + 1:]
         
        for t in combinationNth(list_collect, nth-1):
            keep.append([index]+t)
             
    return keep


#this methods creates the knn graph with using euclidean distance formula
def graphEuclidean(all_vertices,k):
    newGraph = nx.Graph()
    distances = []
    keep = []
    for i in range(0, len(all_vertices)):
        newGraph.add_node(i)
         
    for t,g in enumerate(all_vertices):
        #this part calculates the euclidean distances between all_vertices 
        distances = list(map(lambda itr: calculateEuclidean(g, itr), all_vertices))
        # this part orders the distances as increasingly                  
        neighbors = indexReturn(distances,k)  
        #this part adds edges to the graph by adding optimal weight values
        for it in neighbors:
            if distances[it] != 0:  # this part checks the error of divide by zero
                newGraph.add_edge(t, it, weight=1.0 / distances[it])
           
    return newGraph           
          
# this method calculates the euclidean distance between points
def calculateEuclidean(a, b):
    
    sum_sq = np.square(float(a[0]) - float(b[0])) + np.square(float(a[1]) - float(b[1]))
    calculateEuclidean = np.sqrt(sum_sq)
    return calculateEuclidean
   

# this method orders the list of small elements' indexes as increasingly
def indexReturn(distances,k):
    indexes = []
    sorted_distances = []
    sorted_distances = sorted(distances)
    for i in range(0,len(sorted_distances)):
        indexes.append(np.where(distances == sorted_distances[i])[0][0])
    return indexes[1:k+1]
       

# this method creates knn graph with using graphEuclidean method
def createKnnGraph(df, k):
    all_vertices = []
    for p in df.itertuples():
            all_vertices.append(p[1:])
            
    return graphEuclidean(all_vertices,k)

 
def beforeChameleonDraw(df):
    plt.scatter(df[0],df[1])
    plt.title('Before Chameleon Clustering')
    plt.show()


def afterChameleonDraw(df):
    df.plot(kind='scatter', c=df['cluster'], cmap='jet' , x=0, y=1,title = 'After Chameleon Clustering')
    plt.show()    



def closenessChameleon(closenessGraph, c1, c2):
    edgeList = []
    
    for g in c1:
        for t in c2:
            if g in closenessGraph:
                if t in closenessGraph[g]:
                    edgeList.append((g, t))
    
    
    listWeights = []
    
    for pair in edgeList:
        listWeights.append(closenessGraph[pair[0]][pair[1]]['weight'])
        
    averageWeight = np.mean(listWeights)
        
    cluster = closenessGraph.subgraph(c1)
    edgeList = cluster.edges()
    
    
    partitionList = []
    cluster = cluster.copy()
    cluster = graphPartition(cluster)
    
  
    partitionList =  [t for t in cluster.nodes if cluster.nodes[t]['cluster'] in [0]],[t for t in cluster.nodes if cluster.nodes[t]['cluster'] in [1]] 
    
    edges = []
    
    for g in partitionList[0]:
        for t in partitionList[1]:
            if g in cluster:
                if t in cluster[g]:
                    edges.append((g, t))
    
    
    weights = []
    weights2 = []
    for it,it2 in zip(edgeList,edges):
        weights.append(cluster[it[0]][it[1]]['weight'])
        weights2.append(cluster[it2[0]][it2[1]]['weight'])         
    
    
    cluster2 = closenessGraph.subgraph(c2)
    edgeList = cluster2.edges()
    
    cluster2 = cluster2.copy()
    cluster2 = graphPartition(cluster2)
              
    partitionList2 =  [t for t in cluster2.nodes if cluster2.nodes[t]['cluster'] in [0]],[t for t in cluster2.nodes if cluster2.nodes[t]['cluster'] in [1]] 
    edges2 = []
    
    for g in partitionList2[0]:
        for t in partitionList2[1]:
            if g in cluster2:
                if t in cluster2[g]:
                    edges2.append((g, t))
                  
    weights3 = []
    weights4 = []
    for it3,it4 in zip(edgeList,edges2):
         weights3.append(cluster2[it3[0]][it3[1]]['weight'])
         weights4.append(cluster2[it4[0]][it4[1]]['weight'])         
        

    return np.mean(listWeights) / ((np.sum(weights) / (np.sum(weights) + np.sum(weights3)) *
                                    np.mean(weights2)) + (np.sum(weights3) / (np.sum(weights) + np.sum(weights3)) * np.mean(weights4)))



   
def interconnectivity(interGraph, cls1, cls2):
    
    edgeList = []
    for g in cls1:
        for t in cls2:
            if g in interGraph:
                if t in interGraph[g]:
                    edgeList.append((g, t))
    
    
    
    total = []
    
    for iter1 in edgeList:
        total.append(interGraph[iter1[0]][iter1[1]]['weight'])
    
    totalWeight = np.sum(total)    
    
    weightList = []
    edgesList = []
    cluster = interGraph.subgraph(cls1)

    cluster = cluster.copy()
    cluster = graphPartition(cluster)
    partitionList2 =  [t for t in cluster.nodes if cluster.nodes[t]['cluster'] in [0]],[t for t in cluster.nodes if cluster.nodes[t]['cluster'] in [1]] 
    
    
    edgesList = []
    for g in partitionList2[0]:
        for t in partitionList2[1]:
            if g in cluster:
                if t in cluster[g]:
                    edgesList.append((g, t))
        
    for it in edgesList:
        weightList.append(cluster[it[0]][it[1]]['weight'])        
    
    weightList2 = []
    edgesList2 = []
    cluster2 = interGraph.subgraph(cls2)
    
    partitionList = []
    cluster2 = cluster2.copy()
    cluster2 = graphPartition(cluster2)
    for t in cluster2.nodes:
        if cluster2.nodes[t]['cluster'] in [0] or cluster2.nodes[t]['cluster'] in [1]:
            partitionList.append(IterableInt(t))
   
        
    edgesList2 = []
    for g in partitionList[0]:
        for t in partitionList[1]:
            if g in cluster2:
                if t in cluster2[g]:
                    edgesList2.append((g, t))
            
    for it2 in edgesList2:
        weightList2.append(cluster2[it2[0]][it2[1]]['weight'])
    if (np.sum(weightList) + np.sum(weightList2)) != 0:
        return (totalWeight / ((np.sum(weightList) + np.sum(weightList2)) / 2.0))*closenessChameleon(interGraph, cls1, cls2)*closenessChameleon(interGraph, cls1, cls2)
    
    
def mergingPhase(knnGraph, data, numb):
    
    uniqueClusters = np.unique(data['cluster'])
    optimumMerging = 0
    limitNum = len(uniqueClusters)
    cls1 = -1
    cls2 = -2
    if limitNum > numb:
        for combination in combinationNth(uniqueClusters, 2):
            u, v = combination
            if u != v:
                list1 = [t for t in knnGraph.nodes if knnGraph.nodes[t]['cluster'] in [u]]   
                list2 = [t for t in knnGraph.nodes if knnGraph.nodes[t]['cluster'] in [v]]
    
                edgeList = []
                                                    
                for g in list1:
                    for t in list2:
                        if g in knnGraph:
                            if t in knnGraph[g]:
                                edgeList.append((g, t))
                                
                if not edgeList:
                    continue
                optimum = interconnectivity(knnGraph, list1, list2)
                if optimum is not None and optimum > optimumMerging:
                    optimumMerging = optimum
                    cls1 = u
                    cls2 = v
            
                    
        if optimumMerging > 0:
            data.loc[data['cluster'] == cls2, 'cluster'] = cls1
            for _, r in enumerate(knnGraph.nodes()):
                if knnGraph.nodes[r]['cluster'] == cls2:
                    knnGraph.nodes[r]['cluster'] = cls1
            return True        
        
        else:
            return False
    else:
        return False




def chameleonClustering(df, k, knn):
    i = 0
    graph = createKnnGraph(df, knn)
    graph = helperGraphPartition(graph, df)  
    while i<30:
        mergingPhase(graph, df, k)
        i = i+1



def graphPartition(partition, data=None):
    _, cut = metis.part_graph(partition, 2, objtype='cut', ufactor=250)
    
    if data is not None:
        data['cluster'] = nx.get_node_attributes(partition, 'cluster').values()
        
    for g, t in enumerate(partition.nodes()):
        partition.nodes[t]['cluster'] = cut[g]    
    return partition



def helperGraphPartition(part, data=None):
    limit_control = 0
    upper_limit = 30
    for _, g in enumerate(part.nodes()):
        part.nodes[g]['cluster'] = 0
    collect = {}
    collect[0] = len(part.nodes())

    while limit_control < upper_limit:
        keep = -1
        numb = 0
        for key, val in collect.items():
            if val > numb:
                numb = val
                keep = key
        list1 = [t for t in part.nodes if part.nodes[t]['cluster'] == keep]
        list2 = part.subgraph(list1)
        _, coll = metis.part_graph(list2, 2, objtype='cut', ufactor=250)
        temp = 0
        for g, t in enumerate(list2.nodes()):
            if coll[g] == 1:
                part.nodes[t]['cluster'] = limit_control + 1
                temp = temp + 1
        collect[keep] = collect[keep] - temp
        collect[limit_control + 1] = temp
        limit_control = limit_control + 1

    _, coll = metis.part_graph(part, upper_limit)
    if data is not None:
        data['cluster'] = nx.get_node_attributes(part, 'cluster').values()
    return part



if __name__ == "__main__":
    
    # reading csv
    dataFrame = pd.read_csv('data.csv', sep=',',
                     header=None)
    
    beforeChameleonDraw(dataFrame)
    
    knn_value = 15
    number_of_clusters = 4
    
    chameleonClustering(dataFrame,number_of_clusters,knn_value)

    afterChameleonDraw(dataFrame)




