import data_gen
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import heapq
import datetime
import os


class Graph:
    def __init__(self):
        self.nodes_xy = {}
        self.dict_edges = {}
        self.num_node = 100
        for i in range(self.num_node):
            neighbors = {}
            if i == 0:
                neighbors[i + 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif i == 9:
                neighbors[i - 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif i == 90:
                neighbors[i + 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i - 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif i == 99:
                neighbors[i - 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i - 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif 0 < i < 9:
                neighbors[i - 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif 90 < i < 99:
                neighbors[i - 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i - 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif i % 10 == 0 and (i > 0 or i < 90):
                neighbors[i - 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            elif i % 10 == 9 and (i > 9 or i < 99):
                neighbors[i - 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i + 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
                neighbors[i - 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 1}
            else:
                neighbors[i - 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 2}
                neighbors[i + 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 2}
                neighbors[i + 1] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 2}
                neighbors[i - 10] = {'weight': 1, 'dist': 2, 'velo': 1, 'road type': 2}

            self.dict_edges[i] = neighbors
            self.nodes_xy[i] = (int(i/10), i%10)

        # print(self.nodes)

    def neighbors(self, id):
        return self.dict_edges[id].keys()

    def weight(self, fromnode, tonode):
        return self.dict_edges[fromnode][tonode]['weight']

    def distance(self, fromnode, tonode):
        return self.dict_edges[fromnode][tonode]['dist']

    def velocity(self, fromnode, tonode):
        return self.dict_edges[fromnode][tonode]['velo']

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.dict_edges[fromenode][tonode]["dist"]
        return distance

class Node:
    def __init__(self, info): # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat', 'long'
        # print("node")
        self.node_id = info[0]
        self.node_type = info[1]
        self.node_name = info[2]
        self.lati = info[3]
        self.long = info[4]
        print(self.node_id)
        # return {'NODE_ID': self.node_id, 'NODE_TYPE':self.node_type , 'NODE_NAME': self.node_name, 'lat': self.lati, 'long': self.long}

class Link:
    def __init__(self, info): # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH'
        # print("link")
        self.link_id = info[0]
        self.f_node = info[1]
        self.t_node = info[2]
        self.max_spd = info[3]
        self.length = info[4]
        print(self.link_id)



class Graph_jeju:
    def __init__(self, datapath):
        self.link_data, self.node_data, self.traffic_info, self.cs_info = data_gen.network_info(datapath)
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1
                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))

                    errorlink += 1
            else:
                maxspd = self.link_data[l]['MAX_SPD']
                self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]



    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')


if __name__ == "__main__":
    # now = datetime.datetime.now()
    # basepath = os.getcwd()
    # resultdir = '{0:02}-{1:02} {2:02}-{3:02} result'.format(now.month, now.day, now.hour, now.minute)
    # print(os.path.join(basepath, resultdir))
    # dirpath = os.path.join(basepath, resultdir)
    # createFolder(dirpath)
    # qq = PriorityQueue()
    # qq.put(1, 10)
    # qq.put(2, 20)
    # qq.put(3, 30)
    #
    # n = qq.get()
    # print(n)


    # g = Graph_jeju('data/20191001_5Min_modified.csv')
    # print(g.get_link_id(4050001500, 4050001800))

    # print(g.neighbors_list[4070043200])
    # nlist = np.random.random_integers(0, 10, 50)
    # print(nlist)
    # plt.plot(range(50), nlist, 'x', label='t1 EVCS')
    # nlist = np.random.random_integers(0, 10, 50)
    # plt.plot(range(50), nlist, '+', label='t2 EVCS')
    # fig = plt.gcf()
    # fig.savefig('t1.png', facecolor='#eeeeee')
    # plt.clf()
    #
    #
    plt.figure(figsize=(12, 6), dpi=300)
    plt.title('Initial SOC')
    plt.xlabel('EV ID')
    plt.ylabel('SOC')
    nlist = np.random.random_integers(0, 10, 50)
    print(nlist)

    plt.plot(range(50), nlist, '-', label='t1 EVCS')
    nlist = np.random.random_integers(0, 10, 50)
    plt.plot(range(50), nlist, '-', label='t2 EVCS')
    #
    # plt.show()
    # fig = plt.gcf()
    plt.savefig('test.png', facecolor='#eeeeee')
    plt.clf()
