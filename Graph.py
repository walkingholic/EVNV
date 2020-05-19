import data_gen
import pprint as pp

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
        self.link_data, self.node_data, self.traffic_info = data_gen.network_info(datapath)
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0
        remove_link_list = []
        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1
                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))

                    errorlink += 1
            else:
                remove_link_list.append(l)



        print('Num links from traffic data', len(self.traffic_info.keys()))

        for l in remove_link_list:
            self.link_data.pop(l)

        print('Modified Num link', len(self.link_data.keys()))

        node = set()
        for l in self.link_data.keys():
            node.add(self.link_data[l]['F_NODE'])
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            node.add(self.link_data[l]['T_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        if 4070043200 in node:
            print('4070043200 in ')

        test = {}
        for n in node:
            if n in self.node_data.keys():
                test[n] = self.node_data[n]
            else:
                print(n, 'something error')
        print(test)
        self.node_data = test
        print(self.node_data)

        print(len(node), len(test), len(self.node_data))

        print('Modified Num node', len(self.node_data.keys()))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

        # print('neighbors_list', len(self.neighbors_list.keys()))
        # print('link 4070105400', self.link_data[4070105400])



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
        # print(n, link_id_list, self.link_data[link_id_list[0]]['WEIGHT'])
        # print( fromnode, tonode)
        # for lid in link_id_list:
        #     pp.pprint(self.link_data[lid])
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

    def nodes_xy(self, nidx):
        # print(node_id)
        # print(self.node_data[nidx]['long'])
        # print(self.node_data[nidx]['lat'])
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

if __name__ == "__main__":

    g = Graph_jeju('data/20191001_5Min_modified.csv')
    # print(g.get_link_id(4050001500, 4050001800))

    print(g.neighbors_list[4070043200])
