
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

#
# g = Graph()
# print(g.dict_edges)