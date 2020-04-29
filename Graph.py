
class Graph:
    def __init__(self):
        self.nodes = {}
        self.dict_edges = {}

        for i in range(100):
            neighbors = {}
            if i == 0:
                neighbors[i + 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 1] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif i == 9:
                neighbors[i - 1] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 10] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif i == 90:
                neighbors[i + 1] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i - 10] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif i == 99:
                neighbors[i - 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i - 1] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif 0 < i < 9:
                neighbors[i - 1] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 1] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif 90 < i < 99:
                neighbors[i - 1] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i - 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 1] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif i % 10 == 0 and (i > 0 or i < 90):
                neighbors[i - 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 1] = {'cost': 1, 'dist': 1, 'velo': 1}
            elif i % 10 == 9 and (i > 9 or i < 99):
                neighbors[i - 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i - 1] = {'cost': 1, 'dist': 1, 'velo': 1}
            else:
                neighbors[i - 1] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 10] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i + 1] = {'cost': 1, 'dist': 1, 'velo': 1}
                neighbors[i - 10] = {'cost': 1, 'dist': 1, 'velo': 1}

            self.dict_edges[i] = neighbors
            self.nodes[i] = (int(i/10), i%10)

        # print(self.nodes)

    def neighbors(self, id):
        return self.dict_edges[id].keys()

    def cost(self, fromnode, tonode):
        return self.dict_edges[fromnode][tonode]['cost']

    def distance(self, fromnode, tonode):

        return 0

#
# g = Graph()
# print(g.dict_edges)