import numpy as np
import matplotlib.pyplot as plt
import heapq
from Graph import Graph

np.set_printoptions(precision = 4)

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class CS:
    def __init__(self, node_id, price):
        self.id = node_id
        self.price = price

class EV:
    def __init__(self, id, t_start, soc, source, destination):
        self.id = id
        self.t_start = t_start
        self.SOC = soc
        self.source = source
        self.destination = destination

def gen_envir():
    cs_location_list = [33, 37, 63, 67]
    CS_list = []
    for l in cs_location_list:
        cs = CS(l, 0.5)
        CS_list.append(cs)
    EV_list = []


    for e in range(100):

        t_start =  np.random.uniform(0, 96*15)
        soc = np.random.normal(0.5, 0.2)
        destination = np.random.random_integers(0, 99)
        while e == destination:
            destination = np.random.random_integers(0, 99)
        ev = EV(e, t_start, soc, e, destination)
        EV_list.append(ev)

    # print(CS_list)
    # print(EV_list)
    return CS_list, EV_list

def main():

    CS_list, EV_list =gen_envir()
    simple_graph = Graph()
    source = EV_list[0].source
    destination = EV_list[0].destination

    current_location = source

    while current_location != destination:
        print('\n', current_location, destination)
        came_from, cost_so_far = a_star_search(simple_graph, current_location, destination)
        path = reconstruct_path(came_from, current_location, destination)
        print("PATH: ", path)
        print("COST: ", cost_so_far[destination])
        current_location = path[1]

    return 0


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(graph.nodes[goal], graph.nodes[next])
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

if __name__ == "__main__":
    main()