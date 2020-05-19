import numpy as np
import matplotlib.pyplot as plt
import heapq
from Graph import Graph
from Graph import Graph_jeju
import pprint as pp

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
    def __init__(self, node_id, price, wait):
        self.id = node_id
        self.price = price
        self.waittime = wait
        self.chargingpower = 60 # kw

class EV:
    def __init__(self, id, t_start, soc, source, destination):
        self.id = id
        self.t_start = t_start
        self.SOC = soc
        self.source = source
        self.destination = destination
        self.maxBCAPA= 60  # kw
        self.curr_location = source
        self.next_location = source
        self.ECRate = 0.2 # kwh/km
        self.traveltime = 0 # hour
        self.charged = 0
        self.cs = None
        self.energyconsumption = 0.0
        self.chargingtime = 0.0
        self.chargingcost = 0.0
        self.waitingtime = 0.0
        self.csstayingtime = 0.0

def gen_envir():

    simple_graph = Graph()
    print(len(simple_graph.dict_edges))

    cs_location_list = [32, 37, 62, 67]
    CS_list = []
    for l in cs_location_list:
        cs = CS(l, 0.5, 0.3)
        CS_list.append(cs)
    EV_list = []

    for e in range(1000):

        t_start =  np.random.uniform(0, 96*15)
        soc = np.random.uniform(0.3, 0.5)
        while soc <= 0.0 or soc > 1.0 :
            soc = np.random.uniform(0.3, 0.5)

        source = np.random.random_integers(0, 99)
        # source = e
        destination = np.random.random_integers(0, 99)
        while destination in cs_location_list[:]:
            destination = np.random.random_integers(0, 99)
        ev = EV(e, t_start, soc, source, destination)
        EV_list.append(ev)


    return CS_list, EV_list, simple_graph



def update_envir_dist( CS_list, simple_graph, sim_time):
    count = 0
    for i in range(simple_graph.num_node):
        edges = simple_graph.dict_edges[i]
        # print(edges)
        for j, value in edges.items():
            count = count +1
            if edges[j]['road type'] == 1:
                edges[j]['velo'] = np.random.normal(120, 12)
                while edges[j]['velo'] <= 0:
                    edges[j]['velo'] = np.random.normal(120, 12)
            else:
                edges[j]['velo'] = np.random.normal(60, 6)
                while edges[j]['velo'] <= 1:
                    edges[j]['velo'] = np.random.normal(60, 6)
            edges[j]['weight'] = edges[j]['dist']


    for c in CS_list:
        c.price = np.random.normal(0.5, 0.02)
        c.waittime = np.random.normal(0.5, 0.05)
        while c.waittime < 0:
            c.waittime = np.random.normal(0.5, 0.05)

def update_envir_time( CS_list, simple_graph, sim_time):
    for i in range(simple_graph.num_node):
        edges = simple_graph.dict_edges[i]
        # print(edges)
        for j, value in edges.items():
            if edges[j]['road type'] == 1:
                edges[j]['velo'] = np.random.normal(120, 12)
                while edges[j]['velo'] <= 0:
                    edges[j]['velo'] =np.random.normal(120, 12)
            else:
                edges[j]['velo'] = np.random.normal(60, 6)
                while edges[j]['velo'] <= 1:
                    edges[j]['velo'] = np.random.normal(60, 6)
            edges[j]['weight'] = edges[j]['dist']/edges[j]['velo'] + (edges[j]['dist']*0.2)/60

    for c in CS_list:
        c.price = np.random.normal(0.5, 0.02)
        c.waittime = np.random.normal(0.5, 0.05)
        while c.waittime < 0:
            c.waittime = np.random.normal(0.5, 0.05)

def update_envir_time_power( CS_list, simple_graph, sim_time):
    for i in range(simple_graph.num_node):
        edges = simple_graph.dict_edges[i]
        # print(edges)
        for j, value in edges.items():
            if edges[j]['road type'] == 1:
                edges[j]['velo'] = np.random.normal(120, 12)
                while edges[j]['velo'] <= 0:
                    edges[j]['velo'] = np.random.normal(120, 12)
            else:
                edges[j]['velo'] = np.random.normal(60, 6)
                while edges[j]['velo'] <= 1:
                    edges[j]['velo'] = np.random.normal(60, 6)
            edges[j]['weight'] = edges[j]['dist']/edges[j]['velo'] + (edges[j]['dist']*0.2)/60

    for c in CS_list:
        c.price = np.random.normal(0.5, 0.1)
        c.waittime = np.random.normal(0.5, 0.1)
        while c.waittime < 0:
            c.waittime = np.random.normal(0.5, 0.1)




def update_ev(pev, simple_graph):
    dist = simple_graph.dict_edges[pev.curr_location][pev.next_location]['dist']
    velo = simple_graph.dict_edges[pev.curr_location][pev.next_location]['velo']
    pev.traveltime = pev.traveltime + dist/velo  # time unit : hour
    pev.SOC = pev.SOC - (dist*pev.ECRate)/pev.maxBCAPA
    pev.energyconsumption = pev.energyconsumption + dist*pev.ECRate
    # print(pev.curr_location, pev.traveltime, pev.SOC)

def routing_method_time(CS_list, pev, simple_graph):
    evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
    paths_info = []

    if pev.cs != None and pev.cs.id == pev.curr_location:
        # print(pev.curr_location)
        pev.charged = 1
        pev.waitingtime = pev.cs.waittime
        pev.chargingtime = (1.0 - pev.SOC) * pev.maxBCAPA / pev.cs.chargingpower
        pev.csstayingtime = pev.waitingtime + pev.chargingtime

    if pev.charged != 1:
        for cs in CS_list:

            came_from, cost_so_far = dijkstra_search(simple_graph, pev.curr_location, cs.id)
            front_path = reconstruct_path(came_from, pev.curr_location, cs.id)
            front_path_distance = simple_graph.get_path_distance(front_path)
            front_weight = cost_so_far[cs.id]

            if evcango <= front_path_distance:
                break

            came_from, cost_so_far = dijkstra_search(simple_graph, cs.id, pev.destination)
            rear_path = reconstruct_path(came_from, cs.id, pev.destination)
            rear_path_distance = simple_graph.get_path_distance(rear_path)
            rear_weight = cost_so_far[pev.destination]

            total_dist = front_path_distance + rear_path_distance
            charingtime = ((1.0-pev.SOC)*pev.maxBCAPA-total_dist*pev.ECRate)/cs.chargingpower
            # total_weight = front_weight + rear_weight + charingtime + cs.waittime
            total_weight = front_weight + rear_weight

            paths_info.append((total_weight, total_dist, front_path + rear_path[1:], cs))
    else:
        came_from, cost_so_far = dijkstra_search(simple_graph, pev.curr_location, pev.destination)
        path = reconstruct_path(came_from, pev.curr_location, pev.destination)
        path_distance = simple_graph.get_path_distance(path)
        path_weight = cost_so_far[pev.destination]
        paths_info.append((path_weight, path_distance, path, pev.cs))

    return paths_info

def routing_method_dist(CS_list, pev, simple_graph):
    evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
    paths_info = []

    if pev.cs != None and pev.cs.id == pev.curr_location:
        # print(pev.curr_location)
        pev.charged = 1
        pev.waitingtime = pev.cs.waittime
        pev.chargingtime = (1.0 - pev.SOC) * pev.maxBCAPA / pev.cs.chargingpower
        pev.csstayingtime = pev.waitingtime + pev.chargingtime

    if pev.charged != 1:
        for cs in CS_list:

            came_from, cost_so_far = dijkstra_search(simple_graph, pev.curr_location, cs.id)
            front_path = reconstruct_path(came_from, pev.curr_location, cs.id)
            front_path_distance = simple_graph.get_path_distance(front_path)
            front_weight = cost_so_far[cs.id]

            if evcango <= front_path_distance:
                break

            came_from, cost_so_far = dijkstra_search(simple_graph, cs.id, pev.destination)
            rear_path = reconstruct_path(came_from, cs.id, pev.destination)
            rear_path_distance = simple_graph.get_path_distance(rear_path)
            rear_weight = cost_so_far[pev.destination]

            charingtime = (1.0 - pev.SOC) * pev.maxBCAPA / cs.chargingpower
            total_weight = front_weight + rear_weight
            total_dist = front_path_distance + rear_path_distance
            paths_info.append((total_weight, total_dist, front_path + rear_path[1:], cs))
    else:
        came_from, cost_so_far = dijkstra_search(simple_graph, pev.curr_location, pev.destination)
        path = reconstruct_path(came_from, pev.curr_location, pev.destination)
        path_distance = simple_graph.get_path_distance(path)
        path_weight = cost_so_far[pev.destination]
        paths_info.append((path_weight, path_distance, path, pev.cs))

    return paths_info

def routing_method_time_power(CS_list, pev, simple_graph):
    evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
    paths_info = []

    if pev.cs != None and pev.cs.id == pev.curr_location:
        # print(pev.curr_location)
        pev.charged = 1
        pev.waitingtime = pev.cs.waittime
        pev.chargingtime = (1.0 - pev.SOC) * pev.maxBCAPA / pev.cs.chargingpower
        pev.csstayingtime = pev.waitingtime + pev.chargingtime

    if pev.charged != 1:
        for cs in CS_list:

            came_from, cost_so_far = dijkstra_search(simple_graph, pev.curr_location, cs.id)
            front_path = reconstruct_path(came_from, pev.curr_location, cs.id)
            front_path_distance = simple_graph.get_path_distance(front_path)
            front_weight = cost_so_far[cs.id]

            if evcango <= front_path_distance:
                break

            came_from, cost_so_far = dijkstra_search(simple_graph, cs.id, pev.destination)
            rear_path = reconstruct_path(came_from, cs.id, pev.destination)
            rear_path_distance = simple_graph.get_path_distance(rear_path)
            rear_weight = cost_so_far[pev.destination]

            charingtime = (1.0-pev.SOC)*pev.maxBCAPA/cs.chargingpower
            total_weight = front_weight + rear_weight  + charingtime + cs.waittime
            total_dist = front_path_distance+rear_path_distance
            paths_info.append((total_weight, total_dist, front_path + rear_path[1:], cs))
    else:
        came_from, cost_so_far = dijkstra_search(simple_graph, pev.curr_location, pev.destination)
        path = reconstruct_path(came_from, pev.curr_location, pev.destination)
        path_distance = simple_graph.get_path_distance(path)
        path_weight = cost_so_far[pev.destination]
        paths_info.append((path_weight, path_distance, path, pev.cs))

    return paths_info

def sim_algorithm_dist():
    np.random.seed(100)
    CS_list, EV_list, simple_graph = gen_envir()

    for ev in EV_list:
        print("EV ID:{} SOC:{} S:{} D:{} TA:{}".format(ev.id, ev.SOC, ev.source, ev.destination, ev.t_start))

    for cs in CS_list:
        print("CS ID:{} Price:{}".format(cs.id, cs.price))

    result_weight = []
    result_dist = []
    result_energyconsumption = []
    result_traveltime = []
    result_chargingtime = []
    result_waitingtime = []
    result_stayingtime = []

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n================================================================")
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        update_envir_dist(CS_list, simple_graph, sim_time)
        final_path = []
        weight = []
        final_path.append(pev.curr_location)


        while pev.next_location != pev.destination or pev.charged != 1:
            update_envir_dist(CS_list, simple_graph, sim_time)
            pev.curr_location = pev.next_location

            paths_info = routing_method_dist(CS_list, pev, simple_graph)
            paths_info.sort(key=lambda element: element[0])

            total_weight, total_dist, path, cs = paths_info[0]
            pev.cs = cs
            print(pev.charged, pev.traveltime, len(paths_info), path)

            pev.next_location = path[1]
            final_path.append(pev.next_location)
            weight.append(simple_graph.dict_edges[pev.curr_location][pev.next_location]["weight"])
            update_ev(pev, simple_graph)

        print(final_path)
        print(weight)
        sum_weight = sum(weight)
        sum_dist = simple_graph.get_path_distance(final_path)
        print("Total weight: ", sum_weight)
        print("Total Dist: ", sum_dist)
        print("Total Travel time: ", pev.traveltime)
        print("Total Chargingtime(wait+charing): ", pev.chargingtime)

        result_weight.append(sum_weight)
        result_dist.append(sum_dist)
        result_traveltime.append(pev.traveltime)
        result_chargingtime.append(pev.chargingtime)

        result_energyconsumption.append(pev.energyconsumption)

        result_waitingtime.append(pev.waitingtime)
        result_stayingtime.append(pev.csstayingtime)

    print('=============================================================================')
    print(result_weight)
    print(result_dist)
    print(result_traveltime)
    print(result_chargingtime)
    print("Total weight: ", sum(result_weight))
    print("Total Dist: ", sum(result_dist))
    print("Total Travel time: ", sum(result_traveltime))
    print("Total Charging time: ", sum(result_chargingtime))

    print("Total Energy Consumption: ", sum(result_energyconsumption))

    print("Total Waiting time: ", sum(result_waitingtime))
    print("Total Staying time: ", sum(result_stayingtime))

    return result_weight, result_dist, result_traveltime, result_chargingtime, EV_list, result_energyconsumption, result_waitingtime, result_stayingtime

def sim_algorithm_time():
    np.random.seed(100)
    CS_list, EV_list, simple_graph = gen_envir()

    for ev in EV_list:
        print("EV ID:{} SOC:{} S:{} D:{} TA:{}".format(ev.id, ev.SOC, ev.source, ev.destination, ev.t_start))

    for cs in CS_list:
        print("CS ID:{} Price:{}".format(cs.id, cs.price))

    result_weight = []
    result_dist = []
    result_traveltime = []
    result_chargingtime = []
    result_energyconsumption = []

    result_waitingtime = []
    result_stayingtime = []

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n================================================================")
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        update_envir_time(CS_list, simple_graph, sim_time)
        final_path = []
        weight = []
        final_path.append(pev.curr_location)

        while pev.next_location != pev.destination or pev.charged != 1:
            update_envir_time(CS_list, simple_graph, sim_time)
            pev.curr_location = pev.next_location

            paths_info = routing_method_time(CS_list, pev, simple_graph)
            paths_info.sort(key=lambda element: element[0])

            total_weight, total_dist, path, cs = paths_info[0]
            pev.cs = cs
            print(pev.charged, pev.traveltime, len(paths_info), path)

            pev.next_location = path[1]
            final_path.append(pev.next_location)
            weight.append(simple_graph.dict_edges[pev.curr_location][pev.next_location]["weight"])
            update_ev(pev, simple_graph)

        print(final_path)
        print(weight)
        sum_weight = sum(weight)
        sum_dist = simple_graph.get_path_distance(final_path)
        print("Total weight: ", sum_weight)
        print("Total Dist: ", sum_dist)
        print("Total Travel time: ", pev.traveltime)
        print("Total Chargingtime(wait+charing): ", pev.chargingtime)
        result_weight.append(sum_weight)
        result_dist.append(sum_dist)
        result_traveltime.append(pev.traveltime)
        result_chargingtime.append(pev.chargingtime)

        result_energyconsumption.append(pev.energyconsumption)

        result_waitingtime.append(pev.waitingtime)
        result_stayingtime.append(pev.csstayingtime)

    print('=============================================================================')
    print(result_weight)
    print(result_dist)
    print(result_traveltime)
    print(result_chargingtime)
    print("Total weight: ", sum(result_weight))
    print("Total Dist: ", sum(result_dist))
    print("Total Travel time: ", sum(result_traveltime))
    print("Total Charging time: ", sum(result_chargingtime))

    print("Total Energy Consumption: ", sum(result_energyconsumption))

    print("Total Waiting time: ", sum(result_waitingtime))
    print("Total Staying time: ", sum(result_stayingtime))

    return result_weight, result_dist, result_traveltime, result_chargingtime, EV_list, result_energyconsumption, result_waitingtime, result_stayingtime

def main():

    result_weight_dist, result_dist_dist, result_traveltime_dist, result_chargingtime_dist, pevdist, result_energy_dist, result_waitingtime_dist, result_stayingtime_dist = sim_algorithm_dist()

    result_weight_time, result_dist_time, result_traveltime_time, result_chargingtime_time, pevtime, result_energy_time, result_waitingtime_time, result_stayingtime_time = sim_algorithm_time()


    # result_weight_time_power, result_dist_time_power, result_traveltime_time_power, result_chargingtime_time_power, pevtimepower = sim_algorithm_time_power()
    # for ev in pevtime:
    #     print("EV ID:{} SOC:{} S:{} D:{} TA:{}".format(ev.id, ev.SOC, ev.source, ev.destination, ev.t_start))
    # for ev in pevtimepower:
    #     print("EV ID:{} SOC:{} S:{} D:{} TA:{}".format(ev.id, ev.SOC, ev.source, ev.destination, ev.t_start))
    print('===================================================================')
    print('Weight')
    print(sum(result_weight_time))
    print(sum(result_weight_dist))

    print('Travel time')
    print(sum(result_traveltime_time))
    print(sum(result_traveltime_dist))

    print('Distance')
    print(sum(result_dist_time))
    print(sum(result_dist_dist))

    print('Charging time')
    print(sum(result_chargingtime_time))
    print(sum(result_chargingtime_dist))

    print('Waiting time')
    print(sum(result_waitingtime_time))
    print(sum(result_waitingtime_dist))

    print('Staying time')
    print(sum(result_stayingtime_time))
    print(sum(result_stayingtime_dist))

    print('Energy')
    print(sum(result_energy_time))
    print(sum(result_energy_dist))

    # print(sum(result_traveltime_time_power))
    plt.plot(result_traveltime_time, label='Traveltime_time')
    plt.plot(result_traveltime_dist, label='Traveltime_dist', linestyle='--')
    plt.legend()
    # plt.plot(result_traveltime_time_power)
    plt.show()

    plt.plot(result_dist_time, label='Dist_time')
    plt.plot(result_dist_dist, label='Dist_dist', linestyle='--')
    # plt.plot(result_dist_time_power)
    plt.legend()
    plt.show()

    plt.plot(result_chargingtime_time, label='Chargingtime_time')
    plt.plot(result_chargingtime_dist, label='Chargingtime_dist', linestyle='--')
    # plt.plot(result_chargingtime_dist)
    plt.legend()
    plt.show()

    plt.plot(result_waitingtime_time, label='Waiting at CS_time')
    plt.plot(result_waitingtime_dist, label='Waiting at CS_dist', linestyle='--')
    # plt.plot(result_chargingtime_dist)
    plt.legend()
    plt.show()

    plt.plot(result_stayingtime_time, label='Staying at CS_time')
    plt.plot(result_stayingtime_dist, label='Staying at CS_dist', linestyle='--')
    # plt.plot(result_chargingtime_dist)
    plt.legend()
    plt.show()

    plt.plot(result_energy_time, label='Energy consumption_time')
    plt.plot(result_energy_dist, label='Energy consumption_dist', linestyle='--')
    # plt.plot(result_chargingtime_dist)
    plt.legend()
    plt.show()



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
            new_cost = cost_so_far[current] + graph.weight(current, next)
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
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        # print('path.append(current)', current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

if __name__ == "__main__":

    print('jeju')
    graph = Graph_jeju('data/20191001_5Min_modified.csv')

    for i in range(1):
        graph.source_node_set = list(graph.source_node_set)
        graph.destination_node_set = list(graph.destination_node_set)

        source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set))]
        destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set))]

        # source = 4060061500
        # destination = 4060061600

        print(source, destination)

        came_from, cost_so_far = a_star_search(graph, source, destination)
        path = reconstruct_path(came_from, source, destination)
        # print(path)
        dist = 0

        for i in range(len(path)-1):
            n, link_list = graph.get_link_id(path[i], path[i+1])
            dist += graph.link_data[link_list[0]]['LENGTH']
            # print(path[i], path[i+1], 'length: ', n, graph.link_data[link_list[0]]['LENGTH'])

        print(dist)
