import numpy as np
import matplotlib.pyplot as plt
import heapq
from Graph import Graph
from Graph import Graph_jeju
import pprint as pp
import copy
import datetime
import os
import csv

np.set_printoptions(precision = 4)
UNITtimecost = 0.75

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

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
    def __init__(self, node_id, price, wait, long, lat, alpha):
        self.id = node_id
        self.price = price
        self.waittime = wait
        self.chargingpower = 60 # kw
        self.alpha = alpha
        self.x = long
        self.y = lat


class EV:
    def __init__(self, id, t_start, soc, source, destination):
        self.id = id
        self.t_start = t_start
        self.chaging_effi = 0.9
        self.SOC = soc
        self.init_SOC = soc
        self.req_SOC = 0.8
        self.source = source
        self.destination = destination
        self.maxBCAPA= 60  # kw
        self.curr_location = source
        self.next_location = source
        self.ECRate = 0.2 # kwh/km
        self.traveltime = 0 # hour
        self.charged = 0
        self.cs=None
        self.csid = None
        self.energyconsumption = 0.0
        self.chargingtime = 0.0
        self.chargingcost = 0.0
        self.waitingtime = 0.0
        self.csstayingtime = 0.0
        self.drivingdistance = 0.0
        self.drivingtime = 0.0
        self.charingenergy = 0.0
        self.fdist=0
        self.rdist=0
        self.path=[]
        self.predic_totaltraveltime = 0.0




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
        # print('frontier.get()', current)
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def gen_envir_jeju(traffic_data_path, num_evs):
    np.random.seed(10)
    graph = Graph_jeju(traffic_data_path)

    EV_list = []

    for e in range(num_evs):
        t_start =  np.random.uniform(0, 1200)
        soc = np.random.uniform(0.3, 0.5)
        while soc <= 0.0 or soc > 1.0 :
            soc = np.random.uniform(0.3, 0.5)
        graph.source_node_set = list(graph.source_node_set)
        graph.destination_node_set = list(graph.destination_node_set)
        source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]

        destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]

        while destination in graph.cs_info.keys():
            destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        # source = 4080021700
        # destination = 4070008103
        ev = EV(e, t_start, soc, source, destination)
        EV_list.append(ev)

    CS_list = []
    for l in graph.cs_info:

        # print('gen cs')
        # alpha = np.random.uniform(0.03, 0.07)
        alpha = np.random.uniform(0.03, 0.07)

        price = np.random.normal(alpha, 0.15 * alpha)
        while price < 0:
            price = np.random.normal(alpha, 0.15 * alpha)

        waittime = np.random.normal(-1200 * (price - 0.07), 20)
        # waittime = np.random.normal(alpha, alpha*0.1)

        while waittime < 0:
            waittime = 0

        cs = CS(l, price, waittime, graph.cs_info[l]['long'], graph.cs_info[l]['lat'], alpha)
        CS_list.append(cs)

    return EV_list, CS_list, graph


def gen_evcs_random(cs):
    cs.price = np.random.normal(cs.alpha, 0.15 * cs.alpha)
    while cs.price < 0:
        cs.price = np.random.normal(cs.alpha, 0.15 * cs.alpha)

    cs.waittime = np.random.normal(-1200 * (cs.price - 0.07), 20)
    while cs.waittime < 0:
        cs.waittime = 0

def update_envir_timeweight(CS_list, graph, sim_time):

    time_idx = int(sim_time/5)
    count = 0
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']/velo

    for cs in CS_list:
        gen_evcs_random(cs)

def update_envir_costweight(CS_list, pev, graph, sim_time):
    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        gen_evcs_random(cs)
        avg_price += cs.price
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        # velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH'] * pev.ECRate * avg_price

def update_envir_costtimeweight(CS_list, pev, graph, sim_time):
    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        gen_evcs_random(cs)
        avg_price += cs.price
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        roadtime = graph.link_data[l_id]['LENGTH'] / velo
        roadcost = graph.link_data[l_id]['LENGTH'] * pev.ECRate * avg_price
        graph.link_data[l_id]['WEIGHT'] = roadtime*UNITtimecost + roadcost

def update_envir_distweight(CS_list, graph, sim_time):

    time_idx = int(sim_time/5)
    count = 0
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']

    for cs in CS_list:
        gen_evcs_random(cs)

def routing_time(CS_list, pev, simple_graph):
    evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
    paths_info = []

    if pev.cs != None and pev.cs.id == pev.curr_location:
        pev.charged = 1
        pev.waitingtime = pev.cs.waittime
        pev.chargingtime = (1.0 - pev.SOC) * pev.maxBCAPA / pev.cs.chargingpower
        pev.csstayingtime = pev.waitingtime + pev.chargingtime

    if pev.charged != 1:

        (x0, y0) = simple_graph.nodes_xy(pev.source)
        (x1, y1) = simple_graph.nodes_xy(pev.curr_location)
        (x2, y2) = simple_graph.nodes_xy(pev.destination)
        tmp = []

        for cs in CS_list:
            p1 = abs(x0 - x1) + abs(y0 - y1)
            p2 = abs(x1 - x2) + abs(y1 - y2)
            tmp.append((cs, p1+p2))
        tmp.sort(key=lambda element: element[1])

        # pp.pprint(tmp)

        if len(tmp) != 0:
            CS_list=[]
            for t in tmp[:10]:
                cs,_ = t
                CS_list.append(cs)


        print('CS', len(CS_list))

        for cs in CS_list:
            came_from, cost_so_far = a_star_search(simple_graph, pev.curr_location, cs.id)
            front_path = reconstruct_path(came_from, pev.curr_location, cs.id)
            front_path_distance = simple_graph.get_path_distance(front_path)
            front_weight = cost_so_far[cs.id]
            if evcango <= front_path_distance:
                continue
            came_from, cost_so_far = a_star_search(simple_graph, cs.id, pev.destination)
            rear_path = reconstruct_path(came_from, cs.id, pev.destination)
            rear_path_distance = simple_graph.get_path_distance(rear_path)
            rear_weight = cost_so_far[pev.destination]
            total_dist = front_path_distance + rear_path_distance
            chargingtime = ((1.0-pev.SOC)*pev.maxBCAPA-total_dist*pev.ECRate)/cs.chargingpower
            total_weight = front_weight + rear_weight
            paths_info.append((total_weight, total_dist, front_path + rear_path[1:], cs))
    else:
        came_from, cost_so_far = a_star_search(simple_graph, pev.curr_location, pev.destination)
        path = reconstruct_path(came_from, pev.curr_location, pev.destination)
        path_distance = simple_graph.get_path_distance(path)
        path_weight = cost_so_far[pev.destination]
        paths_info.append((path_weight, path_distance, path, pev.cs))

    return paths_info

def sim_main():
    print('jeju')
    EV_list, CS_list, graph = gen_envir_jeju('data/20191001_5Min_modified.csv', 10000)

    sim_n = 0
    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))

        update_envir_distweight(CS_list, graph, sim_time)
        final_path = []
        weight = []
        final_path.append(pev.curr_location)

        while pev.next_location != pev.destination or pev.charged != 1:
            update_envir_distweight(CS_list, graph, sim_time)
            pev.curr_location = pev.next_location

            paths_info = routing_time(CS_list, pev, graph)
            paths_info.sort(key=lambda element: element[0])

            print('paths_info', len(paths_info))
            total_weight, total_dist, path, cs = paths_info[0]
            pev.cs = cs
            print(pev.charged, pev.traveltime, path)
            pev.next_location = path[1]
            final_path.append(pev.next_location)
            weight.append(graph.weight(pev.curr_location, pev.next_location))
            sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location,sim_time)

        print(final_path)
        print(weight)
        sum_weight = sum(weight)
        sum_dist = graph.get_path_distance(final_path)
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        print("Total weight: ", sum_weight)
        print("Total Dist(km): ", sum_dist)
        print("Total Travel time(h): ", pev.traveltime)
        print("Total Chargingtime(wait+charing): ", pev.chargingtime)
        sim_n += 1

def heuristic_astar(a, b):
    (x1, y1) = a
    (x2, y2) = b

    return abs(x1 - x2) + abs(y1 - y2)

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




def sim_main_source_to_cs():

    np.random.seed(10)
    print('jeju')
    EV_list, CS_list, graph = gen_envir_jeju('data/20191001_5Min_modified.csv', 100)

    sim_n = 0
    expected_set = []
    real_set = []

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, None, pev.t_start))
        evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
        start = pev.source
        # goal = pev.destination
        path_list = PriorityQueue()
        update_envir_timeweight(CS_list, graph, sim_time)

        for cs in CS_list:
            goal = cs.id
            came_from, cost_so_far = a_star_search(graph, start, goal)
            path = reconstruct_path(came_from, start, goal)
            path_distance = graph.get_path_distance(path)

            if path_distance <= evcango:
                d_time = graph.get_path_weight(path)
                dist = graph.get_path_distance(path)
                c_time = (pev.maxBCAPA - dist*0.2)/60
                w = d_time + cs.waittime + c_time
                path_list.put((path, cs, d_time, c_time), w)
                # print('CS', cs.id,' W: ', w)


        path, cs, d, c = path_list.get()
        path_distance = graph.get_path_distance(path)
        print(path)
        print('cs alpha:', cs.alpha)
        print('Price: ', cs.price)
        print('Distance(km): ', path_distance)
        print('Expected Driving time(h): ', d)
        print('Expected Charging time(h): ', c)
        print('Expected Waiting time(h): ', cs.waittime)
        print('Expected Total time(h): ',  cs.waittime+d+c)
        expected_set.append((d,c,cs.waittime,cs.waittime+d+c))



        for i in range(len(path) - 1):

            time_idx = int(sim_time / 5)
            fromenode = path[i]
            tonode = path[i + 1]
            dist = graph.distance(fromenode, tonode)
            velo = graph.velocity(fromenode,tonode, time_idx)
            d_time = dist/velo
            pev.energyconsumption += dist*0.2
            pev.drivingtime += d_time
            pev.drivingdistance += dist

            print(sim_time, fromenode, tonode, dist, velo, d_time)
            sim_time += d_time*60

        cs.price = np.random.normal(cs.alpha, 0.15 * cs.alpha)
        while cs.price < 0:
            cs.price = np.random.normal(cs.alpha, 0.15 * cs.alpha)

        cs.waittime = np.random.normal(cs.alpha, cs.alpha*0.1)

        while cs.waittime < 0:
            cs.waittime = 0


        pev.waitingtime = cs.waittime
        sim_time += cs.waittime
        pev.chargingtime = (pev.maxBCAPA - pev.energyconsumption)/60
        sim_time += pev.chargingtime

        print('cs alpha:', cs.alpha)
        print('Price: ', cs.price)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(h): ', pev.drivingtime)
        print('Real Charging time(h): ', pev.chargingtime)
        print('Real Waiting time(h): ', cs.waittime)
        print('Real Total time(h): ', cs.waittime + pev.chargingtime + pev.drivingtime)
        real_set.append((pev.drivingtime, pev.chargingtime, cs.waittime, cs.waittime+ pev.chargingtime + pev.drivingtime))

        sim_n += 1

    edt_list = []
    ect_list = []
    ewt_list = []
    ett_list = []

    rdt_list = []
    rct_list = []
    rwt_list = []
    rtt_list = []

    for i in range(len(real_set)):
        edt, ect, ewt, ett = expected_set[i]
        rdt, rct, rwt, rtt = real_set[i]
        edt_list.append(edt)
        ect_list.append(ect)
        ewt_list.append(ewt)
        ett_list.append(ett)

        rdt_list.append(rdt)
        rct_list.append(rct)
        rwt_list.append(rwt)
        rtt_list.append(rtt)

    plt.plot(edt_list, label='Expected Driving time')
    plt.plot(rdt_list, label='Real Driving time', linestyle='--')
    plt.legend()
    # plt.plot(result_traveltime_time_power)
    plt.show()

    plt.plot(ect_list, label='Expected Charging time')
    plt.plot(rct_list, label='Real Charging time', linestyle='--')
    plt.legend()
    # plt.plot(result_traveltime_time_power)
    plt.show()

    plt.plot(ewt_list, label='Expected Waiting time')
    plt.plot(rwt_list, label='Real Waiting time', linestyle='--')
    plt.legend()
    # plt.plot(result_traveltime_time_power)
    plt.show()

    plt.plot(ett_list, label='Expected Total time')
    plt.plot(rtt_list, label='Real Total time', linestyle='--')
    plt.legend()
    # plt.plot(result_traveltime_time_power)
    plt.show()

def sim_main_first_time_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 최소시간을 갖는 충전경로를 찾는 방법으로 미래에 정보 없이 현재 상황에서 충전경로를 설정
    # Min total travel time
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
        start = pev.source
        end = pev.destination

        paths_info = PriorityQueue()
        update_envir_timeweight(CS_list, graph, sim_time)
        for cs in CS_list:
            evcs = cs.id

            came_from, cost_so_far = a_star_search(graph, start, evcs)
            front_path = reconstruct_path(came_from, start, evcs)
            front_path_distance = graph.get_path_distance(front_path)

            if front_path_distance > evcango:
                continue

            came_from, cost_so_far = a_star_search(graph, evcs, end)
            rear_path = reconstruct_path(came_from, evcs, end)
            rear_path_distance = graph.get_path_distance(rear_path)

            final_path = front_path + rear_path[1:]

            dist = graph.get_path_distance(final_path)
            d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
            remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
            c_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
            c_time = (c_energy/(cs.chargingpower*pev.chaging_effi))*60
            w = d_time + cs.waittime + c_time
            totaltraveltime = d_time + cs.waittime + c_time

            paths_info.put((remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)

        remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
        pev.predic_totaltraveltime = totaltraveltime
        pev.rdist = rpath_dist
        path_distance = graph.get_path_distance(path)
        pev.cs = evcs
        pev.path = copy.deepcopy(path)

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]

            if evcs.id == fromenode and pev.charged !=1:
                # print(evcs.id, evcs.waittime, evcs.price)
                gen_evcs_random(evcs)
                # print(evcs.id, evcs.waittime, evcs.price)
                pev.charingenergy = pev.maxBCAPA*pev.req_SOC - pev.SOC*pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                print(pev.charingenergy)
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy/(evcs.chargingpower*pev.chaging_effi))*60
                sim_time += pev.chargingtime
                sim_time += evcs.waittime
                pev.waitingtime = evcs.waittime
                pev.charged = 1
            sim_time = update_ev(pev, graph, fromenode, tonode, sim_time)

        print()

        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1

    return EV_list

def sim_main_first_cost_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 최소시간을 갖는 충전경로를 찾는 방법으로 미래에 정보 없이 현재 상황에서 충전경로를 설정
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0


    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
        start = pev.source
        end = pev.destination

        paths_info = PriorityQueue()
        update_envir_costweight(CS_list, pev, graph, sim_time)
        for cs in CS_list:
            evcs = cs.id

            came_from, cost_so_far = a_star_search(graph, start, evcs)
            front_path = reconstruct_path(came_from, start, evcs)
            front_path_distance = graph.get_path_distance(front_path)

            if front_path_distance > evcango:
                continue

            came_from, cost_so_far = a_star_search(graph, evcs, end)
            rear_path = reconstruct_path(came_from, evcs, end)
            rear_path_distance = graph.get_path_distance(rear_path)

            final_path = front_path + rear_path[1:]

            dist = graph.get_path_distance(final_path)
            cost_road = graph.get_path_weight(final_path)*60
            d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
            remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
            c_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
            c_time = (c_energy/(cs.chargingpower*pev.chaging_effi))*60
            w = cost_road + c_energy*cs.price
            totaltraveltime = d_time + cs.waittime + c_time

            paths_info.put((
                           remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs,
                           d_time, c_time, c_energy, totaltraveltime), w)

        remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
        pev.predic_totaltraveltime = totaltraveltime
        pev.rdist = rpath_dist
        path_distance = graph.get_path_distance(path)
        pev.cs = evcs
        pev.path = copy.deepcopy(path)

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]

            if evcs.id == fromenode and pev.charged !=1:

                gen_evcs_random(evcs)

                pev.charingenergy = pev.maxBCAPA*pev.req_SOC - pev.SOC*pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                print(pev.charingenergy)
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy/(evcs.chargingpower*pev.chaging_effi))*60
                sim_time += pev.chargingtime
                sim_time += evcs.waittime
                pev.waitingtime = evcs.waittime
                pev.charged = 1
            sim_time = update_ev(pev, graph, fromenode, tonode, sim_time)

        print()

        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1




    return EV_list

def sim_main_first_cost_time_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 최소시간을 갖는 충전경로를 찾는 방법으로 미래에 정보 없이 현재 상황에서 충전경로를 설정
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0


    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
        start = pev.source
        end = pev.destination

        paths_info = PriorityQueue()
        update_envir_costtimeweight(CS_list, pev, graph, sim_time)
        for cs in CS_list:
            evcs = cs.id

            came_from, cost_so_far = a_star_search(graph, start, evcs)
            front_path = reconstruct_path(came_from, start, evcs)
            front_path_distance = graph.get_path_distance(front_path)

            if front_path_distance > evcango:
                continue

            came_from, cost_so_far = a_star_search(graph, evcs, end)
            rear_path = reconstruct_path(came_from, evcs, end)
            rear_path_distance = graph.get_path_distance(rear_path)

            final_path = front_path + rear_path[1:]

            dist = graph.get_path_distance(final_path)
            cost_road = graph.get_path_weight(final_path)*60 # cost_road = road cost + road time cost
            d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
            remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
            c_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
            c_time = (c_energy/(cs.chargingpower*pev.chaging_effi))*60
            w = cost_road + UNITtimecost*(cs.waittime+c_time) + c_energy*cs.price  # road cost + road time cost + (waiting time + charging time) + charging cost
            totaltraveltime = d_time + cs.waittime + c_time

            paths_info.put((
                           remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs,
                           d_time, c_time, c_energy, totaltraveltime), w)

        remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
        pev.predic_totaltraveltime = totaltraveltime
        pev.rdist = rpath_dist
        path_distance = graph.get_path_distance(path)
        pev.cs = evcs
        pev.path = copy.deepcopy(path)

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]

            if evcs.id == fromenode and pev.charged !=1:

                gen_evcs_random(evcs)

                pev.charingenergy = pev.maxBCAPA*pev.req_SOC - pev.SOC*pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                print(pev.charingenergy)
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy/(evcs.chargingpower*pev.chaging_effi))*60
                sim_time += pev.chargingtime
                sim_time += evcs.waittime
                pev.waitingtime = evcs.waittime
                pev.charged = 1
            sim_time = update_ev(pev, graph, fromenode, tonode, sim_time)

        print()

        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1

    return EV_list

def sim_main_first_dist_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 최소시간을 갖는 충전경로를 찾는 방법으로 미래에 정보 없이 현재 상황에서 충전경로를 설정
    # Min total distance

    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        evcango = pev.SOC * pev.maxBCAPA / pev.ECRate
        start = pev.source
        end = pev.destination

        paths_info = PriorityQueue()
        update_envir_distweight(CS_list, graph, sim_time)
        for cs in CS_list:
            evcs = cs.id

            came_from, cost_so_far = a_star_search(graph, start, evcs)
            front_path = reconstruct_path(came_from, start, evcs)
            front_path_distance = graph.get_path_distance(front_path)

            if front_path_distance > evcango:
                continue

            came_from, cost_so_far = a_star_search(graph, evcs, end)
            rear_path = reconstruct_path(came_from, evcs, end)
            rear_path_distance = graph.get_path_distance(rear_path)

            final_path = front_path + rear_path[1:]

            dist = graph.get_path_distance(final_path)
            d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
            remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
            c_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
            c_time = (c_energy/(cs.chargingpower*pev.chaging_effi))*60
            w = dist
            totaltraveltime = d_time + cs.waittime + c_time

            paths_info.put( (remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs, d_time, c_time, c_energy, totaltraveltime) , w)

        remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
        pev.predic_totaltraveltime = totaltraveltime
        pev.rdist = rpath_dist
        path_distance = graph.get_path_distance(path)
        pev.cs = evcs
        pev.path = copy.deepcopy(path)


        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]

            if evcs.id == fromenode and pev.charged !=1:

                gen_evcs_random(evcs)

                pev.charingenergy = pev.maxBCAPA*pev.req_SOC - pev.SOC*pev.maxBCAPA
                pev.chargingcost = pev.charingenergy*evcs.price
                print(pev.charingenergy)
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy/(evcs.chargingpower*pev.chaging_effi))*60
                sim_time += pev.chargingtime
                sim_time += evcs.waittime
                pev.waitingtime = evcs.waittime
                pev.charged = 1
            sim_time = update_ev(pev, graph, fromenode, tonode, sim_time)

        print()
        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)
        sim_n += 1

    return EV_list

def update_ev(pev, simple_graph, fnode, tnode, sim_time):
    time_idx = int(sim_time / 5)
    dist = simple_graph.distance(fnode, tnode)
    velo = simple_graph.velocity(fnode, tnode, time_idx)

    time_diff = (dist/velo) * 60
    pev.traveltime = pev.traveltime + time_diff
    pev.drivingtime += time_diff
    pev.drivingdistance += dist
    soc_before = pev.SOC
    pev.SOC = pev.SOC - (dist * pev.ECRate) / pev.maxBCAPA
    pev.energyconsumption = pev.energyconsumption + dist * pev.ECRate
    # print("fnode {} tnode {} dist {} velo {} soc_b {} soc_a {}".format(fnode, tnode, dist, velo, soc_before, pev.SOC))
    if pev.charged != 1:
        pev.fdist += dist

    return sim_time + time_diff

def sim_result_presentation(time_EV_list, dist_EV_list, cost_EV_list, costtime_EV_list):

    now = datetime.datetime.now()
    basepath = os.getcwd()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} result'.format(now.month, now.day, now.hour, now.minute)
    print(os.path.join(basepath, resultdir))
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    time_rdt_list = []
    time_rct_list = []
    time_rwt_list = []
    time_rtt_list = []
    time_rce_list = []
    time_final_SOC = []
    time_driving_dist = []
    time_cs_list = []
    time_init_SOC_list = []
    time_fdist_list = []
    time_charging_cost_list = []
    time_predic_totaltraveltime_list = []

    dist_rdt_list = []
    dist_rct_list = []
    dist_rwt_list = []
    dist_rtt_list = []
    dist_rce_list = []
    dist_final_SOC = []
    dist_driving_dist = []
    dist_cs_list = []
    dist_init_SOC_list = []
    dist_fdist_list = []
    dist_charging_cost_list = []
    dist_predic_totaltraveltime_list = []

    cost_rdt_list = []
    cost_rct_list = []
    cost_rwt_list = []
    cost_rtt_list = []
    cost_rce_list = []
    cost_final_SOC = []
    cost_driving_dist = []
    cost_cs_list = []
    cost_init_SOC_list = []
    cost_fdist_list = []
    cost_charging_cost_list = []
    cost_predic_totaltraveltime_list = []

    costtime_rdt_list = []
    costtime_rct_list = []
    costtime_rwt_list = []
    costtime_rtt_list = []
    costtime_rce_list = []
    costtime_final_SOC = []
    costtime_driving_dist = []
    costtime_cs_list = []
    costtime_init_SOC_list = []
    costtime_fdist_list = []
    costtime_charging_cost_list = []
    costtime_predic_totaltraveltime_list = []

    for pev in time_EV_list:
        time_cs_list.append(pev.cs.id)
        time_init_SOC_list.append(pev.init_SOC)
        time_fdist_list.append(pev.fdist)
        time_charging_cost_list.append(pev.chargingcost)
        time_rdt_list.append(pev.drivingtime)
        time_rct_list.append(pev.chargingtime)
        time_rwt_list.append(pev.waitingtime)
        time_rtt_list.append(pev.waitingtime + pev.chargingtime + pev.drivingtime)
        time_rce_list.append(pev.charingenergy)
        time_final_SOC.append(pev.SOC)
        time_driving_dist.append(pev.drivingdistance)
        time_predic_totaltraveltime_list.append(pev.predic_totaltraveltime)

    for pev in dist_EV_list:
        dist_cs_list.append(pev.cs.id)
        dist_init_SOC_list.append(pev.init_SOC)
        dist_fdist_list.append(pev.fdist)
        dist_charging_cost_list.append(pev.chargingcost)
        dist_rdt_list.append(pev.drivingtime)
        dist_rct_list.append(pev.chargingtime)
        dist_rwt_list.append(pev.waitingtime)
        dist_rtt_list.append(pev.waitingtime + pev.chargingtime + pev.drivingtime)
        dist_rce_list.append(pev.charingenergy)
        dist_final_SOC.append(pev.SOC)
        dist_driving_dist.append(pev.drivingdistance)
        dist_predic_totaltraveltime_list.append(pev.predic_totaltraveltime)

    for pev in cost_EV_list:
        cost_cs_list.append(pev.cs.id)
        cost_init_SOC_list.append(pev.init_SOC)
        cost_fdist_list.append(pev.fdist)
        cost_charging_cost_list.append(pev.chargingcost)
        cost_rdt_list.append(pev.drivingtime)
        cost_rct_list.append(pev.chargingtime)
        cost_rwt_list.append(pev.waitingtime)
        cost_rtt_list.append(pev.waitingtime + pev.chargingtime + pev.drivingtime)
        cost_rce_list.append(pev.charingenergy)
        cost_final_SOC.append(pev.SOC)
        cost_driving_dist.append(pev.drivingdistance)
        cost_predic_totaltraveltime_list.append(pev.predic_totaltraveltime)

    for pev in costtime_EV_list:
        costtime_cs_list.append(pev.cs.id)
        costtime_init_SOC_list.append(pev.init_SOC)
        costtime_fdist_list.append(pev.fdist)
        costtime_charging_cost_list.append(pev.chargingcost)
        costtime_rdt_list.append(pev.drivingtime)
        costtime_rct_list.append(pev.chargingtime)
        costtime_rwt_list.append(pev.waitingtime)
        costtime_rtt_list.append(pev.waitingtime + pev.chargingtime + pev.drivingtime)
        costtime_rce_list.append(pev.charingenergy)
        costtime_final_SOC.append(pev.SOC)
        costtime_driving_dist.append(pev.drivingdistance)
        costtime_predic_totaltraveltime_list.append(pev.predic_totaltraveltime)


    for i in range(len(time_EV_list)):
        tpev = time_EV_list[i]
        dpev = dist_EV_list[i]
        cpev = cost_EV_list[i]
        ctpev = costtime_EV_list[i]

        x_a = []
        y_a = []
        for nid in tpev.path:
            x, y = graph.nodes_xy(nid)
            x_a.append(x)
            y_a.append(y)
        plt.plot(x_a, y_a, '--', label='time')
        cs_x, cs_y = graph.nodes_xy(tpev.cs.id)
        plt.plot(cs_x, cs_y, 'o', label='time_EVCS')

        x_b = []
        y_b = []
        for nid in dpev.path:
            x, y = graph.nodes_xy(nid)
            x_b.append(x)
            y_b.append(y)
        plt.plot(x_b, y_b, '-', label='dist')
        cs_x, cs_y = graph.nodes_xy(dpev.cs.id)
        plt.plot(cs_x, cs_y, 'o', label='dist_EVCS')

        x_c = []
        y_c = []
        for nid in cpev.path:
            x, y = graph.nodes_xy(nid)
            x_c.append(x)
            y_c.append(y)
        plt.plot(x_c, y_c, ':', label='cost')
        cs_x, cs_y = graph.nodes_xy(cpev.cs.id)
        plt.plot(cs_x, cs_y, 'd', label='cost_EVCS')

        x_c = []
        y_c = []
        for nid in ctpev.path:
            x, y = graph.nodes_xy(nid)
            x_c.append(x)
            y_c.append(y)
        plt.plot(x_c, y_c, ':', label='costtime')
        cs_x, cs_y = graph.nodes_xy(ctpev.cs.id)
        plt.plot(cs_x, cs_y, 'D', label='costtime_EVCS')

        s_x, s_y = graph.nodes_xy(tpev.source)
        plt.plot(s_x, s_y, 'p', label='Source')
        d_x, d_y = graph.nodes_xy(tpev.destination)
        plt.plot(d_x, d_y, 'h', label='Destination')

        plt.legend()
        fig = plt.gcf()
        fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='#eeeeee', dpi=300)
        plt.clf()

    plt.figure(figsize=(12, 6), dpi=300)

    plt.title('Diff predic & real total travel time (time)')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(time_predic_totaltraveltime_list, label='Predic totaltraveltime')
    plt.plot(time_rtt_list, label='Real totaltraveltime', linestyle='--')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Diff predic-real total travel time (time).png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Diff predic & real total travel time (dist)')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(dist_predic_totaltraveltime_list, label='Predic totaltraveltime')
    plt.plot(dist_rtt_list, label='Real totaltraveltime', linestyle='--')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Diff predic-real total travel time (dist).png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Diff predic & real total travel time (cost)')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(cost_predic_totaltraveltime_list, label='Predic totaltraveltime')
    plt.plot(cost_rtt_list, label='Real totaltraveltime', linestyle='--')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Diff predic-real total travel time (cost).png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Diff predic & real total travel time (costtime)')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(costtime_predic_totaltraveltime_list, label='Predic totaltraveltime')
    plt.plot(costtime_rtt_list, label='Real totaltraveltime', linestyle='--')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Diff predic-real total travel time (costtime).png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12,6), dpi=300)
    plt.title('Charging Cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    plt.plot(time_charging_cost_list, label='TIME Charging Cost')
    plt.plot(dist_charging_cost_list, label='DIST Charging Cost', linestyle='--')
    plt.plot(cost_charging_cost_list, label='COST Charging Cost', linestyle=':')
    plt.plot(costtime_charging_cost_list, label='COSTtime Charging Cost', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Selected EVCS')
    plt.xlabel('EV index')
    plt.ylabel('EVCS ID')
    plt.plot(range(len(time_cs_list)), time_cs_list, 'x', label='TIME EVCS')
    plt.plot(range(len(time_cs_list)), dist_cs_list, '+', label='DIST EVCS')
    plt.plot(range(len(time_cs_list)), cost_cs_list, 'd', label='COST EVCS')
    plt.plot(range(len(time_cs_list)), costtime_cs_list, 'd', label='COSTtime EVCS')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Driving distance')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    plt.plot(time_driving_dist, label='TIME Driving distance')
    plt.plot(dist_driving_dist, label='DIST Driving distance', linestyle='--')
    plt.plot(cost_driving_dist, label='COST Driving distance', linestyle=':')
    plt.plot(costtime_driving_dist, label='COSTtime Driving distance', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving distance.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12, 6), dpi=300)
    plt.title('Distance from S to EVCS')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    plt.plot(time_fdist_list, label='TIME front distance')
    plt.plot(dist_fdist_list, label='DIST front distance', linestyle='--')
    plt.plot(cost_fdist_list, label='COST front distance', linestyle=':')
    plt.plot(costtime_fdist_list, label='COSTtime front distance', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Distance from S to EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12, 6), dpi=300)
    plt.title('Driving time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(time_rdt_list, label='TIME Driving time')
    plt.plot(dist_rdt_list, label='DIST Driving time', linestyle='--')
    plt.plot(cost_rdt_list, label='COST Driving time', linestyle=':')
    plt.plot(costtime_rdt_list, label='COSTtime Driving time', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12, 5), dpi=300)
    plt.title('Charging energy')
    plt.xlabel('EV ID')
    plt.ylabel('Energy(kWh)')
    plt.plot(time_rce_list, label='TIME Charging energy')
    plt.plot(dist_rce_list, label='DIST Charging energy', linestyle='--')
    plt.plot(cost_rce_list, label='COST Charging energy', linestyle=':')
    plt.plot(costtime_rce_list, label='COSTtime Charging energy', linestyle=':')
    plt.legend()
    fig.savefig('{}/Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12, 6), dpi=300)
    plt.title('Charging time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(time_rct_list, label='TIME Charging time')
    plt.plot(dist_rct_list, label='DIST Charging time', linestyle='--')
    plt.plot(cost_rct_list, label='COST Charging time', linestyle=':')
    plt.plot(costtime_rct_list, label='COSTtime Charging time', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Charging time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12, 6), dpi=300)
    plt.title('Waiting time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(time_rwt_list, label='TIME Waiting time')
    plt.plot(dist_rwt_list, label='DIST Waiting time', linestyle='--')
    plt.plot(cost_rwt_list, label='COST Waiting time', linestyle=':')
    plt.plot(costtime_rwt_list, label='COSTtime Waiting time', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Waiting time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # plt.figure(figsize=(12, 6), dpi=300)
    plt.title('Total travel time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    plt.plot(time_rtt_list, label='TIME Total travel time')
    plt.plot(dist_rtt_list, label='DIST Total travel time', linestyle='--')
    plt.plot(cost_rtt_list, label='COST Total travel time', linestyle=':')
    plt.plot(costtime_rtt_list, label='COSTtime Total travel time', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Total travel time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('EV SOC')
    plt.xlabel('EV ID')
    plt.ylabel('SOC(%)')
    plt.plot(time_init_SOC_list, label='TIME init_SOC')
    plt.plot(dist_init_SOC_list, label='DIST init_SOC', linestyle='--')
    plt.plot(cost_init_SOC_list, label='COST init_SOC', linestyle=':')
    plt.plot(costtime_init_SOC_list, label='COSTtime init_SOC', linestyle=':')
    plt.plot(time_final_SOC, label='TIME Final SOC')
    plt.plot(dist_final_SOC, label='DIST Final SOC', linestyle='--')
    plt.plot(cost_final_SOC, label='COST Final SOC', linestyle=':')
    plt.plot(costtime_final_SOC, label='COSTtime Final SOC', linestyle=':')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/ev SOC.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    # print('Diff predic & real total travel time (time)')
    # print(time_predic_totaltraveltime_list)
    # print(time_rtt_list)
    # print(sum(time_predic_totaltraveltime_list) / len(time_predic_totaltraveltime_list))
    # print(sum(time_rtt_list) / len(time_rtt_list))
    #
    # print('Diff predic & real total travel time (dist)')
    # print(dist_predic_totaltraveltime_list)
    # print(dist_rtt_list)
    # print(sum(dist_predic_totaltraveltime_list) / len(dist_predic_totaltraveltime_list))
    # print(sum(dist_rtt_list) / len(dist_rtt_list))
    #
    # print('Diff predic & real total travel time (cost)')
    # print(cost_predic_totaltraveltime_list)
    # print(cost_rtt_list)
    # print(sum(cost_predic_totaltraveltime_list) / len(cost_predic_totaltraveltime_list))
    # print(sum(cost_rtt_list) / len(cost_rtt_list))
    #
    # print('EVCS')
    # print(time_cs_list)
    # print(dist_cs_list)
    # print(cost_cs_list)
    #
    # print('Charging Cost')
    # print(time_charging_cost_list)
    # print(dist_charging_cost_list)
    # print(cost_charging_cost_list)
    # print(sum(time_charging_cost_list) / len(time_charging_cost_list))
    # print(sum(dist_charging_cost_list) / len(dist_charging_cost_list))
    # print(sum(cost_charging_cost_list) / len(cost_charging_cost_list))
    #
    # print('Driving distance')
    # print(time_driving_dist)
    # print(dist_driving_dist)
    # print(cost_driving_dist)
    # print(sum(time_driving_dist) / len(time_driving_dist))
    # print(sum(dist_driving_dist) / len(dist_driving_dist))
    # print(sum(cost_driving_dist) / len(cost_driving_dist))
    #
    # print('Distance from S to EVCS')
    # print(time_fdist_list)
    # print(dist_fdist_list)
    # print(cost_fdist_list)
    # print(sum(time_fdist_list) / len(time_fdist_list))
    # print(sum(dist_fdist_list) / len(dist_fdist_list))
    # print(sum(cost_fdist_list) / len(cost_fdist_list))
    #
    # print('Driving time')
    # print(time_rdt_list)
    # print(dist_rdt_list)
    # print(cost_rdt_list)
    # print(sum(time_rdt_list) / len(time_rdt_list))
    # print(sum(dist_rdt_list) / len(dist_rdt_list))
    # print(sum(cost_rdt_list) / len(cost_rdt_list))
    #
    # print('Charging energy')
    # print(time_rce_list)
    # print(dist_rce_list)
    # print(cost_rce_list)
    # print(sum(time_rce_list) / len(time_rce_list))
    # print(sum(dist_rce_list) / len(dist_rce_list))
    # print(sum(cost_rce_list) / len(cost_rce_list))
    #
    #
    # print('Charging time')
    # print(time_rct_list)
    # print(dist_rct_list)
    # print(cost_rct_list)
    # print(sum(time_rct_list) / len(time_rct_list))
    # print(sum(dist_rct_list) / len(dist_rct_list))
    # print(sum(cost_rct_list) / len(cost_rct_list))
    #
    # print('Waiting time')
    # print(time_rwt_list)
    # print(dist_rwt_list)
    # print(cost_rwt_list)
    # print(sum(time_rwt_list) / len(time_rwt_list))
    # print(sum(dist_rwt_list) / len(dist_rwt_list))
    # print(sum(cost_rwt_list) / len(cost_rwt_list))
    #
    # print('Total travel time')
    # print(time_rtt_list)
    # print(dist_rtt_list)
    # print(cost_rtt_list)
    # print(sum(time_rtt_list) / len(time_rtt_list))
    # print(sum(dist_rtt_list) / len(dist_rtt_list))
    # print(sum(cost_rtt_list) / len(cost_rtt_list))
    #
    # print('init_SOC')
    # print(time_init_SOC_list)
    # print(dist_init_SOC_list)
    # print(cost_init_SOC_list)
    # print(sum(time_init_SOC_list) / len(time_init_SOC_list))
    # print(sum(dist_init_SOC_list) / len(dist_init_SOC_list))
    # print(sum(cost_init_SOC_list) / len(cost_init_SOC_list))
    #
    # print('Final SOC')
    # print(time_final_SOC)
    # print(dist_final_SOC)
    # print(cost_final_SOC)
    # print(sum(time_final_SOC) / len(time_final_SOC))
    # print(sum(dist_final_SOC) / len(dist_final_SOC))
    # print(sum(cost_final_SOC) / len(cost_final_SOC))

    fw = open('{}/result.txt'.format(resultdir), 'w', encoding='UTF8')

    fw.write(str('Diff predic & real total travel time (time)') + '\n')
    fw.write(str(time_predic_totaltraveltime_list) + '\n')
    fw.write(str(time_rtt_list) + '\n')
    fw.write(str(sum(time_predic_totaltraveltime_list) / len(time_predic_totaltraveltime_list)) + '\n')
    fw.write(str(sum(time_rtt_list) / len(time_rtt_list)) + '\n')

    fw.write(str('Diff predic & real total travel time (dist)') + '\n')
    fw.write(str(dist_predic_totaltraveltime_list) + '\n')
    fw.write(str(dist_rtt_list) + '\n')
    fw.write(str(sum(dist_predic_totaltraveltime_list) / len(dist_predic_totaltraveltime_list)) + '\n')
    fw.write(str(sum(dist_rtt_list) / len(dist_rtt_list)) + '\n')

    fw.write(str('Diff predic & real total travel time (cost)') + '\n')
    fw.write(str(cost_predic_totaltraveltime_list) + '\n')
    fw.write(str(cost_rtt_list) + '\n')
    fw.write(str(sum(cost_predic_totaltraveltime_list) / len(cost_predic_totaltraveltime_list)) + '\n')
    fw.write(str(sum(cost_rtt_list) / len(cost_rtt_list)) + '\n')

    fw.write(str('Diff predic & real total travel time (costtime)') + '\n')
    fw.write(str(costtime_predic_totaltraveltime_list) + '\n')
    fw.write(str(costtime_rtt_list) + '\n')
    fw.write(str(sum(costtime_predic_totaltraveltime_list) / len(costtime_predic_totaltraveltime_list)) + '\n')
    fw.write(str(sum(costtime_rtt_list) / len(costtime_rtt_list)) + '\n')

    fw.write(str('Charging Cost') + '\n')
    fw.write(str(time_charging_cost_list) + '\n')
    fw.write(str(dist_charging_cost_list) + '\n')
    fw.write(str(cost_charging_cost_list) + '\n')
    fw.write(str(costtime_charging_cost_list) + '\n')
    fw.write(str(sum(time_charging_cost_list) / len(time_charging_cost_list)) + '\n')
    fw.write(str(sum(dist_charging_cost_list) / len(dist_charging_cost_list)) + '\n')
    fw.write(str(sum(cost_charging_cost_list) / len(cost_charging_cost_list)) + '\n')
    fw.write(str(sum(costtime_charging_cost_list) / len(costtime_charging_cost_list)) + '\n')

    fw.write(str('Driving distance') + '\n')
    fw.write(str(time_driving_dist) + '\n')
    fw.write(str(dist_driving_dist) + '\n')
    fw.write(str(cost_driving_dist) + '\n')
    fw.write(str(costtime_driving_dist) + '\n')
    fw.write(str(sum(time_driving_dist) / len(time_driving_dist)) + '\n')
    fw.write(str(sum(dist_driving_dist) / len(dist_driving_dist)) + '\n')
    fw.write(str(sum(cost_driving_dist) / len(cost_driving_dist)) + '\n')
    fw.write(str(sum(costtime_driving_dist) / len(costtime_driving_dist)) + '\n')

    fw.write(str('Distance from S to EVCS') + '\n')
    fw.write(str(time_fdist_list) + '\n')
    fw.write(str(dist_fdist_list) + '\n')
    fw.write(str(cost_fdist_list) + '\n')
    fw.write(str(costtime_fdist_list) + '\n')
    fw.write(str(sum(time_fdist_list) / len(time_fdist_list)) + '\n')
    fw.write(str(sum(dist_fdist_list) / len(dist_fdist_list)) + '\n')
    fw.write(str(sum(cost_fdist_list) / len(cost_fdist_list)) + '\n')
    fw.write(str(sum(costtime_fdist_list) / len(costtime_fdist_list)) + '\n')

    fw.write(str('Driving time') + '\n')
    fw.write(str(time_rdt_list) + '\n')
    fw.write(str(dist_rdt_list) + '\n')
    fw.write(str(cost_rdt_list) + '\n')
    fw.write(str(costtime_rdt_list) + '\n')
    fw.write(str(sum(time_rdt_list) / len(time_rdt_list)) + '\n')
    fw.write(str(sum(dist_rdt_list) / len(dist_rdt_list)) + '\n')
    fw.write(str(sum(cost_rdt_list) / len(cost_rdt_list)) + '\n')
    fw.write(str(sum(costtime_rdt_list) / len(costtime_rdt_list)) + '\n')

    fw.write(str('Charging energy') + '\n')
    fw.write(str(time_rce_list) + '\n')
    fw.write(str(dist_rce_list) + '\n')
    fw.write(str(cost_rce_list) + '\n')
    fw.write(str(costtime_rce_list) + '\n')
    fw.write(str(sum(time_rce_list) / len(time_rce_list)) + '\n')
    fw.write(str(sum(dist_rce_list) / len(dist_rce_list)) + '\n')
    fw.write(str(sum(cost_rce_list) / len(cost_rce_list)) + '\n')
    fw.write(str(sum(costtime_rce_list) / len(costtime_rce_list)) + '\n')

    fw.write(str('Charging time') + '\n')
    fw.write(str(time_rct_list) + '\n')
    fw.write(str(dist_rct_list) + '\n')
    fw.write(str(cost_rct_list) + '\n')
    fw.write(str(costtime_rct_list) + '\n')
    fw.write(str(sum(time_rct_list) / len(time_rct_list)) + '\n')
    fw.write(str(sum(dist_rct_list) / len(dist_rct_list)) + '\n')
    fw.write(str(sum(cost_rct_list) / len(cost_rct_list)) + '\n')
    fw.write(str(sum(costtime_rct_list) / len(costtime_rct_list)) + '\n')

    fw.write(str('Waiting time') + '\n')
    fw.write(str(time_rwt_list) + '\n')
    fw.write(str(dist_rwt_list) + '\n')
    fw.write(str(cost_rwt_list) + '\n')
    fw.write(str(costtime_rwt_list) + '\n')
    fw.write(str(sum(time_rwt_list) / len(time_rwt_list)) + '\n')
    fw.write(str(sum(dist_rwt_list) / len(dist_rwt_list)) + '\n')
    fw.write(str(sum(cost_rwt_list) / len(cost_rwt_list)) + '\n')
    fw.write(str(sum(costtime_rwt_list) / len(costtime_rwt_list)) + '\n')

    fw.write(str('Total travel time') + '\n')
    fw.write(str(time_rtt_list) + '\n')
    fw.write(str(dist_rtt_list) + '\n')
    fw.write(str(cost_rtt_list) + '\n')
    fw.write(str(costtime_rtt_list) + '\n')
    fw.write(str(sum(time_rtt_list) / len(time_rtt_list)) + '\n')
    fw.write(str(sum(dist_rtt_list) / len(dist_rtt_list)) + '\n')
    fw.write(str(sum(cost_rtt_list) / len(cost_rtt_list)) + '\n')
    fw.write(str(sum(costtime_rtt_list) / len(costtime_rtt_list)) + '\n')

    fw.write(str('init_SOC') + '\n')
    fw.write(str(time_init_SOC_list) + '\n')
    fw.write(str(dist_init_SOC_list) + '\n')
    fw.write(str(cost_init_SOC_list) + '\n')
    fw.write(str(costtime_init_SOC_list) + '\n')
    fw.write(str(sum(time_init_SOC_list) / len(time_init_SOC_list)) + '\n')
    fw.write(str(sum(dist_init_SOC_list) / len(dist_init_SOC_list)) + '\n')
    fw.write(str(sum(costtime_init_SOC_list) / len(costtime_init_SOC_list)) + '\n')

    fw.write(str('Final SOC') + '\n')
    fw.write(str(time_final_SOC) + '\n')
    fw.write(str(dist_final_SOC) + '\n')
    fw.write(str(cost_final_SOC) + '\n')
    fw.write(str(costtime_final_SOC) + '\n')
    fw.write(str(sum(time_final_SOC) / len(time_final_SOC)) + '\n')
    fw.write(str(sum(dist_final_SOC) / len(dist_final_SOC)) + '\n')
    fw.write(str(sum(cost_final_SOC) / len(cost_final_SOC)) + '\n')
    fw.write(str(sum(costtime_final_SOC) / len(costtime_final_SOC)) + '\n')

    fw.close()

def sim_result_general_presentation(resultdir, numev, **results):
    keyname = ''
    for key in results.keys():
        keyname += '_'+ key
    basepath = os.getcwd()
    resultdir = resultdir+'/result{}'.format(keyname)
    print(os.path.join(basepath, resultdir))
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    plt.figure(figsize=(12, 6), dpi=300)

    keylist = list(results.keys())


    for i in range(numev):
        for key, EVlist in results.items():
            pev = EVlist[i]
            xx = []
            yy = []
            for nid in pev.path:
                x, y = graph.nodes_xy(nid)
                xx.append(x)
                yy.append(y)
            plt.plot(xx, yy, label=key)
            cs_x, cs_y = graph.nodes_xy(pev.cs.id)
            plt.plot(cs_x, cs_y, 'D', label=key+' EVCS')

        s_x, s_y = graph.nodes_xy(pev.source)
        plt.plot(s_x, s_y, 'p', label='Source')
        d_x, d_y = graph.nodes_xy(pev.destination)
        plt.plot(d_x, d_y, 'h', label='Destination')

        plt.legend()
        fig = plt.gcf()
        fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='#eeeeee', dpi=300)
        plt.clf()



    plt.title('Selected EVCS')
    plt.xlabel('EV index')
    plt.ylabel('EVCS ID')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cs.id)
        plt.plot(range(len(r1_list)), r1_list,'x', label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Charging Cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.chargingcost)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Driving distance')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.drivingdistance)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving distance.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Distance from S to EVCS')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.fdist)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Distance from S to EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Driving time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    for key, EVlist in results.items():
        numev = len(EVlist)
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.drivingtime)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Charging energy')
    plt.xlabel('EV ID')
    plt.ylabel('Energy(kWh)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.charingenergy)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Charging time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.chargingtime)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Charging time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Waiting time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.waitingtime)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Waiting time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Total travel time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(min)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.waitingtime + ev.chargingtime + ev.drivingtime)
        plt.plot(r1_list, label=key)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Total travel time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('EV SOC')
    plt.xlabel('EV ID')
    plt.ylabel('SOC(%)')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        plt.plot(r1_list, label=key+'init SOC')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.SOC)
        plt.plot(r1_list, label=key+'final SOC')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/ev SOC.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    #=======================================================================================

def sim_result_text(resultdir, **results):

    fw = open('{}/result.txt'.format(resultdir), 'w', encoding='UTF8')

    fw.write('Charging Cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.chargingcost)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.drivingdistance)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Distance from S to EVCS\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.fdist)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.drivingtime)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Charging energy\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.charingenergy)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.chargingtime)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Waiting time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.waitingtime)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('Total travel time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.waitingtime + ev.chargingtime + ev.drivingtime)
        fw.write(key + '\n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.write('EV SOC\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        fw.write(key + 'init \n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.SOC)
        fw.write(key + 'final \n')
        fw.write(str(sum(r1_list) / len(r1_list)) + '\t')
        fw.write(str(r1_list) + '\n')

    fw.close()

def sim_main_time_every_node_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================time_every sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))

        start = pev.source
        end = pev.destination
        pev.curr_location = start
        pev.path.append(start)

        while pev.curr_location != end:
            here = pev.curr_location
            paths_info = PriorityQueue()
            update_envir_timeweight(CS_list, graph, sim_time)
            evcango = pev.SOC * pev.maxBCAPA / pev.ECRate

            if pev.cs != None  and pev.cs.id == pev.curr_location:
                evcs=pev.cs
                print('CS')
                gen_evcs_random(evcs)
                pev.charingenergy = pev.maxBCAPA * pev.req_SOC - pev.SOC * pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy / (evcs.chargingpower * pev.chaging_effi)) * 60
                pev.waitingtime = evcs.waittime
                pev.charged = 1
                sim_time += pev.chargingtime
                sim_time += evcs.waittime

            if not pev.charged:
                cslist = []
                for cs in CS_list:
                    airdist = heuristic(graph.nodes_xy(pev.curr_location), graph.nodes_xy(cs.id))
                    if airdist < evcango:
                        cslist.append(cs)
                for cs in cslist:
                    evcs_id = cs.id
                    came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                    front_path = reconstruct_path(came_from, here, evcs_id)
                    front_path_distance = graph.get_path_distance(front_path)
                    if front_path_distance > evcango:
                        continue
                    came_from, cost_so_far = a_star_search(graph, evcs_id, end)
                    rear_path = reconstruct_path(came_from, evcs_id, end)
                    rear_path_distance = graph.get_path_distance(rear_path)
                    final_path = front_path + rear_path[1:]
                    dist = graph.get_path_distance(final_path)
                    d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
                    remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                    c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                    c_time = (c_energy / (cs.chargingpower * pev.chaging_effi)) * 60
                    w = d_time + cs.waittime + c_time
                    totaltraveltime = d_time + cs.waittime + c_time
                    paths_info.put((remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)

                remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
                pev.cs = evcs
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

            else:
                came_from, cost_so_far = a_star_search(graph, here, end)
                path = reconstruct_path(came_from, here, end)
                path_distance = graph.get_path_distance(path)
                if path_distance > evcango:
                    print('error')
                    input()
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

        evcs = pev.cs
        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1

    return EV_list

def sim_main_dist_every_node_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================dist_every sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))

        start = pev.source
        end = pev.destination
        pev.curr_location = start
        pev.path.append(start)

        while pev.curr_location != end:
            here = pev.curr_location
            paths_info = PriorityQueue()
            update_envir_distweight(CS_list, graph, sim_time)
            evcango = pev.SOC * pev.maxBCAPA / pev.ECRate

            if pev.cs != None  and pev.cs.id == pev.curr_location:
                evcs=pev.cs
                print('CS')
                gen_evcs_random(evcs)
                pev.charingenergy = pev.maxBCAPA * pev.req_SOC - pev.SOC * pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy / (evcs.chargingpower * pev.chaging_effi)) * 60
                pev.waitingtime = evcs.waittime
                pev.charged = 1
                sim_time += pev.chargingtime
                sim_time += evcs.waittime

            if not pev.charged:
                cslist = []
                for cs in CS_list:
                    airdist = heuristic(graph.nodes_xy(pev.curr_location), graph.nodes_xy(cs.id))
                    if airdist < evcango:
                        cslist.append(cs)
                for cs in cslist:
                    evcs_id = cs.id
                    came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                    front_path = reconstruct_path(came_from, here, evcs_id)
                    front_path_distance = graph.get_path_distance(front_path)
                    if front_path_distance > evcango:
                        continue
                    came_from, cost_so_far = a_star_search(graph, evcs_id, end)
                    rear_path = reconstruct_path(came_from, evcs_id, end)
                    rear_path_distance = graph.get_path_distance(rear_path)
                    final_path = front_path + rear_path[1:]
                    dist = graph.get_path_distance(final_path)
                    d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
                    remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                    c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                    c_time = (c_energy / (cs.chargingpower * pev.chaging_effi)) * 60
                    w = dist
                    totaltraveltime = d_time + cs.waittime + c_time
                    paths_info.put((remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)

                remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
                pev.cs = evcs
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

            else:
                came_from, cost_so_far = a_star_search(graph, here, end)
                path = reconstruct_path(came_from, here, end)
                path_distance = graph.get_path_distance(path)
                if path_distance > evcango:
                    print('error')
                    input()
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

        evcs = pev.cs
        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1

    return EV_list

def sim_main_cost_every_node_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================cost_everysim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))

        start = pev.source
        end = pev.destination
        pev.curr_location = start
        pev.path.append(start)

        while pev.curr_location != end:
            here = pev.curr_location
            paths_info = PriorityQueue()
            update_envir_costweight(CS_list,pev, graph, sim_time)
            evcango = pev.SOC * pev.maxBCAPA / pev.ECRate

            if pev.cs != None  and pev.cs.id == pev.curr_location:
                evcs=pev.cs
                print('CS')
                gen_evcs_random(evcs)
                pev.charingenergy = pev.maxBCAPA * pev.req_SOC - pev.SOC * pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy / (evcs.chargingpower * pev.chaging_effi)) * 60
                pev.waitingtime = evcs.waittime
                pev.charged = 1
                sim_time += pev.chargingtime
                sim_time += evcs.waittime

            if not pev.charged:
                cslist = []
                for cs in CS_list:
                    airdist = heuristic(graph.nodes_xy(pev.curr_location), graph.nodes_xy(cs.id))
                    if airdist < evcango:
                        cslist.append(cs)
                for cs in cslist:
                    evcs_id = cs.id
                    came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                    front_path = reconstruct_path(came_from, here, evcs_id)
                    front_path_distance = graph.get_path_distance(front_path)
                    if front_path_distance > evcango:
                        continue
                    came_from, cost_so_far = a_star_search(graph, evcs_id, end)
                    rear_path = reconstruct_path(came_from, evcs_id, end)
                    rear_path_distance = graph.get_path_distance(rear_path)
                    final_path = front_path + rear_path[1:]
                    cost_road = graph.get_path_weight(final_path) * 60

                    dist = graph.get_path_distance(final_path)
                    d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
                    remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                    c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                    c_time = (c_energy / (cs.chargingpower * pev.chaging_effi)) * 60
                    w = cost_road + c_energy*cs.price
                    totaltraveltime = d_time + cs.waittime + c_time
                    paths_info.put((remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)

                remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
                pev.cs = evcs
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

            else:
                came_from, cost_so_far = a_star_search(graph, here, end)
                path = reconstruct_path(came_from, here, end)
                path_distance = graph.get_path_distance(path)
                if path_distance > evcango:
                    print('error')
                    input()
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

        evcs = pev.cs
        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1

    return EV_list

def sim_main_cost_time_every_node_check(t_EV_list, t_CS_list, t_graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    EV_list = copy.deepcopy(t_EV_list)
    CS_list = copy.deepcopy(t_CS_list)
    graph = copy.deepcopy(t_graph)
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================cost_time_every sim: {}==================================".format(sim_n))
        print("ID {}    S:{}   D:{}  Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))

        start = pev.source
        end = pev.destination
        pev.curr_location = start
        pev.path.append(start)

        while pev.curr_location != end:
            here = pev.curr_location
            paths_info = PriorityQueue()
            update_envir_costtimeweight(CS_list, pev, graph, sim_time)
            evcango = pev.SOC * pev.maxBCAPA / pev.ECRate

            if pev.cs != None  and pev.cs.id == pev.curr_location:
                evcs=pev.cs
                print('CS')
                gen_evcs_random(evcs)
                pev.charingenergy = pev.maxBCAPA * pev.req_SOC - pev.SOC * pev.maxBCAPA
                pev.chargingcost = pev.charingenergy * evcs.price
                pev.SOC = pev.req_SOC
                pev.chargingtime = (pev.charingenergy / (evcs.chargingpower * pev.chaging_effi)) * 60
                pev.waitingtime = evcs.waittime
                pev.charged = 1
                sim_time += pev.chargingtime
                sim_time += evcs.waittime

            if not pev.charged:
                cslist = []
                for cs in CS_list:
                    airdist = heuristic(graph.nodes_xy(pev.curr_location), graph.nodes_xy(cs.id))
                    if airdist < evcango:
                        cslist.append(cs)
                for cs in cslist:
                    evcs_id = cs.id
                    came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                    front_path = reconstruct_path(came_from, here, evcs_id)
                    front_path_distance = graph.get_path_distance(front_path)
                    if front_path_distance > evcango:
                        continue
                    came_from, cost_so_far = a_star_search(graph, evcs_id, end)
                    rear_path = reconstruct_path(came_from, evcs_id, end)
                    rear_path_distance = graph.get_path_distance(rear_path)

                    final_path = front_path + rear_path[1:]
                    cost_road = graph.get_path_weight(final_path) * 60
                    dist = graph.get_path_distance(final_path)
                    d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5)) * 60
                    remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                    c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                    c_time = (c_energy / (cs.chargingpower * pev.chaging_effi)) * 60
                    w = cost_road + UNITtimecost*(cs.waittime+c_time) + c_energy*cs.price
                    totaltraveltime = d_time + cs.waittime + c_time
                    paths_info.put((remainenergy, final_path, front_path, front_path_distance, rear_path, rear_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)

                remainenergy, path, fpath, fpath_dist, rpath, rpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
                pev.cs = evcs
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

            else:
                came_from, cost_so_far = a_star_search(graph, here, end)
                path = reconstruct_path(came_from, here, end)
                path_distance = graph.get_path_distance(path)
                if path_distance > evcango:
                    print('error')
                    input()
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
                pev.curr_location = pev.next_location

        evcs = pev.cs
        print('CS alpha:', evcs.alpha)
        print('CS Waiting: ', evcs.waittime)
        print('Distance(km): ', pev.drivingdistance)
        print('Real Driving time(m): ', pev.drivingtime)
        print('Real Charging energy(kwh): ', pev.charingenergy)
        print('Real Charging time(m): ', pev.chargingtime)
        print('Real Waiting time(m): ', evcs.waittime)
        print('Real Total time(m): ', evcs.waittime + pev.chargingtime + pev.drivingtime)

        sim_n += 1

    return EV_list

if __name__ == "__main__":

    npev = 20
    now = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02}'.format(now.month, now.day, now.hour, now.minute)
    basepath = os.getcwd()

    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    np.random.seed(100)
    print('jeju')
    EV_list, CS_list, graph = gen_envir_jeju('data/20191001_5Min_modified.csv', npev)
    # for pev in EV_list:
    #     print("{} {} {} {} {} {}".format(pev.id, pev.source, pev.destination, pev.t_start, pev.init_SOC, pev.SOC))
    # for css in CS_list:
    #     print("{}   {}  {}  {}".format(css.id, css.alpha, css.price, css.waittime))

    time_v2_EV_list = sim_main_time_every_node_check(EV_list, CS_list, graph)
    dist_v2_EV_list = sim_main_dist_every_node_check(EV_list, CS_list, graph)
    cost_v2_EV_list = sim_main_cost_every_node_check(EV_list, CS_list, graph)
    costtime_v2_EV_list = sim_main_cost_time_every_node_check(EV_list, CS_list, graph)

    time_EV_list = sim_main_first_time_check(EV_list, CS_list, graph)
    dist_EV_list = sim_main_first_dist_check(EV_list, CS_list, graph)
    cost_EV_list = sim_main_first_cost_check(EV_list, CS_list, graph)
    costtime_EV_list = sim_main_first_cost_time_check(EV_list, CS_list, graph)

    sim_result_text(resultdir, one_time=time_EV_list, one_dist=dist_EV_list, one_cost=cost_EV_list, one_costtime=costtime_EV_list, every_time=time_v2_EV_list, every_dist=dist_v2_EV_list, every_cost=cost_v2_EV_list, every_costtime=costtime_v2_EV_list)

    sim_result_general_presentation(resultdir, npev, one_time=time_EV_list, every_time=time_v2_EV_list)
    sim_result_general_presentation(resultdir, npev, one_dist=dist_EV_list, every_dist=dist_v2_EV_list)
    sim_result_general_presentation(resultdir, npev, one_cost=cost_EV_list, every_cost=cost_v2_EV_list)
    sim_result_general_presentation(resultdir, npev, one_costtime=costtime_EV_list, every_costtime=costtime_v2_EV_list)
    sim_result_general_presentation(resultdir, npev, one_time=time_EV_list, one_dist=dist_EV_list, one_cost=cost_EV_list, one_costtime=costtime_EV_list)
    sim_result_general_presentation(resultdir, npev, every_time=time_v2_EV_list, every_dist=dist_v2_EV_list, every_cost=cost_v2_EV_list, every_costtime=costtime_v2_EV_list)
    #
    #







