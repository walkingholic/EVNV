import csv
import datetime
import os
import numpy as np
from Graph import Graph
import matplotlib.pyplot as plt

# now = datetime.datetime.now()
# basepath = os.getcwd()
# resultdir = '{0:02}-{1:02} {2:02}-{3:02} result'.format(now.month, now.day, now.hour, now.minute)
#
# # f = open('{}/result.csv'.format(resultdir), 'w', encoding='UTF8')
# fw = open('result.txt', 'w')
# # nlist = np.random.random_integers(0, 10, 50)
# nlist = 12.5
# fw.write(str(nlist)+'\n')
#
# fw.close()
#

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

def test(**dic):
    keyname = ''
    for key in dic.keys():
        keyname += key + '_'
    a=123
    print('{0}{}'.format(a, keyname))
    print(len(dic), dic.get('EV_list_1'))

    print(len(dic), dic.popitem())
    print(len(dic), dic.popitem())

    # key, list = itm
    # print(key, list )


    # plt.title('Charging Cost')
    # plt.xlabel('EV ID')
    # plt.ylabel('Cost($)')
    # for key, EVlist in dic.items():
    #     print(key, EVlist)
    #     r1_list = []
    #     for ev in EVlist:
    #         r1_list.append(ev.SOC)
    #     plt.plot(r1_list, label=key+'evev')
    # plt.legend()
    # plt.show()

def make_simple_data_gen():
    f = open('data/node_info_100evs.csv', 'w', encoding='UTF8', newline='')
    fw = csv.writer(f)

    graph = Graph()

    fw.writerow(['NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'TURN_P', 'REMARK', 'x', 'y'])
    for i in range(graph.num_node):
        print(i, graph.nodes_xy[i])
        x, y = graph.nodes_xy[i]
        fw.writerow([i, 0, 0, 0, 0, float(x), float(y)])
    f.close()

    f = open('data/link_info_100evs.csv', 'w', encoding='UTF8', newline='')
    fw = csv.writer(f)
    fw.writerow(
        ['LINK_ID', 'F_NODE', 'T_NODE', 'LANES_', 'ROAD_RANK_', 'ROAD_TYPE_', 'ROAD_NO_', 'ROAD_NAME_', 'ROAD_USE_',
         'MULTILINK', 'CONNECT_', 'MAX_SPD_', 'REST_VEH_', 'REST_W_', 'REST_H_', 'LENGTH', 'REMARK_'])

    linkid = 0
    for i in range(graph.num_node):
        edges = graph.dict_edges[i]

        for j, value in edges.items():
            print(linkid, i, j, value)
            if edges[j]['road type'] == 1:
                fw.writerow([linkid, i, j, 'LANES_', 'ROAD_RANK_', 1, 'ROAD_NO_', 'ROAD_NAME_',
                             'ROAD_USE_', 'MULTILINK', 'CONNECT_', '80', 'REST_VEH_', 'REST_W_', 'REST_H_',
                             edges[j]['dist'] * 1000,
                             'REMARK_'])
            else:
                fw.writerow([linkid, i, j, 'LANES_', 'ROAD_RANK_', 2, 'ROAD_NO_', 'ROAD_NAME_',
                             'ROAD_USE_', 'MULTILINK', 'CONNECT_', '60', 'REST_VEH_', 'REST_W_', 'REST_H_',
                             edges[j]['dist'] * 1000,
                             'REMARK_'])
            linkid += 1


if __name__ == "__main__":

    print('test')
    linetype = ['-','--', ':','-.']
    for i, v in enumerate(a):
        # print(i, v)
        print(a[i])

    # make_simple_data_gen()


    # EV_list_1 = []
    # num_evs = 10
    # for e in range(num_evs):
    #     t_start = np.random.uniform(0, 1200)
    #     soc = np.random.uniform(0.3, 0.5)
    #     while soc <= 0.0 or soc > 1.0:
    #         soc = np.random.uniform(0.3, 0.5)
    #     source = np.random.random_integers(0, 20)
    #     destination = np.random.random_integers(0, 20)
    #
    #     ev = EV(e, t_start, soc, source, destination)
    #     EV_list_1.append(ev)
    #
    # EV_list_2 = []
    # num_evs = 10
    # for e in range(num_evs):
    #     t_start = np.random.uniform(0, 1200)
    #     soc = np.random.uniform(0.3, 0.5)
    #     while soc <= 0.0 or soc > 1.0:
    #         soc = np.random.uniform(0.3, 0.5)
    #     source = np.random.random_integers(0, 20)
    #     destination = np.random.random_integers(0, 20)
    #
    #     ev = EV(e, t_start, soc, source, destination)
    #     EV_list_2.append(ev)
    #
    # EV_list_3 = []
    # num_evs = 10
    # for e in range(num_evs):
    #     t_start = np.random.uniform(0, 1200)
    #     soc = np.random.uniform(0.3, 0.5)
    #     while soc <= 0.0 or soc > 1.0:
    #         soc = np.random.uniform(0.3, 0.5)
    #     source = np.random.random_integers(0, 20)
    #     destination = np.random.random_integers(0, 20)
    #
    #     ev = EV(e, t_start, soc, source, destination)
    #     EV_list_3.append(ev)
    #
    # test(EV_list_1=EV_list_1, EV_list_2=EV_list_2, EV_list_3=EV_list_3)
