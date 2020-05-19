import csv
# import os
# import pandas as pd
from Graph import Graph_jeju



graph = Graph_jeju()
f = open('data/20200501_5Min_modified.csv', 'r', encoding='UTF8')
rdr = csv.reader(f)
a = 0
linenum = 0
link_traffic = {}
for line in rdr:
    linenum += 1
    if int(line[1]) > 4000000000:
        if int(line[1]) in link_traffic:
            link_traffic[int(line[1])].append(float(line[2]))
        else:
            link_traffic[int(line[1])] = [float(line[2])]
        # print(line)
print('l', linenum)
f.close()

# for link in link_traffic:
#     if len(link_traffic[link]) < 288:
#         print(link, len(link_traffic[link]), link_traffic[link])
#
# print(len(link_traffic))

for l in (graph.link_data):
    if l in link_traffic.keys():
        print(l, graph.link_data[l]['LENGTH'], len(link_traffic[l]), link_traffic[l])