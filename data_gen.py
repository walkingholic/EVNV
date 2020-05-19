import csv

def network_info(datapath):
    f = open('data/link_info.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_data = {}

    for line in rdr:
        if linenum == 0:
            a = line
        else:

            link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15]), 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
        linenum += 1
    print('total links', linenum-1)
    f.close()


    f = open('data/node_info.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2], 'lat': float(line[6]),
                                      'long': float(line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat', 'long'
        linenum += 1
    print('total nodes', linenum-1)
    f.close()

    f = open(datapath, 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_traffic = {}
    for line in rdr:
        linenum += 1
        if int(line[1]) > 4000000000 and int(line[1]) < 4090000000:
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


    return link_data, node_data, link_traffic

# link_data,link_pair, _, list = network_info()
#
# print(link_pair)
#
# for k in link_pair.keys():
#     print(link_pair.get(k))

