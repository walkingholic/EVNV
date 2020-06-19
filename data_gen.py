import csv
import pprint as pp

def network_info(datapath):

    f = open('data/node_info.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[6]),
                                       'long': float(line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat', 'long'
        linenum += 1
    print('total nodes', linenum - 1)
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
    # print('l', linenum)
    f.close()


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
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])/1000, 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
        linenum += 1
    print('total links', linenum-1)
    f.close()

    f = open('data/chargingstation.csv', 'r')
    rdr = csv.reader(f)
    linenum = 0
    cs_info = {}
    for line in rdr:
        if linenum == 0:
            print(line)
        else:
            if line[7] == 'Y':
                mindist = 100000
                n_id = -1
                # print(linenum, line[7],line[17],line[16], line)
                x1 = float(line[17])
                y1 = float(line[16])

                for n in node_data.keys():
                    x2 = node_data[n]['long']
                    y2 = node_data[n]['lat']
                    diff = abs(x1 - x2) + abs(y1 - y2)
                    if mindist > diff:
                        mindist = diff
                        n_id = n
                # print(n_id, mindist )
                if n_id in cs_info.keys():
                    if diff < cs_info[n_id]['diff_node']:
                        cs_info[n_id] = {'CS_ID': n_id, 'CS_NAME': line[0], 'lat': node_data[n_id]['lat'], 'long': node_data[n_id]['long'],'real_lat': float(line[16]),
                                         'real_long': float(line[17]), 'diff_node': mindist}
                else:
                    cs_info[n_id] = {'CS_ID': n_id, 'CS_NAME': line[0], 'lat': node_data[n_id]['lat'],
                                    'long': node_data[n_id]['long'], 'real_lat': float(line[16]),
                                    'real_long': float(line[17]), 'diff_node': mindist}


        linenum+=1
    f.close()


    return link_data, node_data, link_traffic, cs_info


# network_info('data/20191001_5Min_modified.csv')