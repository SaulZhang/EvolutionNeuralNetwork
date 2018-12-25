from Constants import OUTPUT0, OUTPUT1, INPUT0, INPUT1, INPUT2, INPUT3,INPUT4, INPUT5, INPUT6, INPUT7,INPUT8, INPUT9, INPUT10, INPUT11,INPUT12, INPUT13, INPUT14, INPUT15,INPUT16, INPUT17, INPUT18, INPUT19,INPUT20, INPUT21, INPUT22, INPUT23,INPUT24, INPUT25, INPUT26, INPUT27,INPUT28, INPUT29,MUTATE_ADD_NODE_RATE,MUTATE_ADD_CONNECTION_RATE,CROSSOVER_RATE,POPULATION_SIZE,INITIAL_POPULATION_SIZE,GENERATION
from tensorflow_utils import build_and_test
import numpy as np
import sys

Iteration = GENERATION
INPUT_LIST = [INPUT0, INPUT1, INPUT2, INPUT3,INPUT4, INPUT5, INPUT6, INPUT7,INPUT8, INPUT9, INPUT10, INPUT11,INPUT12, INPUT13, INPUT14, INPUT15,INPUT16, INPUT17, INPUT18, INPUT19,INPUT20, INPUT21, INPUT22, INPUT23,INPUT24, INPUT25, INPUT26, INPUT27,INPUT28, INPUT29]
OUTPUT_LIST = [OUTPUT0, OUTPUT1]
'''
增加一条连接
'''
def add_connection(connections, genotype):

    #获得父代个体基因型中当前激活的连接的编号
    enabled_innovations = [k for k in genotype.keys() if genotype[k]]
    # print("enabled_innovations:",enabled_innovations)

    #获得父代个体基因型中当前激活的连接的结点
    enabled_connections = [connections[cns] for cns in enabled_innovations]
    # print("enabled_innovations:",enabled_connections)

    # get reachable nodes
    # 获取当前可达的结点，包括连接出边以及连接入边的结点
    froms = set([fr[1] for fr in enabled_connections])#集合中不能存在重复的元素
    tos = set([to[2] for to in enabled_connections])

    nodes = sorted(list(froms.union(tos)))#取并集，然后从小到大进行排列

    # print("froms",froms)
    # print("tos",tos)
    # print("nodes",nodes)

    # select random two:
    r1 = np.random.randint(0,len(nodes))
    r2 = np.random.randint(0,len(nodes) - 1)
    
    #确保r1和r2不是重复的结点
    if r2 >= r1:
        r2 += 1

    r1 = nodes[r1]
    r2 = nodes[r2]
    from_node = r2 if r2 < r1 else r1
    to_node = r2 if r2 > r1 else r1

    assert(from_node < to_node)

    # prevent connections from input to input nodes and output to output nodes.
    # todo change this
    
    # 确保不存在输入连接输入，输出连接输出的边
    # if (from_node == INPUT0 and to_node == INPUT1 or from_node == OUTPUT0 and to_node == OUTPUT1 or 
    #         from_node == INPUT1 and to_node == INPUT2 or from_node == INPUT2 and to_node == INPUT3 or
    #         from_node == OUTPUT1 and to_node == OUTPUT2):
    #     return add_connection(connections, genotype)
    #     
    if ((from_node in INPUT_LIST and to_node in INPUT_LIST) or (from_node in OUTPUT_LIST and to_node in OUTPUT_LIST)):
        return add_connection(connections, genotype)#递归重新创建

    #判断该边是否已经存在
    # check if connection already there
    if not any(from_node == c[1] and to_node == c[2] for c in connections):
        connections.append((len(connections), from_node, to_node))

        #激活新加入的边的状态
        genotype[len(connections) - 1 ] = True

    assert(len(genotype.keys()) <= len(connections))
    return connections, genotype


'''
增加一个结点
'''
def add_node(connections, genotype, debug=False):
    # select random connection that is enabled
    # 获得父代个体基因型中当前激活的连接的编号
    enabled_innovations = [k for k in genotype.keys() if genotype[k]]

    # get random connection:
    r = np.random.randint(0,len(enabled_innovations))

    #获取一条激活的边的编号
    connections_innovation_index = enabled_innovations[r]

    #获取改变
    connection_to_split = connections[connections_innovation_index]

    #改变的起始结点
    from_node = connection_to_split[1]

    #改变的终止结点
    to_node = connection_to_split[2]

    #生成新结点的编号
    new_node = (to_node - from_node) / 2 + from_node

    if debug:
        print("from:", from_node)
        print("to:", to_node)
        print("new:", new_node)
    # todo: what to do if node id already exist? -> just leave it be.

    # add two new connection items: from_node -> new_node; new_node -> to_node
    # check if already existing beforehand.
    # todo: there should be a smarter way to do this than just give up.
    if not from_node < new_node:
        return connections, genotype

    if not new_node < to_node:
        return connections, genotype

    assert(from_node < new_node)
    assert(new_node < to_node)

    # check from to
    # 判断第一条边
    # 如果不存在重复的边，则加入与新的结点连接的新边
    if not any(from_node == c[1] and new_node == c[2] for c in connections):
        #在原来的基础之上，累加新边的id值
        id = len(connections)
        connections.append((id, from_node, new_node))
        #激活该边
        genotype[id] = True
    #如果新生成的边和原有的边发生了重复
    else:
        #记录重复边的id号
        ind = [c[0] for c in connections if c[1] == from_node and c[2] == new_node]
        #激活该边
        genotype[ind[0]] = True

    # 判断第一条边
    if not any(new_node == c[1] and to_node == c[2] for c in connections):
        id = len(connections)
        connections.append((id, new_node, to_node))
        genotype[id] = True
    else:
        ind = [c[0] for c in connections if new_node == c[1] and to_node == c[2]]
        genotype[ind[0]] = True

    # add new node
    # disable old connection where we now inserted a new node
    # 比如原来有1->3，现在在中间插入结点2，形成边1->2和2->3，那么相应的应该把边1->3废弃掉
    genotype[connections_innovation_index] = False

    assert (len(genotype.keys()) <= len(connections))

    return connections, genotype


'''
不同的基因型之间进行交叉
'''
def crossover(connections, genotype0, performance0 , genotype1, performance1):
    # 1. matching genes are inherited at random (everything is made up and the weights don't matter here)
    # 1. 如果上下均匹配上的话就随机的挑选一条边
    # 2. disjoint and excess from the more fit parent
    # 2. 如果存在不匹配或正剩余的话就是否选取改边取决于适应度更高的个体
    # 3. preset chance to disable gene if its disabled in either parent
    # 3. 如果父代双方有一方中改边的状态为disable，那么offspring中改边的状态就为disable

    # new genes should be always in the end
    # 对两个基因型的边序号进行排序，得到两个对应的边索引列表
    k_0 = sorted(genotype0.keys())
    k_1 = sorted(genotype1.keys())

    # inherit disjoint from more fit parent
    offspring_genotype = {}
    if performance0 > performance1 and len(k_0) > len(k_1):
        # 0 is better and has more genes
        for l in connections:
            innovation_num = l[0]
            if innovation_num in k_0:
                offspring_genotype[innovation_num] = genotype0[innovation_num]
            elif innovation_num in k_1:
                offspring_genotype[innovation_num] = genotype1[innovation_num]

    elif performance1 > performance0 and len(k_1) > len(k_0):
        # 1 is better and has more genes
        for l in connections:
            innovation_num = l[0]
            if innovation_num in k_1:
                offspring_genotype[innovation_num] = genotype1[innovation_num]
            elif innovation_num in k_0:
                offspring_genotype[innovation_num] = genotype0[innovation_num]

    elif len(k_1) < len(k_0):
        for k in k_1:
            offspring_genotype[k] = genotype1[k]

    elif len(k_0) <= len(k_1):
        for k in k_0:
            offspring_genotype[k] = genotype0[k]

    return offspring_genotype


'''
评估每一组基因型的适应度
'''
def eval_fitness(connections, genotype, x, y, x_test, y_test, run_id="1",Iteration=0):
    perf_train = build_and_test(connections, genotype, x, y, x_test, y_test, run_id=run_id,Iteration=Iteration)
    return perf_train


'''
启动神经元的进化
'''
def start_neuroevolution(x, y, x_test, y_test):
    """starts neuron evolution on binary dataset"""

    #初始化连接的集合，由于初始只有输入和输出层（每层两个结点），因此存在4条连接
    # connections = [(0, INPUT0, OUTPUT0), (1, INPUT0, OUTPUT1), (2, INPUT0, OUTPUT2), 
    #                (3, INPUT1, OUTPUT0), (4, INPUT1, OUTPUT1), (5, INPUT1, OUTPUT2),
    #                (6, INPUT2, OUTPUT0), (7, INPUT2, OUTPUT1), (8, INPUT2, OUTPUT2),
    #                (9, INPUT3, OUTPUT0), (10, INPUT3, OUTPUT1), (11, INPUT3, OUTPUT2)]

    connections = []
    cnt = 0
    for INPUT in INPUT_LIST:
        for OUTPUT in OUTPUT_LIST:
            connections.append((cnt,INPUT,OUTPUT))
            cnt+=1
    genodict = {}

    for i in range(len(connections)):
        genodict[i] = True


    #初始化种群的基因组，一开始有5个基因型
    # genotypes = [{0: True, 1: True, 2: True, 3: True, 4:True,5: True, 6: True, 7: True, 8: True, 9:True,10: True, 11: True} for no in range(INITIAL_POPULATION_SIZE)]
    
    genotypes = [genodict for no in range(INITIAL_POPULATION_SIZE)]
    
    #一共进行Iteration代的迭代
    for its in range(Iteration):
        print ("iteration", its)

        fitnesses = []
        # test networks，测试每一个基因型所对应的网络结构的表现，将所有模型的表现存储在适应度list当中
        for i in range(0,len(genotypes)):
            fitnesses.append(eval_fitness(connections, genotypes[i], x, y, x_test, y_test, run_id=str(its) + "/" + str(i),Iteration=its))

        # get indices of sorted list
        # 将每一个模型的适应度按照降序进行排序，得到按适应度降序的索引列表
        fitnesses_sorted_indices = [i[0] for i in reversed(sorted(enumerate(fitnesses), key=lambda x: x[1]))]

        # print("fitnesses_sorted_indices:",fitnesses_sorted_indices)

        print("connections:\n")

        #打印当前所有的边的集合
        print (connections)

        #打印按照降序排序的每一个基因型的accuracy以及建立连接的边
        for ra in range(0,len(fitnesses_sorted_indices)):
            print(fitnesses[fitnesses_sorted_indices[ra]], genotypes[fitnesses_sorted_indices[ra]])

        # run evolutions
        # todo: fiddle with parameters, include size of network in fitness?
        
        #存储当前最好的5个基因型的列表
        new_gen = []
        # copy five best survivors already
        m = 5
        if m > len(fitnesses):
            m = len(fitnesses)

        for i in range(0,m):
            print ("adding:", fitnesses[fitnesses_sorted_indices[i]], genotypes[fitnesses_sorted_indices[i]])
            new_gen.append(genotypes[fitnesses_sorted_indices[i]])

        for i in range(0,len(fitnesses_sorted_indices)):
            fi = fitnesses_sorted_indices[i]
            p = np.random.uniform()
            # select the best for mutation and breeding, kill of worst.
            
            #选择表现最好的个体来进行变异以及基因重组
            # 0.2的概率会添加一条边
            if p <= MUTATE_ADD_CONNECTION_RATE:
                # mutate
                # 更新边集合以及产生一个新的变异个体
                connections, gen = add_connection(connections, genotypes[fi])
                new_gen.append(gen)
            p = np.random.uniform()

            #0.5的概率会添加一个结点
            if p <= MUTATE_ADD_NODE_RATE:
                #mutate
                # 更新边集合以及产生一个新的变异个体
                connections, gen = add_node(connections, genotypes[fi])
                new_gen.append(gen)

            p = np.random.uniform()

            #0.1的概率会对两个模型进行基因交叉重组
            if p <= CROSSOVER_RATE:
                # select random for breeding
                r = np.random.randint(0,len(fitnesses))
                r2 = np.random.randint(0,len(fitnesses) - 1)
                if r2 >= r:
                    r2 +=1
                gen = crossover(connections, genotypes[r2], fitnesses[r2], genotypes[r], fitnesses[r])
                new_gen.append(gen)
                new_gen.append(genotypes[i])#基因重组完之后，仍需保留原来的父母双反中的一方
            
            # 种群的大小为10
            # stop if we have 5 candidates
            # 当new_gen中已经有超过10个基因组(5个最佳的+5个候选的)的时候就跳出select过程
            if len(new_gen) > POPULATION_SIZE:
                break

        genotypes = new_gen







