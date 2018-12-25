"""builds graph from genes"""
import tensorflow as tf
import numpy as np
import math
from Constants import OUTPUT0, OUTPUT1, INPUT0, INPUT1, INPUT2, INPUT3,INPUT4, INPUT5, INPUT6, INPUT7,INPUT8, INPUT9, INPUT10, INPUT11,INPUT12, INPUT13, INPUT14, INPUT15,INPUT16, INPUT17, INPUT18, INPUT19,INPUT20, INPUT21, INPUT22, INPUT23,INPUT24, INPUT25, INPUT26, INPUT27,INPUT28, INPUT29, train_epoch, train_writer_epoch,BENCH_ACCURANCY


BENCH_ACCURANCY = BENCH_ACCURANCY

#根据连接的结点，计算to结点的最终输出值
def add_node(inputs,name="stdname"):
    '''
    tf.stack( values,axis=0,name=’stack’) 
    功能：将两个数组按照指定的方向进行叠加，生成一个新的数组。 
    参数axis：当其取0时表示按照x轴方向进行叠加，当其取1时表示按照y轴进行叠加
    
    tf.squeeze
    功能：从tensor中删除所有大小是1的维度
    例子：[[1 2 3]
           [2 3 4]]squeeze=>［1 2 3]
                             [2 3 4］
    '''

    with tf.name_scope(name):
        init_vals = tf.truncated_normal([len(inputs), 1], stddev=1. / math.sqrt(2))
        w = tf.Variable(init_vals)
        b = tf.Variable(tf.zeros([1]))

        if len(inputs) > 1:
            in_tensor = tf.transpose(tf.squeeze(tf.stack(inputs)))
            output = tf.nn.relu(tf.matmul(in_tensor, w) + b, name=name)
            return output

        #一维无需矩阵乘，无需转置
        else:
            in_tensor = tf.squeeze(tf.stack(inputs))
            output = tf.transpose(tf.nn.relu(tf.multiply(in_tensor, w) + b, name=name))
            return output


def build_and_test(connections, genotype, x, y, x_test, y_test, run_id="1",Iteration=0):

    global BENCH_ACCURANCY 
    
    with tf.name_scope("input"):
        x0 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x0")
        x1 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x1")
        x2 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x2")
        x3 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x3")
        x4 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x4")
        x5 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x5")
        x6 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x6")
        x7 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x7")
        x8 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x8")
        x9 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x9")
        x10 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x10")
        x11 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x11")
        x12 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x12")
        x13 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x13")
        x14 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x14")
        x15 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x15")
        x16 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x16")
        x17 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x17")
        x18 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x18")
        x19 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x19")
        x20 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x20")
        x21 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x21")
        x22 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x22")
        x23 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x23")
        x24 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x24")
        x25 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x25")
        x26 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x26")
        x27 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x27")
        x28 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x28")
        x29 = tf.placeholder(shape=[None,1], dtype=tf.float32, name="x29")
    with tf.name_scope("ground_truth"):
        y_ = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="y_")

    # connections contains only (innovation_num, from, to)
    # genotype contains only {innovation_num: True/False)

    # need a list containing [node, inputs]
    # for all the same to's from connections collect from's (where genotype exists and says enabled)
    # connection can only exist from lower number to higher in the network

    # filter out disabled and non existent genes in this phenotype
    genotype_keys = sorted(genotype.keys())

    # filter connections
    exisiting_connections = []
    for i in range(0,len(genotype_keys)):
        if genotype[genotype_keys[i]]:
            exisiting_connections.append(connections[genotype_keys[i]])

    # collect nodes and connections: sort by to field from connections
    
    # 根据to的结点编号对connections集合中的边进行排序
    # print("connections_sorted before:",exisiting_connections)
    connections_sorted = sorted(exisiting_connections, key=lambda connections: connections[2])
    # print("connections_sorted after:",connections_sorted)

    # merge the same nodes
    connections_merged = [[connections_sorted[0][2],[connections_sorted[0][1]]]]
    # print(connections_merged)

    #找出与每一个to结点有连接的所有from结点，to结点从小到大进行排序，
    #只需要从小到大遍历一遍to结点即可找到所有与其相连的所有from结点
    for i in range(1,len(connections_sorted)):
        # same as last node
        if connections_sorted[i][2] == connections_merged[-1][0]:
            connections_merged[-1][1].append(connections_sorted[i][1])
        else:
            connections_merged.append([connections_sorted[i][2],[connections_sorted[i][1]]])
        # print(i,"  : ",connections_merged)

    tf_nodes_dict = {INPUT0: x0, INPUT1: x1,INPUT2: x2,INPUT3: x3,
                     INPUT4: x4, INPUT5: x5,INPUT6: x6,INPUT7: x7,
                     INPUT8: x8, INPUT9: x9,INPUT10: x10,INPUT11: x11,
                     INPUT12: x12, INPUT13: x13,INPUT14: x14,INPUT15: x15,
                     INPUT16: x16, INPUT17: x17,INPUT18: x18,INPUT19: x19,
                     INPUT20: x20, INPUT21: x21,INPUT22: x22,INPUT23: x23,
                     INPUT24: x24, INPUT25: x25,INPUT26: x26,INPUT27: x27,
                     INPUT28: x28, INPUT29: x29}#声明结点字典，初始值为两个输入结点
                                            #{0:x0,1:x1}

    # print(tf_nodes_dict)

    for cn in connections_merged:
        node_cons = cn[1]
        node_id = cn[0]#以to结点的id作为字典结构的索引
        tf_input_nodes = [tf_nodes_dict[node_key] for node_key in node_cons]
        node_name = str(node_id) + "_"
        #结点的名字为to_froms的组合
        for na in node_cons:
            node_name += "_" + str(na)
        # print("tf_input_nodes:",tf_input_nodes)
        # print("node_name:",node_name)
        # print("tf_nodes_dict",tf_nodes_dict[INPUT0])
        # print("type(tf_input_nodes)",type(tf_input_nodes))

        tf_nodes_dict[cn[0]] = add_node(tf_input_nodes, name=node_name)

    num_nodes = tf.constant(len(tf_nodes_dict.keys()))
    tf.summary.scalar("num_nodes", num_nodes)

    with tf.name_scope("softmax_output"):
        # requery output nodes to add to optimization
        # 获得计算图中的输出节点
        output_0 = tf_nodes_dict[OUTPUT0]
        output_1 = tf_nodes_dict[OUTPUT1]

        output_final_pre = tf.transpose(tf.squeeze(tf.stack([output_0,output_1])))

        #在最后再加一层softmax层
        W_o1 = tf.Variable(tf.truncated_normal([2, 2], stddev=1. / math.sqrt(2)))
        b_o1 = tf.Variable(tf.zeros([2]))
        #得到每一类的分类概率
        output_final = tf.nn.softmax(tf.matmul(output_final_pre,W_o1) + b_o1, name="output_softmax")

    with tf.name_scope("loss"):
        #计算分类的交叉熵损失函数
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_final,labels=y_)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss",loss)

    with tf.name_scope("optimizer"):
        #采用梯度下降法进行优化
        opt = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.name_scope("accuracy"):
        #计算最终的分类准确率
        correct_prediction = tf.equal(tf.argmax(output_final, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy = tf.reduce_mean(tf.abs(output_final - y_))
        tf.summary.scalar("accuracy", accuracy)

    init = tf.initialize_all_variables()
    sess = tf.Session()

    #定义tensorboard的相关写操作
    tf.summary.merge_all()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train/' + run_id,
                                          sess.graph)

    saver = tf.train.Saver(max_to_keep=4)

    sess.run(init)

    '''
    np.expand_dims:
    Insert a new axis, corresponding to a given position in the array shape.
    '''

    for i in range(train_epoch*len(tf_nodes_dict.keys())):
        _, loss_val, summary = sess.run([opt, loss, merged], feed_dict={x0: np.expand_dims(x[:, 0], 1),x1: np.expand_dims(x[:, 1], 1),
                                            x2: np.expand_dims(x[:, 2], 1),x3: np.expand_dims(x[:, 3], 1),x4: np.expand_dims(x[:, 4], 1),
                                            x5: np.expand_dims(x[:, 5], 1),x6: np.expand_dims(x[:, 6], 1),x7: np.expand_dims(x[:, 7], 1),
                                            x8: np.expand_dims(x[:, 8], 1),x9: np.expand_dims(x[:, 9], 1),x10: np.expand_dims(x[:, 10], 1),
                                            x11: np.expand_dims(x[:, 11], 1),x12: np.expand_dims(x[:, 12], 1),x13: np.expand_dims(x[:, 13], 1),
                                            x14: np.expand_dims(x[:, 14], 1),x15: np.expand_dims(x[:, 15], 1),x16: np.expand_dims(x[:, 16], 1),
                                            x17: np.expand_dims(x[:, 17], 1),x18: np.expand_dims(x[:, 18], 1),x19: np.expand_dims(x[:, 19], 1),
                                            x20: np.expand_dims(x[:, 20], 1),x21: np.expand_dims(x[:, 21], 1),x22: np.expand_dims(x[:, 22], 1),
                                            x23: np.expand_dims(x[:, 23], 1),x24: np.expand_dims(x[:, 24], 1),x25: np.expand_dims(x[:, 25], 1),
                                            x26: np.expand_dims(x[:, 26], 1),x27: np.expand_dims(x[:, 27], 1),x28: np.expand_dims(x[:, 28], 1),
                                            x29: np.expand_dims(x[:, 29], 1),y_: y})
        if i % train_writer_epoch == 0:
            train_writer.add_summary(summary, i)

    # acc_test = sess.run([accuracy], feed_dict={x0: np.expand_dims(x_test[:, 0], 1), x1: np.expand_dims(x_test[:, 1], 1),
    #                                 x2: np.expand_dims(x_test[:, 2], 1), x3: np.expand_dims(x_test[:, 3], 1), y_: y_test})

    acc_test = sess.run([accuracy], feed_dict={x0: np.expand_dims(x_test[:, 0], 1),x1: np.expand_dims(x_test[:, 1], 1),
                                            x2: np.expand_dims(x_test[:, 2], 1),x3: np.expand_dims(x_test[:, 3], 1),x4: np.expand_dims(x_test[:, 4], 1),
                                            x5: np.expand_dims(x_test[:, 5], 1),x6: np.expand_dims(x_test[:, 6], 1),x7: np.expand_dims(x_test[:, 7], 1),
                                            x8: np.expand_dims(x_test[:, 8], 1),x9: np.expand_dims(x_test[:, 9], 1),x10: np.expand_dims(x_test[:, 10], 1),
                                            x11: np.expand_dims(x_test[:, 11], 1),x12: np.expand_dims(x_test[:, 12], 1),x13: np.expand_dims(x_test[:, 13], 1),
                                            x14: np.expand_dims(x_test[:, 14], 1),x15: np.expand_dims(x_test[:, 15], 1),x16: np.expand_dims(x_test[:, 16], 1),
                                            x17: np.expand_dims(x_test[:, 17], 1),x18: np.expand_dims(x_test[:, 18], 1),x19: np.expand_dims(x_test[:, 19], 1),
                                            x20: np.expand_dims(x_test[:, 20], 1),x21: np.expand_dims(x_test[:, 21], 1),x22: np.expand_dims(x_test[:, 22], 1),
                                            x23: np.expand_dims(x_test[:, 23], 1),x24: np.expand_dims(x_test[:, 24], 1),x25: np.expand_dims(x_test[:, 25], 1),
                                            x26: np.expand_dims(x_test[:, 26], 1),x27: np.expand_dims(x_test[:, 27], 1),x28: np.expand_dims(x_test[:, 28], 1),
                                            x29: np.expand_dims(x_test[:, 29], 1),y_: y_test})
    if(acc_test[0]>BENCH_ACCURANCY):
        BENCH_ACCURANCY = acc_test[0]
        saver.save(sess, "model/best-model-{}-generation-{}".format(acc_test[0],Iteration))

    print("="*20,acc_test)
    sess.close()
    tf.reset_default_graph()

    return acc_test[0]