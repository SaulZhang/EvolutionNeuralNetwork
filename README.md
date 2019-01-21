<html>

<head>
<meta http-equiv=Content-Type content="text/html; charset=gb2312">
<meta name=Generator content="Microsoft Word 15 (filtered)">
<style>
<!--
 /* Font Definitions */
 @font-face
	{font-family:"Cambria Math";
	panose-1:2 4 5 3 5 4 6 3 2 4;}
@font-face
	{font-family:等线;
	panose-1:2 1 6 0 3 1 1 1 1 1;}
@font-face
	{font-family:"\@等线";
	panose-1:2 1 6 0 3 1 1 1 1 1;}
 /* Style Definitions */
 p.MsoNormal, li.MsoNormal, div.MsoNormal
	{margin:0cm;
	margin-bottom:.0001pt;
	text-align:justify;
	text-justify:inter-ideograph;
	font-size:10.5pt;
	font-family:等线;}
p.MsoHeader, li.MsoHeader, div.MsoHeader
	{mso-style-link:"页眉 字符";
	margin:0cm;
	margin-bottom:.0001pt;
	text-align:center;
	layout-grid-mode:char;
	border:none;
	padding:0cm;
	font-size:9.0pt;
	font-family:等线;}
p.MsoFooter, li.MsoFooter, div.MsoFooter
	{mso-style-link:"页脚 字符";
	margin:0cm;
	margin-bottom:.0001pt;
	layout-grid-mode:char;
	font-size:9.0pt;
	font-family:等线;}
a:link, span.MsoHyperlink
	{color:#0563C1;
	text-decoration:underline;}
a:visited, span.MsoHyperlinkFollowed
	{color:#954F72;
	text-decoration:underline;}
span.a
	{mso-style-name:"页眉 字符";
	mso-style-link:页眉;}
span.a0
	{mso-style-name:"页脚 字符";
	mso-style-link:页脚;}
span.xuhao
	{mso-style-name:xuhao;}
span.neirong
	{mso-style-name:neirong;}
span.heading
	{mso-style-name:heading;}
.MsoChpDefault
	{font-family:等线;}
 /* Page Definitions */
 @page WordSection1
	{size:595.3pt 841.9pt;
	margin:72.0pt 90.0pt 72.0pt 90.0pt;
	layout-grid:15.6pt;}
div.WordSection1
	{page:WordSection1;}
-->
</style>

</head>

<body lang=ZH-CN link="#0563C1" vlink="#954F72" style='text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<p class=MsoNormal style='text-indent:44.0pt'><b><span style='font-size:22.0pt'><center><h1><b>基于遗传算法的神经网络结构改进</center></h1></b></span></b></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><b><span
lang=EN-US style='font-size:22.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
style='font-size:12.0pt'><center>author：SaulZhang&nbsp&nbsp&nbsp&nbspSchool：NWPU </center><span lang=EN-US>&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>一、摘要</span></b></p>

<p class=MsoNormal style='text-indent:21.0pt'><span style='font-size:12.0pt'>在设计神经网络的过程中一般都会有许多的超参数需要进行调节，其中就包括学习率，批处理的<span
lang=EN-US>batch</span>大小，隐藏层中隐藏结点的个数，滑动平均模型的参数，模型训练的代数等等。正是由于需要调节的参数众多，采用传统的网格超参数调节的方法需要花费大量的时间，而且一般情况下，我们都是针对于特定结构的神经网络来进行参数的调节的，也就是说在设计神经网络之初，神经网络的结构基本上已经确定下来了，训练的只是不同结点之间的权重，比如传统的前馈神经网络<span
lang=EN-US>3</span>层架构输入层<span lang=EN-US>-</span>隐藏层<span lang=EN-US>-</span>输出层等，其一般层数都是固定的，而且神经元之间一般不会隔层进行连接。因为神经网络的结构千变万化，如果需要对各种不同的组合都进行尝试，其复杂度将难以想象。本文旨在通过进化算法的思想，利用遗传算法对神经网络结构中的结点以及结点之间的边进行编码，在<span
class=MsoHyperlink><span lang=EN-US><a
href="http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29">Breast
Cancer Wisconsin (Diagnostic) Data Set</a></span></span><span lang=EN-US>&nbsp;</span>数据集上进行实验，利用遗传算法的优胜劣汰的进化过程，通过选择、交叉以及变异的过程不断地更新种群，保留下适应度高的个体，最后得到的表现最佳的模型及其拓扑结构，通过机器自动学习到一个表现较佳的模型结构，并与其他方法在该数据集上的表现进行对比。</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>二、数据集介绍</span></b></p>

<p class=MsoNormal style='text-indent:21.0pt'><span style='font-size:12.0pt'>该实验采用的数据集为乳腺癌威斯康星（诊断）数据集，其源自乳腺肿瘤细胞团的细针抽吸的数字化成像的特征。
该实验的目的是利用遗传算法改善神经网络结构通过识别细胞的特征，准确地区分良性和恶性肿瘤细胞以帮助临床诊断。数据集的输入特征维度为<span
lang=EN-US>30</span>维（包括细胞的平均半径、纹理、周长、平滑度、紧凑性、凹度、凹点等），输出维度为<span lang=EN-US>2</span>维（<span
lang=EN-US>M</span>为恶心肿瘤细胞，<span lang=EN-US>B</span>为良性细胞），数据集一共有<span
lang=EN-US>569</span>个样本，其中<span lang=EN-US>M</span>类样本有<span lang=EN-US>212</span>例，<span
lang=EN-US>B</span>类样本<span lang=EN-US>357</span>例。由于该数据集的特征维度较为复杂，因此采用不同结构的神经网络将会对模型的最终表现产生较大的影响，因此为了突显该实验自动学习较优模型结构的目的，故选择采用该数据集进行实验。在数据集的预处理方面本文主要采用的方法是分别对<span
lang=EN-US>30</span>维特征的每一维度进行归一化，减小数据尺度所带来的影响。有关于数据集中特征的分布情况如下图<span
lang=EN-US>Figure1</span>和<span lang=EN-US>Figure2</span>所示。</span></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><span
lang=EN-US><img border=0 width=350 height=251 id="图片 2"
src="https://img-blog.csdnimg.cn/20190119234151579.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"></span></p>

<p class=MsoNormal style='margin-left:84.0pt;text-indent:36.0pt'><span
lang=EN-US style='font-size:9.0pt'>Figure1.</span><span style='font-size:9.0pt'>数据集中不同类样本的分布情况</span></p>

<p class=MsoNormal><span lang=EN-US><img border=0 width=554 height=258 id="图片 3"
src="https://img-blog.csdnimg.cn/20190119234213193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"></span></p>

<p class=MsoNormal style='margin-left:84.0pt;text-indent:36.0pt'><span
lang=EN-US style='font-size:9.0pt'>Figure2.</span><span style='font-size:9.0pt'>数据集中<span
lang=EN-US>M</span>类和<span lang=EN-US>B</span>类样本前十维特征的分布</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:9.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:9.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>三、算法详解</span></b></p>

<p class=MsoNormal><span style='font-size:12.0pt'>遗传算法介绍：</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>编码：</span></b><span
style='font-size:12.0pt'>利用结点集合和边集合对神经网络的结构进行编码，为了简化描述，我们采用如下<span lang=EN-US>Figure3</span>中简化的神经网络结构进行说明。考虑如下图<span
lang=EN-US>Figure3</span>所示的只包括输入层、输出层的全连接神经网络，其中输入结点和输出结点的数量均为<span
lang=EN-US>2</span>。则该神经网络的初始边集中一共包括以下<span lang=EN-US>4</span>条边<span
lang=EN-US>,</span>神经网络中的每一条边采用以下的元组进行表示：<span lang=EN-US>(</span>边的</span><span
style='font-size:11.0pt'>编号，起始结点，终止结点</span><span lang=EN-US style='font-size:
12.0pt'>)</span><span style='font-size:12.0pt'>，</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>Figure3</span><span
style='font-size:12.0pt'>网络对用的边集为：</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>[(0,</span><span
lang=EN-US style='font-size:11.0pt'>INPUT1</span><span lang=EN-US
style='font-size:12.0pt'>,</span><span lang=EN-US style='font-size:11.0pt'>OUTPUT1</span><span
lang=EN-US style='font-size:12.0pt'>),(1,</span><span lang=EN-US
style='font-size:11.0pt'>INPUT1,OUTPUT2</span><span lang=EN-US
style='font-size:12.0pt'>), (2,</span><span lang=EN-US style='font-size:11.0pt'>INPUT2,OUTPUT1</span><span
lang=EN-US style='font-size:12.0pt'>),(3,</span><span lang=EN-US
style='font-size:11.0pt'>INPUT2,OUTPUT2</span><span lang=EN-US
style='font-size:12.0pt'>)]</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>初始的结点集合为<span lang=EN-US>[</span></span><span
lang=EN-US style='font-size:11.0pt'>INPUT1, INPUT2, OUTPUT1, OUTPUT2</span><span
lang=EN-US style='font-size:12.0pt'>].</span></p>

<p class=MsoNormal style='margin-left:105.0pt;text-indent:21.0pt'><span
lang=EN-US><img border=0 width=226 height=166 id="图片 6"
src="https://img-blog.csdnimg.cn/20190119234238867.png"></span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;Figure3</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>有了以上的边集之后我们便可以对神经网络进行编码，基因型如下所示：初始种群中个体的基因型为：<span
lang=EN-US>{0: True, 1: True, 2: True, 3: True}</span>，其表示编号为<span lang=EN-US>0,1,2,3</span>的边均存在（激活）。</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>适应度函数：</span></b><span
style='font-size:12.0pt'>由于该数据集所属的问题为分类问题因此个体的适应度即为模型在测试集上面取得的准确率，由于时间有限，本文暂且仅使用准确率进行评估。</span><span
lang=EN-US style='font-size:11.0pt'>(</span><span style='font-size:11.0pt'>注，最佳的情况应该要结合<span
lang=EN-US>precision</span>和<span lang=EN-US>recall</span>进行评估，如采用<span
lang=EN-US>F1-score</span>作为适应度或采用<span lang=EN-US>recall</span>与<span
lang=EN-US>precision</span>加权相加的方式，值得一提的是，这里<span lang=EN-US>recall</span>的比重应该较大，因为对于医疗诊断的数据集，错误的把一个正例<span
lang=EN-US>(</span>恶性<span lang=EN-US>)</span>样本预测为负例<span lang=EN-US>(</span>良性<span
lang=EN-US>)</span>样本的结果往往影响较大，所以召回率的比重要大与准确率<span lang=EN-US>)</span></span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>选择：</span></b><span
style='font-size:12.0pt'>在这里我们将种群的规模设置为<span lang=EN-US>10</span>个个体，选择具体过程为：保留前五个表现最好的个体（模型），然后通过交叉和变异产生剩余的<span
lang=EN-US>5</span>个新的个体，交叉和变异的过程在下面进行介绍。</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>交叉：</span></b><span
style='font-size:12.0pt'>交叉过程采用如下的形式进行，首先随机从种群中选择出两个个体，参考下图<span lang=EN-US>figure4</span>进行说明，一共需要考虑的有以下四种情况：①如果上下均匹配上的话就随机的挑选一条边（如图中编号为<span
lang=EN-US>1,2,3,4</span>的边）②如果父代双方中存在一方边存在一方边不存在（如编号<span lang=EN-US>6,7,8</span>）就直接将存在的一方的边保留到子代③如果父代双方中出现边剩余（编号<span
lang=EN-US>9,10</span>）的话，是否保留取决于适应度较高的个体，下图中<span lang=EN-US>parent2</span>的适应度较高因此保留编号为<span
lang=EN-US>9</span>，<span lang=EN-US>10</span>的边④如果父代双方至少有一方中边的状态为<span
lang=EN-US>disable</span>，那么子代中该边的状态就置为<span lang=EN-US>disable</span>。 </span></p>

<p class=MsoNormal><span lang=EN-US><img border=0 width=553 height=499 id="图片 7"
src="https://img-blog.csdnimg.cn/20190119234247581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"></span></p>

<p class=MsoNormal><b><span lang=EN-US style='font-size:12.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure.4</span></b><span
style='font-size:12.0pt'>交叉重组的过程</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>变异：</span></b></p>

<p class=MsoNormal><span style='font-size:12.0pt'>这里变异需要分<u>结点变异</u>和<u>边变异</u>两种情况来考虑，由于神经网络中的边和结点之间存在相互依赖的关系，因此两种变异过程之间也相互影响，具体的图解过程参见如下</span><span
lang=EN-US style='font-size:11.0pt'>figure5</span><span style='font-size:11.0pt'>。</span></p>

<p class=MsoNormal style='text-indent:21.0pt'><b><span style='font-size:12.0pt'>边变异</span></b><span
style='font-size:12.0pt'>：选择种群中表项较佳的个体（模型），获取该模型当前激活的边集并导出当前激活的边集上的所有结点。在所有的结点中选取出两个结点，然后在两个结点之间形成一条边<span
lang=EN-US>(</span>考虑特殊情况的处理<span lang=EN-US>)</span>。</span></p>

<p class=MsoNormal style='text-indent:21.0pt'><b><span style='font-size:12.0pt'>结点变异</span></b><span
style='font-size:12.0pt'>：选择种群中表项较佳的模型，获取模型当前激活的边集，从该集合中随机选出一条边，然后在这两条边中间加入一个结点，如下图<span
lang=EN-US>figure5</span>所示：由于在边<span lang=EN-US>(3,4)</span>之间加入了一个结点<span
lang=EN-US>6</span>，故产生了两条新的边<span lang=EN-US>(3,6),(6,4)</span>同时也使得原先的边<span
lang=EN-US>(3,4)</span>的状态变为了<span lang=EN-US>disable</span>，对应的将该个体中（<span
lang=EN-US>3,4</span>）的边设置为<span lang=EN-US>False</span>，在基因型的后面加入对应的两条新产生的边。</span></p>

<p class=MsoNormal><span lang=EN-US><img border=0 width=554 height=365 id="图片 8"
src="https://img-blog.csdnimg.cn/20190119234307258.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"
alt="http://5b0988e595225.cdn.sohucs.com/images/20171005/9d9b5a3b24834ab08b7fcc5e563a7cf8.jpeg"></span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Figure.5</b>
</span><span style='font-size:12.0pt'>变异的过程，上图为增加边的变异，下图为增加结点的变异</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>在定义好以上的关于基础算法的基本操作之后，我们就可以利用遗传算法的流程进行神经网络的训练。具体训练的算法流程图如下<span
lang=EN-US>Figure6</span>所示，</span></p>

<p class=MsoNormal style='text-indent:24.0pt'><span lang=EN-US
style='font-size:12.0pt'><img border=0 width=552 height=367 id="图片 12"
src="https://img-blog.csdnimg.cn/20190119234322860.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"></span></p>

<p class=MsoNormal style='margin-left:84.0pt;text-indent:21.0pt'><b><span
lang=EN-US style='font-size:12.0pt'>Figure.6</span></b><span style='font-size:
12.0pt'>遗传算法的流程图</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>神经网络在训练过程中的结构变化可以用下图<span
lang=EN-US>Figure7</span>进行简略地表示。</span></p>

<p class=MsoNormal><span lang=EN-US><img border=0 width=536 height=310 id="图片 4"
src="https://img-blog.csdnimg.cn/2019011923433474.png"
alt="https://images2015.cnblogs.com/blog/524764/201609/524764-20160905211413582-1926711200.png"></span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Figure.7</b></span><span
style='font-size:12.0pt'>神经网络的进化过程</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>这里需要补充说明的是，在计算适应度函数的过程中，我们采用的是基于反向传播的梯度下降的方法来训练神经网络，本文的实验过程是基于<span
lang=EN-US>TensorFlow</span>深度学习框架实现的，主要细节包括采用了<span lang=EN-US>Adam</span>优化器，指数衰减的学习率，<span
lang=EN-US>sigmoid</span>交叉熵损失函数<span lang=EN-US>,</span>在神经网络的训练中，网络训练的<span
lang=EN-US>epoch</span>与边集的大小成正比，即网络的结构越复杂需要训练的代数就越多。</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>四、实验以及相关对比</span></b></p>

<p class=MsoNormal><span style='font-size:12.0pt'>在实验中考虑到数据集规模较小，本文将数据集按照训练集<span
lang=EN-US>:</span>测试集<span lang=EN-US>=7:3</span>的比例进行划分。</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>模型训练过程汇总的参数设置如下：</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>进化代数<span lang=EN-US>20</span>代，种群的大小<span
lang=EN-US>10</span>，初始神经网络输入节点<span lang=EN-US>30</span>个，输出结点<span
lang=EN-US>2</span>个，边变异的概率为<span lang=EN-US>0.2</span>，结点变异的概率为<span
lang=EN-US>0.1</span>，交叉重组的概率为<span lang=EN-US>0.7</span></span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>下图<span lang=EN-US>Figure8</span>为第一代种群中每一个个体<span
lang=EN-US>(</span>模型<span lang=EN-US>)</span>的适应度（准确率）：</span></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><span
lang=EN-US><img border=0 width=314 height=198 id="图片 5"
src="https://img-blog.csdnimg.cn/20190119234347245.png"></span></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><span
lang=EN-US style='font-size:11.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure</span><span
style='font-size:11.0pt'>第一代种群中每一个个体的适应度</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:11.0pt'>Figure9</span><span
style='font-size:11.0pt'>为训练过程汇总出现的变现最佳的个体的代数，可以看见在第<span lang=EN-US>14</span>代的时候出现了变现最佳的个体</span></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><span
lang=EN-US><img border=0 width=330 height=225 id="图片 10"
src="https://img-blog.csdnimg.cn/20190119234359132.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"></span></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><span
lang=EN-US style='font-size:11.0pt'>Figure9.</span><span style='font-size:11.0pt'>第<span
lang=EN-US>14</span>代中第<span lang=EN-US>10</span>个个体为变现最佳个体</span></p>

<p class=MsoNormal style='margin-left:42.0pt;text-indent:21.0pt'><span
lang=EN-US style='font-size:11.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>利用<span lang=EN-US>TensorFlow</span>的模型可视化工具，对表现最佳模型的计算图的结构进行可视化，得到以下结果：</span></p>

<p class=MsoNormal><span lang=EN-US><img border=0 width=554 height=281 id="图片 1"
src="https://img-blog.csdnimg.cn/20190119234410335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MDUzODg1,size_16,color_FFFFFF,t_70"></span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>在如下<span lang=EN-US>Table1</span>中本文对比了该算法与主要的分类算法在该数据集上的表现效果。</span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>其主要包括以下对比算法：</span><span
lang=EN-US style='font-size:11.0pt'>LogisticsRegression(<b>LR</b></span><span
style='font-size:11.0pt'>）、<span lang=EN-US>LinerDiscriminateAnalysis(<b>LDA</b>)</span>、<span
lang=EN-US>KNeighborsClassfier(<b>KNN</b>)</span>、<span lang=EN-US>RandomForestClassfier(<b>RFC</b>)</span>、<span
lang=EN-US>SupportVectorMachine</span>（<b><span lang=EN-US>SVM</span></b><span
lang=EN-US>)</span></span></p>

<p class=MsoNormal><span style='font-size:12.0pt'>表格中所得的准确率为以上<span lang=EN-US>5</span>种算法在数据上进行<span
lang=EN-US>100</span>次试验所获得的平均值。</span></p>

<p class=MsoNormal><span style='font-size:11.0pt'>在测试集上</span><span lang=EN-US
style='font-size:12.0pt'>5</span><span style='font-size:12.0pt'>类待对比的算法准确率如下所示：</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:11.0pt'>&nbsp;</span></p>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=567
 style='width:425.0pt;border-collapse:collapse;border:none'>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span style='font-size:12.0pt'>实验方法</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>Precision</span></p>
  </td>
 </tr>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>LR(C=100)</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>0.9172</span></p>
  </td>
 </tr>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>LDA</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>0.9569</span></p>
  </td>
 </tr>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>KNN</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>0.9235</span></p>
  </td>
 </tr>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>RFC</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>0.9603</span></p>
  </td>
 </tr>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>SVM(C=100,kernel=’rbf’)</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>0.9067</span></p>
  </td>
 </tr>
 <tr>
  <td width=283 valign=top style='width:212.4pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>Ours</span></p>
  </td>
  <td width=283 valign=top style='width:212.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b><span lang=EN-US style='font-size:12.0pt'>0.9736</span></b></p>
  </td>
 </tr>
</table>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
lang=EN-US style='font-size:11.0pt'>Table1.</span><span style='font-size:11.0pt'>进化神经网络与其他算法的对比结果<span
lang=EN-US>(</span>指标：测试集上的准确率<span lang=EN-US>)</span></span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:11.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:11.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>五、总结与反思</span></b></p>

<p class=MsoNormal style='text-indent:21.0pt'><span style='font-size:12.0pt'>在实验中我们发现，能获得该结果是令人高兴的。当该方法也存在局限性，其中最为突出的便是需要耗费大量的计算资源，由于每一个个体均是一个模型，采用启发式的方法搜索模型结构的解空间的计算复杂度将会很大，在大规模的数据集上面将会对计算设备提出很高的要求。而该方法后续也存在很多值得期待的改进点，如将其应用到卷积神经网络中<span
lang=EN-US>filter</span>的选择以及卷积层池化层的架构中，通过启发式搜索的方法得到模型最佳表现的架构，并对相应的结构进行分析比对，总结卷积神经网络的设计经验。总之，进化算法在神经网络的应用上还有许多需要拓展的领域和令人期待的地方。</span></p>

<p class=MsoNormal style='text-indent:21.0pt'><span lang=EN-US
style='font-size:12.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:12.0pt'>六、参考文献</span></b></p>

<p class=MsoNormal style='margin-left:11.0pt;text-indent:-11.0pt'><span
lang=EN-US style='font-size:11.0pt'>[1]Kenneth O. Stanley,Risto Miikkulainen.Evolving
Neural Networks through Augmenting Topologies[J].The MIT Press
Journals,2002,10(2):99-127.</span></p>

</div>

</body>

</html>
