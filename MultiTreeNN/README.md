## 预处理

源代码表示成：

- NCS
- AST
- DDG（分割成多个DDT树，每个变量一个）
- CDG（同理分成多个CDT，得到从始至终的控制流）
- 其他

## 模型

- 用RNN学习NCS，用一个Tree-RNN（树形循环神经网络，现成的有Tree-LSTM）去学习每个特征，其中Tree-RNN有很多通道和隐含层
- 将上一层的输出分别进行pooling得到高层表征
- 然后利用一层带有attention的RNN将特征结合起来（由于输入树的个数不知道，不方便用MLP）

## 训练

## 服务器

wutiru：387G √

xiaowentao：123G

maokelong：66G

xiaorenjie：109G

yangzhenyu：48G

chenyuzhao：32K

pengzhendong：5.5M

zawnpn：32G

liyi：139G

lifuyang：29G

wuruohao：2G

zlp：21G
