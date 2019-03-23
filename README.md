## Chinese Entity Recognize 2019-2

#### 1.preprocess

generate() 根据 template 采样实体进行填充、生成数据，可省去、替换

merge_sent() 和 label_sent() 分别对 univ、extra 标注，汇总、打乱

#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.represent

label2ind() 增设标签 N，sent2ind() 将每句转换为词索引并填充为相同长度

#### 4.build

通过 trm 构建实体识别模型，对编码器词特征 x 多头线性映射

得到 q、k、v，使用点积注意力得到语境向量 c、再线性映射进行降维

#### 5.recognize

predict() 填充为定长序列、每句返回 (word, pred) 的二元组

#### 6.interface

merge() 将 BIO 标签组合为实体，response() 返回 json 字符串