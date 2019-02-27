## Chinese Entity Recognize 2019-3

#### 1.preprocess

generate() 根据 template 采样实体进行填充、生成数据，可省去、替换

merge_sent() 和 label_sent() 分别对 univ、extra 标注，汇总、打乱

#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.represent

label2ind() 增设标签 N，add_buf() 再对 cnn_sent 头部、尾部进行填充

#### 4.build

train 80% / dev 20% 划分，通过 dnn 的 trm 构建实体识别模型

#### 5.recognize

predict() 填充为定长序列、每句返回 (word, pred) 的二元组

#### 6.interface

merge() 将 BIO 标签组合为实体，response() 返回 json 字符串