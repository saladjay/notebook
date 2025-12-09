graphrag：利用大模型从传入文档里提取节点（实体）和边（关系）, 然后用社区检测算法对整个知识图谱进行划分，划分成多个包含了相关性较高的节点和边的子图，利润也大模型对每个子图进行总结，生成社区报告（摘要）,利用摘要描述每个子图的概况。



# Index

1. Loading Input。读取文件，转换成dataframe
2. Documents into text chunks。将文档切割成chunk
3. Graph extraction。将

### txt Index

#### loading Input

按照utf-8的格式读取文件，将所有文件制作成一个dataframe。读取文件前可以配置settings.yaml，可以设置筛选项等。

| id                     | title  | creation_date | text     | others(optional)              |
| ---------------------- | ------ | ------------- | -------- | ----------------------------- |
| 计算得到sha512hash数值 | 文件名 | 文件创建日期  | 文件内容 | 其他group（还不知道怎么产生） |





## Json Index

### data preprocess

1. 去除json中空字典，替换成null
2. 同类信息合并成一个json
3. 修改load和save parquet的逻辑，将dict对象和json对象以字符串的形式存储，读取的时候在变换回来

### create base text units

无修改

### create final documents

无修改

### extract graph

修改了prompt

太小的模型需要修改prompt的，适配相应的数据





## 意图识别

1. 训了一个意图识别网络
