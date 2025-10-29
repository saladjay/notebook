prompt = """你是一个深度学习专家，拥有一个包含深度学习目标检测标注数据的数据库。

---数据库结构---
==========================================
CREATE TABLE DataSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Additional_data BLOB)
CREATE TABLE Image(Id INTEGER PRIMARY KEY NOT NULL,Path TEXT NOT NULL,DataSet_id INTEGER KEY NOT NULL,Tag_id INTEGER KEY NOT NULL,Additional_data BLOB,UNIQUE(Path,DataSet_id)FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Tag_id) REFERENCES TagSet(Id))
CREATE TABLE Label(Id INTEGER PRIMARY KEY NOT NULL,DataSet_id INTEGER NOT NULL,Image_id INTEGER NOT NULL,Label_class_id INTEGER NOT NULL,RegionType INTEGER NOT NULL,Region BLOB,FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Image_id) REFERENCES Image(Id),FOREIGN KEY(Label_class_id) REFERENCES LabelClass(Id))
CREATE TABLE LabelClass(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Color TEXT UNIQUE DEFAULT NULL,ShortCut TEXT,Additional_data BLOB)
CREATE TABLE TagSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,SpecialOp INTEGER DEFAULT (-1),ShortCut TEXT,Additional_data BLOB)
==========================================

---相关信息---
==========================================
1. 每个项目都有上述五张表。
2. 每个项目的数据集都有一个唯一的Id，Name是数据集的名称，Additional_data是数据集的额外数据。
3. 每个项目提供了人工创建的图片Tag, 每张图片只能拥有一个Tag, 方便软件界面进行Tag筛选。
4. 每个数据集都有多张图片，每张图片都有一个唯一的Id，Path是图片的路径，DataSet_id是数据集的Id，Tag_id是Tag的Id，Additional_data是图片的额外数据。不同数据集的图片的物理地址可以相同。
5. Label中的Region是图片中目标的标注信息，是json格式的数据。
==========================================

请根据以上信息生成多条text2sql的查询问题。问题主题是{subject}。不需要输出SQL语句。回答里不得出现表结构信息和列名。只能使用人类语言。请生成至少十个以上的问题。
输出格式为：
{
    "subject": "{subject}",
    "question": ['问题1', '问题2', '问题3'],
}
"""