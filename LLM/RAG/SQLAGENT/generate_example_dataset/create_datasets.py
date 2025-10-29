import sqlite3
import random
import json
from generate_example_dataset.pseudo_region_utils import get_pseudo_label_regions
def create_database(db_path):
    """
    使用 sqlite3 创建一个空数据库（如果不存在则新建）
    :param db_path: 数据库文件路径
    """
    conn = sqlite3.connect(db_path)
    print(f"已创建/连接到数据库: {db_path}")
    # conn.close()
    return conn

Dataset_schema = "CREATE TABLE DataSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Additional_data BLOB)" 
Image_schema = "CREATE TABLE Image(Id INTEGER PRIMARY KEY NOT NULL,Path TEXT NOT NULL,DataSet_id INTEGER KEY NOT NULL,Tag_id INTEGER KEY NOT NULL,Additional_data BLOB,UNIQUE(Path,DataSet_id)FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Tag_id) REFERENCES TagSet(Id))"
Label_schema = "CREATE TABLE Label(Id INTEGER PRIMARY KEY NOT NULL,DataSet_id INTEGER NOT NULL,Image_id INTEGER NOT NULL,Label_class_id INTEGER NOT NULL,RegionType INTEGER NOT NULL,Region BLOB,FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Image_id) REFERENCES Image(Id),FOREIGN KEY(Label_class_id) REFERENCES LabelClass(Id))"
TagSet_schema = "CREATE TABLE TagSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,SpecialOp INTEGER DEFAULT (-1),ShortCut TEXT,Additional_data BLOB)"
LabelClass_schema = "CREATE TABLE LabelClass(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Color TEXT UNIQUE DEFAULT NULL,ShortCut TEXT,Additional_data BLOB)"

def create_tables(conn):
    """
    创建表
    :param conn: 数据库连接
    """
    # INSERT_YOUR_CODE
    # 判断表存在了就不创建
    cursor = conn.cursor()
    existing_tables = set()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for row in cursor.fetchall():
        existing_tables.add(row[0])

    schema_table_names = [
        ("DataSet", Dataset_schema),
        ("Image", Image_schema),
        ("Label", Label_schema),
        ("TagSet", TagSet_schema),
        ("LabelClass", LabelClass_schema)
    ]
    for table_name, schema in schema_table_names:
        if table_name not in existing_tables:
            conn.execute(schema)
    conn.commit()
    # conn.close()

category_list = set(["破损", "划痕", "脏污", "折痕", "条码", "缺角", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
"良品", "缺陷", "其他", "缺料", "杂物", "大不良", "夹废", "夹异物", "沾污", "污点", "锈蚀", "脏污", "划伤", "裂纹", "腐蚀", "缺料", "杂物", "大不良", "夹废", "夹异物", "沾污", "污点", "锈蚀", "脏污"])

def _random_color():
    color = '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    return color

shortcuts = set(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
def _shortcut():
    global shortcuts
    if len(shortcuts) == 0:
        return None
    if random.random() < 0.5:
        return None
    shortcut = random.choice(list(shortcuts))
    shortcuts.remove(shortcut)
    return shortcut

productId_set = set()
modelIndex_set = set()



def get_pseudo_label_class(nb_classes: int):
    def _get_pseudo_label_class_additional_data():
        global productId_set, modelIndex_set
        productId = len(productId_set)
        productId_set.add(productId)
        modelIndex = len(modelIndex_set)
        modelIndex_set.add(modelIndex)
        labelClassType = random.choice([0, 1])
        return {
            "labelClassType": labelClassType, # 0: 良品, 1: 缺陷
            "linearity": 3, # 线性区分度
            "modelIndex": modelIndex, # 模型输出的下标
            "productId": productId # 料ID
        }
    for i in range(nb_classes):
        category = random.choice(list(category_list))
        category_list.remove(category)
        additional_data = _get_pseudo_label_class_additional_data()
        shortcut = _shortcut()
        color = _random_color()
        labelClass = {"Name": category, "Color": color, "ShortCut": shortcut, "Additional_data": additional_data}
        yield labelClass
        

TAG_set = set(["一般", "特殊", "其他", "未标注","正品","残次品",])
def get_pseudo_tag(nb_tags: int):
    nb_tags = nb_tags - 6
    for TAG in ['默认', "良品", "漏检", "误检", "待定", "重要"]:
        tag = {"Name":TAG,"SpecialOp":-1,"ShortCut":_shortcut(),"Additional_data":None}
        yield tag
    for i in range(nb_tags):
        if len(TAG_set) == 0:
            break
        tag_name = random.choice(list(TAG_set))
        TAG_set.remove(tag_name)
        tag = {"Name":tag_name,"SpecialOp":-1,"ShortCut":_shortcut(),"Additional_data":None}
        yield tag


first_folders = ['20251027', '20251028', '20251029', '20251030', '20251031', '20251101', '20251102', '20251103', '20251104', '20251105']
second_folders = ['saved', 'loujian', 'wujian', 'daibing', 'zhongyao']
third_folders = ['ok', 'no', 'other', None, None]
years = ['2025', '2024', '2023', '2026', '2027', '2028']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
minutes = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59']
seconds = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59']
unique_constraint_set_4_path_dataset_id = set()
def get_pseudo_image(nb_datasets, nb_tags, nb_images):
    def _get_pseudo_image_path():
        global first_folders, second_folders, third_folders, years, months, days, hours, minutes, seconds
        first_folder = random.choice(first_folders)
        second_folder = random.choice(second_folders)
        third_folder = random.choice(third_folders)
        image_name = f"{random.choice(years)}_{random.choice(months)}_{random.choice(days)}_{random.choice(hours)}_{random.choice(minutes)}_{random.choice(seconds)}_{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}.jpg"
        return f"D:/data/{first_folder}/{second_folder}/{third_folder}/{image_name}.jpg" if third_folder is not None else f"D:/data/{first_folder}/{second_folder}/{image_name}.jpg"
    for i in range(nb_images):
        dataset_id = random.choice(nb_datasets)
        path = _get_pseudo_image_path()
        combined_path_dataset_id = f"{path}_{dataset_id}"
        if combined_path_dataset_id in unique_constraint_set_4_path_dataset_id:
            continue
        unique_constraint_set_4_path_dataset_id.add(combined_path_dataset_id)
        yield {
            "Path": path,
            "DataSet_id": dataset_id,
            "Tag_id": random.choice(nb_tags),
            "Additional_data": None
        }
    

dataset_suffix = set(["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10", "train", "test", "val"])
def get_pseudo_dataset(nb_datasets):
    def _get_dataset_additional_data():
        return {
            "locked": True
        } if random.random() < 0.3 else None
    for i in range(nb_datasets):
        dataset_name = f"{random.choice(years)}_{random.choice(months)}_{random.choice(days)}"
        if random.random() < 0.3:
            pass
        else:
            dataset_name += random.choice(list(dataset_suffix))
        data = {"Name": dataset_name, "Additional_data": _get_dataset_additional_data()}
        yield data

policy1 = "每个图片都有一个标签"
policy2 = "50%图片有标签"
policy3 = "图片都有多个标签，部分图片没有标签"
policy4 = "图片都有多个标签，标签重合度很高"
def get_pseudo_label(nb_labels, nb_datasets, nb_tags, nb_images):
    dataset_id_list = [i+1 for i in range(nb_datasets)]
    image_id_list = [i+1 for i in range(nb_images)]
    label_class_id_list = [i+1 for i in range(nb_tags)]
    for i in range(nb_labels):
        # policy = random.choice([policy1, policy2, policy3, policy4])
        policy = policy1
        if policy == policy1:
            yield {
                "DataSet_id": random.choice(dataset_id_list),
                "Image_id": random.choice(image_id_list),
                "Label_class_id": random.choice(label_class_id_list),
                "RegionType": 1,
                "Region": get_pseudo_label_regions(nb_regions = 1)[0]
            }

def create_example_dataset(db_file:str = "example_dataset.db", nb_classes:int = 15, nb_tags:int = 10, nb_datasets:int = 10, nb_images:int = 10000, nb_labels:int = 1000000):
    # 示例用法：创建名为 example_dataset.db 的数据库
    # db_file = "example_dataset.db"
    try:
        conn = create_database(db_file)
        create_tables(conn)
        labelClass_list = []
        # 插入类别数据
        for labelClass in get_pseudo_label_class(nb_classes):
            try:
                # 将 Additional_data 序列化为 JSON 字符串
                additional_data_json = json.dumps(labelClass["Additional_data"]) if labelClass["Additional_data"] else None
                
                if labelClass['ShortCut'] is None:
                    conn.execute("INSERT INTO LabelClass (Name, Color, Additional_data) VALUES (?, ?, ?)", 
                            (labelClass["Name"], labelClass["Color"], additional_data_json))
                else:
                    conn.execute("INSERT INTO LabelClass (Name, Color, ShortCut, Additional_data) VALUES (?, ?, ?, ?)", 
                            (labelClass["Name"], labelClass["Color"], labelClass["ShortCut"], additional_data_json))
                conn.commit()
                labelClass_list.append(labelClass)
            except Exception as e:
                print(f"Error inserting label class: {e}")
                continue

        tag_list = []
        for tag in get_pseudo_tag(nb_tags):
            try:
                # 将 Additional_data 序列化为 JSON 字符串
                additional_data_json = json.dumps(tag["Additional_data"]) if tag["Additional_data"] else None
                
                conn.execute("INSERT INTO TagSet (Name, SpecialOp, ShortCut, Additional_data) VALUES (?, ?, ?, ?)", 
                        (tag["Name"], tag["SpecialOp"], tag["ShortCut"], additional_data_json))
                conn.commit()
                tag_list.append(tag)
            except Exception as e:
                print(f"Error inserting tag: {e}")
                continue
        
        print(f"start inserting dataset")
        dataset_list = []
        for dataset in get_pseudo_dataset(nb_datasets):
            try:
                additional_data_json = json.dumps(dataset["Additional_data"]) if dataset["Additional_data"] else None
                conn.execute("INSERT INTO DataSet (Name, Additional_data) VALUES (?, ?)", (dataset["Name"], additional_data_json))
                conn.commit()
                dataset_list.append(dataset)
                print(f"Inserted dataset: {dataset['Name']}")
            except Exception as e:
                print(f"Error inserting dataset: {e}")
                continue
        
        print(f"start inserting image")
        image_list = []
        for image in get_pseudo_image([i+1 for i in range(nb_datasets)], [i+1 for i in range(nb_tags)], nb_images):
            try:
                conn.execute("INSERT INTO Image (Path, DataSet_id, Tag_id, Additional_data) VALUES (?, ?, ?, ?)", (image["Path"], image["DataSet_id"], image["Tag_id"], image["Additional_data"]))
                conn.commit()
                image_list.append(image)
            except Exception as e:
                print(f"Error inserting image: {e}")
                continue

        label_list = []
        for label in get_pseudo_label(nb_labels, nb_datasets, nb_tags, nb_images):
            try:
                region_json = json.dumps(label["Region"]) if label["Region"] else None
                conn.execute("INSERT INTO Label (DataSet_id, Image_id, Label_class_id, RegionType, Region) VALUES (?, ?, ?, ?, ?)", (label["DataSet_id"], label["Image_id"], label["Label_class_id"], label["RegionType"], region_json))
                conn.commit()
                label_list.append(label)
            except Exception as e:
                print(f"Error inserting label: {e}")
                continue

        conn.close()
    except Exception as e:
        print(f"Error creating example dataset: {e}")
        conn.close()