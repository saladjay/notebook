import struct
import io
import traceback
class weightsWriter:
    def __init__(self, filename):
        self.filename = filename

    def write(self):
        # 写入二进制文件
        with open(self.filename, 'wb') as file:
            file.write(struct.pack('i', 20)) # 写入数组长度
            for nb_weights in range(1,21):
                file.write(struct.pack('i', 2 if nb_weights < 10 else 3)) # 写入每个数组名字的长度
                file.write(struct.pack(f'{2 if nb_weights < 10 else 3}s', f's{nb_weights}'.encode('utf-8'))) # 写入每个数组元素的长度
                file.write(struct.pack('i', nb_weights)) # 写入每个数组元素的长度
                for n in range(1,nb_weights+1):
                    file.write(struct.pack('d', 0.0001 * n))

a = "ab"
try:
    struct.pack('2s', a.encode('utf-8'))
except Exception as e:
    print("Error:", e)
    traceback.print_exc()

# 示例浮点数组
float_array = [1.5, 2.2, 3.3, 4.4, 5.5]
print(float_array)

# 将浮点数组序列化为二进制格式并写入文件
with open('float_array.bin', 'wb') as file:
    for number in float_array:
        file.write(struct.pack('d', number))

# 从二进制文件中读取浮点数组
loaded_array = []
with open('float_array.bin', 'rb') as file:
    while chunk := file.read(8):
        loaded_array.append(struct.unpack('d', chunk)[0])

print(loaded_array)

if __name__ == "__main__":
    try:
        w = weightsWriter("weight.bin")
        w.write()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        # print(e)