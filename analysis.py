import os
import glob
import matplotlib.pyplot as plt

def count_files(directory):
    return len(glob.glob(os.path.join(directory, '*')))

train_counts = []
validation_counts = []

# 遍历train文件夹下的所有子文件夹
for i in range(7):
    train_counts.append(count_files('./images/train/0' + str(i)))

# 遍历validation文件夹下的所有子文件夹
for i in range(7):
    validation_counts.append(count_files('./images/test/0' + str(i)))

# 创建条形图
plt.bar(range(7), train_counts, label='Train')
plt.bar(range(7), validation_counts, bottom=train_counts, label='Validation')

# 在每个条形图上添加具体数量
for i, (t, v) in enumerate(zip(train_counts, validation_counts)):
    plt.text(i, t+v+1, str(t)+','+str(v))

plt.xlabel('Subfolder')
plt.ylabel('File Count')
plt.xticks(range(7), ['00', '01', '02', '03', '04', '05', '06'])
plt.legend()
plt.show()