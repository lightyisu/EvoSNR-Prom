#step2!
import pandas as pd
from sklearn.model_selection import train_test_split
#train -6 dev -2 test -2
# exec(open('/root/autodl-tmp/evosnr_0605/preprocessing/tools/configurator_preprocessing.py').read())

specie='Esch'
data_path='/root/autodl-tmp/evosnr_0605/data/'
target_prom=data_path+f'{specie}/'+'all_PCA.csv'
split_path=data_path+f'{specie}/'+'split/'
# 读取数据
print(target_prom)
df = pd.read_csv(target_prom)

# 第一次分割：80% 训练+验证，20% 测试
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 第二次分割：训练+验证 中的 75% 训练，25% 验证（整体比例 60:20:20）
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# 保存文件
train_df.to_csv(split_path + "split_train.csv", index=False)
val_df.to_csv(split_path +  "split_dev.csv", index=False)
test_df.to_csv(split_path + "split_test.csv", index=False)

print(f'export to {split_path + " split_train.csv"}')
# 打印信息
print(f"总数据: {len(df)}")
print(f"训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"验证集: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")