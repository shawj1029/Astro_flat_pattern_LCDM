import numpy as np
import os
import json
from tqdm import tqdm

file_path = '/home/jiangxiaoqi/New/stage_3_parallel_program_5deg/parallel_program_large' # 这是large的数据，共1000个

with open('/home/jiangxiaoqi/New/stage_4_analysis/l2s_analysis/large/density_large.json', 'r') as f: # 从density_large中获得density
    density = json.load(f)

count_dict_500 = {}
loc_dict_500 = {}
count_dict_600 = {}
loc_dict_600 = {}
count_dict = {}
loc_dict = {}
count_loc = {}

for file_name in tqdm(os.listdir(file_path), desc='Loading data', unit='file'):
    if file_name.endswith('.npy'):
        data = np.load(os.path.join(file_path, file_name))
        count = int(np.sum(data>density[file_name]*500/8540)) # 判断slab的值是否大于阈值
        if count:
            count_dict_500[file_name] = count
            loc = np.where(data>density[file_name]*500/8540)
            loc_dict_500[file_name] = [loc[0].tolist(), loc[1].tolist()]

for file_name in tqdm(os.listdir(file_path), desc='Loading data', unit='file'):
    if file_name.endswith('.npy'):
        data = np.load(os.path.join(file_path, file_name))
        count = int(np.sum(data>density[file_name]*600/8540)) # 判断slab的值是否大于阈值
        if count:
            count_dict_600[file_name] = count
            loc = np.where(data>density[file_name]*600/8540)
            loc_dict_600[file_name] = [loc[0].tolist(), loc[1].tolist()]

count_dict['500'] = count_dict_500
loc_dict['500'] = loc_dict_500
count_dict['600'] = count_dict_600
loc_dict['600'] = loc_dict_600
count_loc['count'] = count_dict
count_loc['loc'] = loc_dict

with open('/home/jiangxiaoqi/New/stage_4_analysis/l2s_analysis/large/count_loc.json', 'w') as f:
    json.dump(count_loc, f)

# 这个脚本保存了500、600的观察者文件名和每一个满足条件观察者下满足条件的slab的数量和角度
                                                              
                        # count_dict_500 -> file_name -> 500 count
            # count_dict  count_dict_600 -> file_name -> 600 count
# count_loc
            # loc_dict    loc_dict_500 -> file_name -> 500 loc
                        # loc_dict_600 -> file_name -> 600 loc

# json文件的内容如上