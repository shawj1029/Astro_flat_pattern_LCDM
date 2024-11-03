import cupy as cp
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from Parallel_programs_5deg import count
from tqdm import tqdm,trange
import time
import json
import h5py
import os

warnings.filterwarnings('ignore', category=FutureWarning)

def process_data_on_gpu(gpu_id, data, each_size):
    with cp.cuda.Device(gpu_id):
        batch_results = cp.zeros((72, 18, 21))  # initialize the result array in cupy format
        for start in trange(0, data.shape[0], each_size, desc=f'{file}',mininterval=60):  # read rows each time
            end = min(start + each_size, data.shape[0])
            batch_data = data[start:end]
            batch_gpu = cp.asarray(batch_data)  # transfer the data to gpu
            counting = count(batch_gpu, redshift).count('inner') # massive calculations
            # counting
            batch_results += counting
            # release the memory
            del batch_gpu, counting
            cp.get_default_memory_pool().free_all_blocks()
            
        batch_results = cp.asnumpy(batch_results)  # return the result in numpy format

    return batch_results

redshift = 0.021
split_number = 9
count_ = 689

for file in sorted(os.listdir('/home/jiangxiaoqi/New/stage_2_pair/pairs/offset'))[689:]:
    if file.endswith('.npy'):
        data = np.load(os.path.join('/home/jiangxiaoqi/New/stage_2_pair/pairs/offset', file))

        print(count_)

        data_split = data.shape[0] // split_number
        split_ratio = [data_split] * split_number
                        #data_split, data_split, data_split, data_split, data_split, data_split, data_split, data_split]
        split_points = np.cumsum(split_ratio)[:-1]
        sub_data = np.split(data, split_points) # This is to split the data into 4 parts with the third one smaller

        sub_data_map = sub_data_map = [sub_data[i] for i in range(split_number)]
        #, sub_data[8], sub_data[9], sub_data[10], sub_data[11], sub_data[12], sub_data[13], sub_data[14], sub_data[15]]
        gpu_data_map = [0,0,0,1,1,1,3,3,3]
        each_size = [17000] * split_number
        #,20000,20000,20000,20000,20000,20000,20000,20000] # Process numbers for each loop

        a = time.time()

        with ProcessPoolExecutor(max_workers = split_number) as executor: # open 16 processes
            futures = [executor.submit(process_data_on_gpu, gpu_id, data, each_size) for gpu_id, data, each_size in zip(gpu_data_map, sub_data_map, each_size)]
            results = [f.result() for f in futures]

        final_result = np.zeros((72,18,21))
        for i in range(split_number):
            final_result += results[i] # sum the result from 2 processes

        print(f'{os.path.splitext(file)[0]},time:',round(time.time()-a,2))
        np.save(f'/home/jiangxiaoqi/New/stage_3_parallel_program_5deg/parallel_program_results/{file}', final_result)

        count_ += 1
