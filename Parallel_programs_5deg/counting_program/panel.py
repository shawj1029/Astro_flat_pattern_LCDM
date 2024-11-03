import cupy as cp
from .initialization import Directions, Distance

class panel():
    def __init__(self, data, redshift): 
    # h是方向矢的总高度，largest_distance是最外面的盘中的粒子距离零点的最大距离
        lightspeed = 3e5
        H_0 = 70
        self.redshift = redshift
        self.data = data
        self.directions = Directions(data)
        self.distance = Distance(data)
        self.largest_distance = self.redshift * lightspeed / H_0
        self.delta_h = self.largest_distance / cp.sqrt(3) / 10.5 # 确保探针盘的厚度直径比为 1/29.7

    def direction_vector(self):
        direction_vector = Directions(self.data).initial_vector() * self.delta_h * 10.5
        # shape => (72, 18, 3)
        return direction_vector
    
    def penels(self): # 这个函数给出了所有的探针中心位置
    # 生成探针盘
        probe = Directions(self.data).initial_vector()
        probe = cp.reshape(probe, (72, 18, 1, 3))
        # shape => (72, 18, 1, 3)
        height = cp.linspace(-10.5 * self.delta_h, 10.5 * self.delta_h, 21, endpoint = True)
        height = cp.reshape(height, (1, 1, 21, 1))
        # shape => (1, 1, 21, 1)
        probe *= height
    # 生成21个探针盘，第四维度是探针中心的坐标
        del height
        cp.get_default_memory_pool().free_all_blocks()
    # 释放内存
        return probe
    # shape => (72, 18, 21, 3)

    def parallel_distance(self): # 这个函数给出了所有数据点到探针盘中心的平行距离
    # 生成所有数据平行于方向矢相对探针中心的距离
        direction_vector = Directions(self.data).initial_vector()
        # shape => (72, 18, 3) 这是单位的方向矢
        direction_vector = cp.reshape(direction_vector, (72, 18, 1, 3))
        # shape => (72, 18, 1, 3)
        direction_vector = cp.repeat(direction_vector, 21, axis = 2)
        # shape => (72, 18, 21, 3) # 这样得到了一个21个探针盘的方向矢，为接下来的计算做准备
        real_distance = cp.einsum('ij,klmj->klmi', self.data, direction_vector, dtype = cp.float32) # (-1,3)和(72,18,21,3)的内积
        # shape => (72, 18, 21, -1) 这是数据点和方向矢的内积
        height = cp.linspace(-10.5 * self.delta_h, 10.5 * self.delta_h, 21, endpoint = True, dtype = cp.float32)
        height = cp.reshape(height, (1, 1, 21, 1))
        # shape => (1, 1, 21, 1) 这是探针的高度，需要用内积减去
        parallel_distance = real_distance - height
    # shape => (72, 18, 21, -1) 这是所有数据点到探针中心的平行距离
        del direction_vector, real_distance, height
        cp.get_default_memory_pool().free_all_blocks()
    # 释放内存
        return parallel_distance # 正负都可以取 (float32)