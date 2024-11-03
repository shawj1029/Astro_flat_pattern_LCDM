from .initialization import Distance
from .panel import panel
import cupy as cp

class judge():
    def __init__(self, data, redshift):
        lightspeed = 3e5
        H_0 = 70
        self.redshift = redshift
        self.data = data
        self.largest_distance = self.redshift * lightspeed / H_0
        self.radius_outer = self.largest_distance * cp.sqrt(2) / cp.sqrt(3)
        self.radius_inner = self.radius_outer * cp.sqrt(6) / 3
        self.delta_h = self.largest_distance / cp.sqrt(3) / 10.5

    def in_the_panel(self, plate = 'inner'):
    # 判断每个数据点是否在探针盘内，如果在，就记为1，否则记为0
        parallel_distance = panel(self.data, self.redshift).parallel_distance() # float32
        parallel_distance = cp.abs(parallel_distance)
        # shape => (72, 18, 21, -1) 这是所有数据点到探针中心的平行距离，注意取绝对值
        distance_to_axis = Distance(self.data).distance_to_axis()
        distance_to_axis = cp.abs(distance_to_axis)
        # shape => (72, 18, -1) 这是所有数据点到所选的方向矢轴的距离，取绝对值(保险起见)
        distance_to_axis = cp.reshape(distance_to_axis, (72, 18, 1, -1))
        # 要对上述两个矢量进行运算，需要让它们维度一致，因此做一个简单的reshape
        distance_to_axis = cp.repeat(distance_to_axis, 21, axis = 2)
        # shape => (72, 18, 21, -1)

        # 现在，我们创造判据(inner指的是这是第一个探针组)
        if plate == 'inner':
            condition = (distance_to_axis < self.radius_outer) & (parallel_distance < 1/2 * self.delta_h)
            # shape => (72, 18, 21, -1) 这是一个布尔类型的数组，表示数据点是否在探针盘内
            del parallel_distance, distance_to_axis
            cp.get_default_memory_pool().free_all_blocks()
            # 释放内存
            return condition
        
        elif plate == 'outer': # 这是更外部的探针
            condition = (self.radius_inner < distance_to_axis) & (distance_to_axis < self.radius_outer) & (parallel_distance < 1/2 * self.delta_h)
            # shape => (72, 18, 21, -1) 这也是一个布尔类型的数组，表示数据点是否在探针盘内
            del parallel_distance, distance_to_axis
            cp.get_default_memory_pool().free_all_blocks()
            # 释放内存
            return condition
        
        else:
            print('wrong parameter: \'plate\'')
    
class count():
    def __init__(self, data, redshift):
        lightspeed = 3e5
        H_0 = 70
        self.redshift = redshift
        self.data = data
        self.largest_distance = self.redshift * lightspeed / H_0
        self.delta_h = self.largest_distance / cp.sqrt(3) / 10.5
        self.radius_outer = self.largest_distance * cp.sqrt(2) / cp.sqrt(3)
        self.radius_inner = self.radius_outer * cp.sqrt(6) / 3
    
    def count(self, plate = 'inner'):
        if plate == 'inner':
            condition = judge.in_the_panel(self)
            # 导入条件
            condition_new = cp.where(condition, 1, 0)
            # 现在，我们获得了condition，框架为(72, 18, 21, -1)，表示数据点是否在相应的探针内

            # 统计每个探针盘内的数据点个数，计算出每个探针盘的数据点数量
            count = cp.sum(condition_new, axis = -1)
            # shape => (72, 18, 21) 这是每个探针盘内的数据点个数
            del condition
            cp.get_default_memory_pool().free_all_blocks()
            # 释放内存
            return count
        
        elif plate == 'outer':
            condition = judge.in_the_panel(self, plate = 'outer')
            # 导入条件
            condition_new = cp.where(condition, 1, 0)
            count = cp.sum(condition_new, axis = -1) 
            # shape => (72, 18, 21) dtype 是 int
            del condition
            cp.get_default_memory_pool().free_all_blocks()
            # 释放内存
            return count
        
        else:
            print('wrong parameter: \'plate\'')