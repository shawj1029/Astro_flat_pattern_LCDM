import cupy as cp

class Directions():
    def __init__(self,data):
        self.data = data

    def initial_vector(self):
    # 生成角度数组
        i_angles = cp.linspace(0, 2 * cp.pi, 72, endpoint = False) 
        # -> （0，2pi）的n1等分，表示每次转动5度
        j_angles = cp.linspace(2 * cp.pi / 90, cp.pi / 2, 18, endpoint = False) 
        # -> （0，pi/2）的n2(取18会好一些)等分，剔除了0度：如果方向矢量在j方向包括0度，那么将存在多个相同的方向向量，这是不合理的

        # 生成所有方向的网格
        I, J = cp.meshgrid(i_angles, j_angles, indexing='ij')
        # 计算方向向量
        direction_vectors = cp.stack([
            cp.cos(I) * cp.sin(J),
            cp.sin(I) * cp.sin(J),
            cp.cos(J)
        ], axis=-1, dtype = cp.float32) # shape => (n1, n2, 3), n1,n2是方向向量的总数，3是每一个方向向量的x,y,z坐标

        # 释放内存
        del I, J, i_angles, j_angles
        cp.get_default_memory_pool().free_all_blocks()

        return direction_vectors
    
    def theta(self):
        direction_vectors = self.initial_vector()
        # --> 接收这个表
        data = cp.reshape(self.data, (1,1,-1,3))
        direction_vectors = cp.reshape(direction_vectors, (72,18,1,3))
        # 为了广播操作，需要先做一些重置
        costheta = cp.sum(data * direction_vectors, axis = -1)
        # shape => (72, 18, -1) 内积
        # self.data.shape => (-1,3)
        distance = cp.sqrt(cp.sum(self.data**2, axis = -1))
        # shape => (-1,) 模长
        distance = cp.reshape(distance, (1,1,-1))
        # shape => (1,1,-1) 为方便广播运算，做一次重置
        theta = cp.arccos(costheta / distance)
        theta = theta.astype(cp.float32) # float32
        # shape => (72, 18, -1) 第三个维度表示有多个点，而前两个维度则表示这个点在不同的方向下与单位矢所成夹角
        del direction_vectors, costheta, distance, data
        cp.get_default_memory_pool().free_all_blocks()
        # 释放内存
        return theta

class Distance():
    def __init__(self, data):
        self.data = data

    def distance_to_axis(self):
        distance_absolute = cp.sqrt(cp.sum(self.data**2, axis = -1))
        # shape => (-1,) 模长
        distance_absolute = cp.reshape(distance_absolute, (1,1,-1))
        # shape => (1,1,-1) 为方便广播运算，做一次重置
        theta = Directions(self.data).theta()
        # shape => (72, 18, -1) 角度表
        distance_to_axis = distance_absolute * cp.sin(theta)
        distance_to_axis = distance_to_axis.astype(cp.float32) # float32
        # shape => (72, 18, -1) 距离表
        del distance_absolute, theta
        cp.get_default_memory_pool().free_all_blocks()
        # 释放内存
        return distance_to_axis