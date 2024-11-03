import cupy as cp
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

class statistical():
    def __init__(self, data, redshift): # 这里data是counting_program的count结果
        H_0 = 70
        lightspeed = 3e5
        self.redshift = redshift
        self.data = data
        self.largest_distance = self.redshift * lightspeed / H_0
        self.delta_h = self.largest_distance / cp.sqrt(3) / 10.5

    def KDE(self):
        # 这个函数用于生成核密度估计
        # 设置带宽和核函数的候选值
        params = {'bandwidth': cp.logspace(-1, 1, 50),
                'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']}

        # 初始化并运行网格搜索
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(self.data)

        # 得到最佳的核密度估计
        kernel = grid.best_params_['kernel']
        bandwidth = grid.best_params_['bandwidth']

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde.fit(self.data)
        return kde, kernel, bandwidth
    
    def KDE_draw_data(self):
        # 给出核密度估计，用于后续作图
        kde, kernel, bandwidth = self.KDE()
        probe = cp.linspace(-10.5 * self.delta_h, 10.5 * self.delta_h, 1000)
        probe = cp.reshape(probe, (-1, 1))
        log_dens = kde.score_samples(probe)
        return probe, cp.exp(log_dens)
    
    