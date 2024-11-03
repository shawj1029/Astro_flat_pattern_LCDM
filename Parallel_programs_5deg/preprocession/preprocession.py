import cupy as cp
import numpy as np
import time

class select(): 
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def grab(self):
        z_0 = np.load('/home/jiangxiaoqi/Data/Original data/z=0.npy')
        z_002 = np.load('/home/jiangxiaoqi/Data/Original data/z=0.02.npy')
        z_005 = np.load('/home/jiangxiaoqi/Data/Original data/z=0.05.npy')
        z_009 = np.load('/home/jiangxiaoqi/Data/Original data/z=0.09.npy')
        z_014 = np.load('/home/jiangxiaoqi/Data/Original data/z=0.14.npy')
        z_019 = np.load('/home/jiangxiaoqi/Data/Original data/z=0.19.npy')

        return z_0, z_002, z_005, z_009, z_014, z_019

    def adapt(self):
        z_0, z_002, z_005, z_009, z_014, z_019 = self.grab()
        z_0, z_002, z_005, z_009, z_014, z_019 = cp.array(z_0), cp.array(z_002), cp.array(z_005), cp.array(z_009), cp.array(z_014), cp.array(z_019)
        # cupy is faster

        arrays = [z_0, z_002, z_005, z_009, z_014, z_019]
        for array in arrays:
            array[:, 0] -= self.x
            array[:, 1] -= self.y
            array[:, 2] -= self.z # adapt the datapoints to the new position

        for array in arrays:
            mask_too_large = array > 1000
            mask_too_small = array < -1000
            array[mask_too_large] -= 2000
            array[mask_too_small] += 2000 # adopt the periodic boundary condition

        return z_0, z_002, z_005, z_009, z_014, z_019

    def merge(self):
        z_0, z_002, z_005, z_009, z_014, z_019 = self.adapt()

        arrays = [z_0, z_002, z_005, z_009, z_014, z_019]
        masks = [[0,44.5],[44.5,156.853],[156.853,312.853],[312.853,512.808],[512.808,732.692],[732.692,2000]]
        candidate = cp.empty((0,3)) # to contain data

        for array,mask in zip(arrays,masks):
            distance = cp.sqrt(cp.sum(array**2, axis=1))
            mask_condition = (distance >= mask[0]) & (distance < mask[1]) # select the datapoints within the redshift requirements
            candidate = cp.vstack((candidate, array[mask_condition]))

        candidate = cp.asnumpy(candidate)
        return candidate

    def save(self):
        np.save(f'/home/jiangxiaoqi/Data/Other coordinates/[{self.x},{self.y},{self.z}].npy', self.merge())


if __name__ == '__main__':
    t=time.time()
    z = select(1, 2, 3).merge()
    print(time.time()-t) # probably 9.3 seconds for each washing
    print(z.shape)
    