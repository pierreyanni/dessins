import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from skimage.transform import rotate

class AugmentData:
    def __init__(self, X, y, classes):
        self.X = X
        self.y = y
        self.classes = classes
        self.side = 28
    
    def plot_image(self, image):       
        print(self.classes[self.y[image]])
        plt.imshow(self.X[image].reshape(self.side, self.side), cmap='gray_r',
               interpolation='nearest')
        plt.show()
    
    def shift_image(self, image, dx, dy):
        image = image.reshape((self.side, self.side))
        shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
        return shifted_image.reshape([-1])
    
    def add_shifted_images(self, deltas=[1]):
        l_X = self.X.tolist()
        l_y = self.y.tolist()
        for delta in deltas:
            for dx, dy in ((delta, 0), (-delta, 0), (0, delta), (0, -delta)):
                for image in self.X:
                    l_X.append(self.shift_image(image, dx, dy))
                l_y += self.y.tolist()
        
        self.X = np.array(l_X)
        self.y = np.array(l_y)
        
    def add_flipped_images(self):
        flipped = [np.fliplr(image.reshape(self.side, self.side)).flatten()
                   for image in self.X]
        self.X = np.array(flipped + self.X.tolist())
        self.y = np.tile(self.y, 2)
        
    def add_rotated_images(self, angles=[-10, 10]):
        l_X = self.X.tolist()
        l_y = self.y.tolist()
        for angle in angles:
            rotated = [rotate(image.reshape(self.side, self.side), angle).flatten() 
                       for image in self.X]
            l_X += rotated
            l_y += self.y.tolist()
        self.X = np.array(l_X)
        self.y = np.array(l_y)
        
    def return_data(self):
        return self.X, self.y