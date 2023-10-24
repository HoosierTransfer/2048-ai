import numpy as np
import matplotlib.pyplot as plt

mu=0.0
std = 0.1
def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 
x = np.zeros((4, 4))

x = gaussian_noise(x,mu,std)
print(x)
plt.imshow(x)
plt.show()