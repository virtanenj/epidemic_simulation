import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# Model steructure:
# 
# - Agents: indivuduals with potentialli varying parameters based
# on certain distributions.
# - Population: number of agents.
# - Simulation: population goes through number of time steps carrying
# out some rules that determines the dynamics of the population. The
# modelled disease transmission is modelled by close contact between
# agents taking into account agent's possible parameters regarding 
# transmission.
# 
# The model is (to be) based on (pandas dataframe or numpy 
# multidimensional arrays?).

# Necessary??
class Agent:
    def __init__(self, id, SIR='susceptible', age=None, home=None):
        self.id = id
        # Assume SIR-model (susceptible, infectious or removed*)
        # *I prefer 'removed' rather than 'recovered' as that should 
        # describe both death and actually recovered.
        self.SIR = SIR
        self.age = age
        # Consider some description of individuals community.
        # How could this be generalized such that the model for work
        # home, hobby and so on doesn't have to be hardcoded?
        self.home = home


def init_population(population_size, box=(0,0,10,10)):
    ''' box=(min x-value, min y-value, max x-value, max y-value) '''
    x = np.random.uniform(box[0], box[2], population_size)
    y = np.random.uniform(box[1], box[3], population_size)
    id = np.arange(population_size)
    SIR = ['susceptible' for i in range(population_size)]
    return pd.DataFrame({'id':id, 'x':x, 'y':y, 'SIR':SIR}), box


def run_simulation(init_population, box, time_steps):
    population_size = init_population.shape[0]
    space_array = np.zeros(time_steps*population_size*2).reshape(time_steps,population_size,2)
    space_array[0,:,:] = init_population[['x','y']].to_numpy()
    for t in range(1,time_steps):
        for individual in range(space_array[t,:,:].shape[0]):
            # Polar coordinates: x=r*cos(phi), y=r*sin(phi)
            delta_r = 1
            delta_phi = np.random.uniform(0,2*np.pi)
            delta_x = delta_r * np.cos(delta_phi)
            delta_y = delta_r * np.sin(delta_phi)
            # update xy-coordinates and check that they stay inside given box
            x_new = delta_x + space_array[t-1,individual,0]
            if (x_new < box[0]) or (x_new > box[2]):
                if x_new < box[0]:
                    x_new = box[2] - abs(box[0] - x_new)
                elif x_new > box[2]:
                    x_new = box[0] + abs(x_new - box[2])
            y_new = delta_y + space_array[t-1,individual,1]
            if (y_new < box[1]) or (y_new > box[3]):
                if y_new < box[1]:
                    y_new = box[3] - abs(box[1] - y_new)
                elif y_new > box[3]:
                    y_new = box[1] + abs(y_new - box[3])
            space_array[t,individual,0] = x_new
            space_array[t,individual,1] = y_new
        # Update possible SIR-state
        distance_matrix = squareform(pdist(space_array[t,:,:]))  # diagonally symmetric
        infectious_distance = 2  # Possible variable
        close_contact = np.where((distance_matrix < infectious_distance) & (distance_matrix > 0))
        print(close_contact)
        print(distance_matrix)
    return space_array


population_size = 5
time_steps = 2

df, box = init_population(population_size)

a = run_simulation(df, box, time_steps)


# import matplotlib.pyplot as plt

# for i in range(a.shape[1]):
#     some_dude = a[:,i,:]
#     # if i==0:
#     #     plt.plot(some_dude[:,0], some_dude[:,1], 'red')
#     # else:
#     #     plt.plot(some_dude[:,0], some_dude[:,1], '.', color='gray')
#     plt.plot(some_dude[:,0], some_dude[:,1], '.')
# plt.show()

# b = np.arange(12).reshape(6,2)
# print(b[0,:])
# for i in range(b.shape[0]):
#     xy_i = b[i,:]
#     for j in range(i+1,b.shape[0]):
#         xy_j = b[j,:]
#         d = np.linalg.norm(xy_i - xy_j)
#         print('i='+str(i)+', j='+str(j)+', d='+str(d))

# einsum = np.einsum('ij,ij->', b,b)
# print(einsum)


# condensed_distance_matrix = pdist(b)
# distance_matrix = squareform(pdist(b))
# print(condensed_distance_matrix)
# print(distance_matrix)

# Distance for example between indexes 2 and 5
# print(distance_matrix[2,5])
