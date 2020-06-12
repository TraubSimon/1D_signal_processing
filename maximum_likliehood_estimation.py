import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom
from scipy.stats import poisson
from utils import *

DISTANCE_MODE = 'euclidian'

# process data
# for i in range(11):
    # filename = './VehiclePoints/example' + str(i) + '.txt'
filename = './VehiclePoints/example' + str(3) + '.txt'
# read point coordinates from file
points = process_file(filename)


# divide left & right points and transform to point class
left_points = get_points_from_one_rail(points, 'left')

# sort points according to the x-axis
sorted_left_points = sort_list_of_points_along_x_axis(left_points)

 # add distance to next point
list_of_distances = []
for point_idx in range(len(sorted_left_points) - 1):
    distance = calculate_distance_between_two_points(sorted_left_points[point_idx], sorted_left_points[point_idx + 1],
                                                       'x')
    list_of_distances.append(distance)

# MLE for geometric distribution p=n/sum(x_i)
p = len(list_of_distances) / sum(list_of_distances)
print('MLE for geometric distrbutions: p = {}'.format(p))


# plot list of distances
list_of_indexes = np.arange(len(list_of_distances))
# plt.scatter(list_of_indexes, list_of_distances)
# plt.show()

# calculate mean, variance and p for the geometric distribution
mean = np.mean(list_of_distances)
mu = mean
p = 1 / mean
variance = 1 - p / (p * p)
print('Calculation of p with mean and p = 1 / mean: p = {}'.format(p))

mean, var, skew, kurt =  poisson.stats(mu, moments='mvsk')
print('poisson: mean = {}, var = {}, skew = {}, kurt = {}'.format(mean, var, skew, kurt))

mean, var, skew, kurt = geom.stats(p, moments='mvsk')
print('geom: mean = {}, var = {}, skew = {}, kurt = {}'.format(mean, var, skew, kurt))

# display the probability mass function
x_geom = np.arange(geom.ppf(0.01, p), geom.ppf(0.99, p))
x_poisson = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))

plt.subplot(121)
plt.plot(x_poisson, poisson.pmf(x_poisson, mu), 'bo', ms=8, label='poisson pmf')
plt.vlines(x_poisson, 0, poisson.pmf(x_poisson, mu), colors='b', lw=5, alpha=0.5)
plt.title('poisson')

plt.subplot(122)
plt.plot(x_geom, geom.pmf(x_geom, p), 'bo', ms=8, label='poisson pmf')
plt.vlines(x_geom, 0, geom.pmf(x_geom, p), colors='b', lw=5, alpha=0.5)
plt.title('geometric')
plt.show()





