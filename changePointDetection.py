import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom
from scipy.stats import poisson
import pymc3 as pm
from utils import *


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

list_of_distances = []
for point_idx in range(len(sorted_left_points) - 1):
    distance = calculate_distance_between_two_points(sorted_left_points[point_idx], sorted_left_points[point_idx + 1],
                                                       'x')
    list_of_distances.append(distance)

x_s = np.arange(len(list_of_distances))

plt.plot(x_s, list_of_distances, 'o', markersize=8)
plt.ylabel("distances")
plt.xlabel("index")
# plt.show()

with pm.Model() as distance_model:

    switchpoint = pm.DiscreteUniform('switchpoint', lower=x_s.min(), upper=x_s.max(), testval=1900)

    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(switchpoint >= x_s, early_rate, late_rate)

    disasters = pm.Poisson('disasters', rate, observed=list_of_distances)

with distance_model:
    trace = pm.sample(10000)

    # pm.traceplot(trace)


