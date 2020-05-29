from utils import *
import statsmodels.api as sm
import numpy as np
from scipy import signal

VARIANCE = 0.01
PLOT_GAUSSIAN_DISTRIBUTION = False
PLOT_SM_GRAPHICS_PartialACF = False
PLOT_SM_GRAPHICS_ACF = False

# sorted and only one side list of values
list_of_values = [1, 5]


list_of_points = []
for value_idx, value in enumerate(list_of_values):
    point = Point(value, 0, 0, value_idx, None, None)
    list_of_points.append(point)

# calculate gaussian_distribution
gaussian_distribution = np.zeros(20)
x_values = np.arange(-10, 10)
for point in list_of_points:
    for x in x_values:
        gaussian_distribution[x + 10] += gaussian(x, point.x, VARIANCE)


# plot raw distribution
if PLOT_GAUSSIAN_DISTRIBUTION:
    plt.plot(x_values, gaussian_distribution)
    plt.show()


# acf plot with sm graphics
if PLOT_SM_GRAPHICS_ACF:
    sm.graphics.tsa.plot_acf(x=gaussian_distribution, title='Autocorrelation')
    plt.show()

# pacf plot with sm graphics
if PLOT_SM_GRAPHICS_PartialACF:
    sm.graphics.tsa.plot_pacf(x=gaussian_distribution, title='Partial Autocorrelation')
    plt.show()

correlation = signal.correlate(gaussian_distribution, gaussian_distribution, 'full')
print('correlation {}'.format(correlation))
plt.plot(correlation)
plt.show()



