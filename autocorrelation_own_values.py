from utils import *
import statsmodels.api as sm
import numpy as np
from scipy import signal

VARIANCE = 10
PLOT_GAUSSIAN_DISTRIBUTION = False
PLOT_SM_GRAPHICS_PartialACF = True
PLOT_SM_GRAPHICS_ACF = True
SHOW_CORRELATION_STEPWISE = False
PLOT_STATSMODEL_SIGNAL_CORRELATION = True
X_STEPS = 1000

# sorted and only one side list of values
list_of_values = [1, 3, 5, 7, 9]


list_of_points = []
for value_idx, value in enumerate(list_of_values):
    point = Point(value, 0, 0, value_idx, None, None)
    list_of_points.append(point)

# calculate gaussian_distribution
gaussian_distribution = np.zeros(X_STEPS)
x_values = np.arange(X_STEPS)
for point in list_of_points:
    for x in x_values:
        gaussian_distribution[x] += gaussian(x, point.x * 100, VARIANCE)


if SHOW_CORRELATION_STEPWISE:
    lags = np.arange(start=1, stop=1000, step=1)
    auto_corr = []

    moved_signal = np.zeros(X_STEPS)
    for lag in lags:
        moved_signal[0:lag] = 0
        moved_signal[lag:] = gaussian_distribution[:-lag]

        added_signal = moved_signal * gaussian_distribution
        auto_corr.append(np.sum(added_signal))
        ax1 = plt.subplot(211)
        plt.plot(moved_signal, 'g', label='moved signal')
        plt.plot(gaussian_distribution, 'r', label='original signal')
        plt.legend()

        plt.subplot(212, sharex=ax1, sharey=ax1)
        plt.plot(added_signal, 'b', label='added signal')
        plt.title(str(np.sum(added_signal)))
        plt.legend()
        plt.show()

    plt.plot(auto_corr)
    plt.show()


# plot raw distribution
if PLOT_GAUSSIAN_DISTRIBUTION:
    plt.plot(x_values, gaussian_distribution)
    plt.show()


# acf plot with sm graphics
if PLOT_SM_GRAPHICS_ACF:
    sm.graphics.tsa.plot_acf(x=gaussian_distribution, title='Autocorrelation', lags=4)
    plt.show()

# pacf plot with sm graphics
if PLOT_SM_GRAPHICS_PartialACF:
    sm.graphics.tsa.plot_pacf(x=gaussian_distribution, title='Partial Autocorrelation', lags=4)
    plt.show()

if PLOT_STATSMODEL_SIGNAL_CORRELATION:
    wanted_correlation = autocorr(gaussian_distribution)
    plt.plot(wanted_correlation)
    plt.show()


