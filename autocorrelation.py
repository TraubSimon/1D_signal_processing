from utils import *
import statsmodels.api as sm
from scipy import signal

# define parameters
DISTANCE_THRESHOLD = 10
VARIANCE = 100
NUM_OF_DELTAS = 10

ONLY_FOLLOWING_POINTS = True

PLOT_POINTS_WITH_NUMBERS = False
PLOT_SM_GRAPHICS_ACF = False
PLOT_SM_GRAPHICS_PartialACF = False
PLOT_XCORR = False

# process data
# for i in range(11):
# filename = '../VehiclePoints/example' + str(i) + '.txt'
filename = '../VehiclePoints/example' + str(3) + '.txt'
# read point coordinates from file
points = process_file(filename)

# divide left & right points and transform to point class
left_points = get_points_from_one_rail(points, 'left')
right_points = get_points_from_one_rail(points, 'right')
transformed_points = transform_to_point_class_list(points)

# sort points according to the x-axis
sorted_right_points = sort_list_of_points_along_x_axis(right_points)
sorted_left_points = sort_list_of_points_along_x_axis(left_points)
sorted_points = sort_list_of_points_along_x_axis(transformed_points)

if PLOT_POINTS_WITH_NUMBERS:
    plot_list_of_points(sorted_right_points, sorted_left_points, True, 'rx')

# calculate gaussian_distribution
right_gaussian_distribution = get_gaussian_distribution(sorted_right_points, VARIANCE)
left_gaussian_distribution = get_gaussian_distribution(sorted_left_points, VARIANCE)

# acf plot with sm graphics
if PLOT_SM_GRAPHICS_ACF:
    sm.graphics.tsa.plot_acf(x=left_gaussian_distribution, title='Autocorrelation Left')
    sm.graphics.tsa.plot_acf(x=right_gaussian_distribution, title='Autocorrelation Right')
    plt.show()
# pacf plot with sm graphics
if PLOT_SM_GRAPHICS_PartialACF:
    sm.graphics.tsa.plot_pacf(x=left_gaussian_distribution, title='Partial Autocorrelation Left')
    sm.graphics.tsa.plot_pacf(x=right_gaussian_distribution, title='Partial Autocorrelation Right')
    plt.show()

# apply delta handy and plot with plt.xcorr
deltas = np.linspace(0, 9, NUM_OF_DELTAS)
colors = np.linspace(0, 1, NUM_OF_DELTAS)
for delta, color in zip(deltas, colors):
    left_moved_gaussian_distribution = move_gaussian_distribution(left_gaussian_distribution, delta)
    right_moved_gaussian_distribution = move_gaussian_distribution(right_gaussian_distribution, delta)

    if PLOT_XCORR:
        plot_xcorr(right_gaussian_distribution, left_gaussian_distribution, right_moved_gaussian_distribution, left_moved_gaussian_distribution, delta)

