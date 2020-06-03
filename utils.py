import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy import ndimage


class GaussianDistribution:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def plot_point(self, color, linestyle, linewidth):
        x_values = np.linspace(-120, 20, 1000)
        plt.plot(x_values, gaussian(x_values, mu=self.mean, sig=self.variance), color=color, ls=linestyle,
                 lw=linewidth)


class Point:
    def __init__(self, x, y, z, idx, nearest_neighbour, distance_to_nearest_neighbour):
        self.x = x
        self.y = y
        self.z = z
        self.idx = idx
        self.nearest_neighbour = nearest_neighbour
        self.distance_to_nearest_neighbour = distance_to_nearest_neighbour
        self.neighbours = []

    def print_point(self):
        print("Point ", str(self.idx), " with neighbour ", str(self.nearest_neighbour), " and distance ",
              str(self.distance_to_nearest_neighbour), " at x=", str(self.x), " y=", str(self.y),
              " and z=", str(self.z))
        if len(self.neighbours) != 0:
            for neighbour_idx, neighbour in enumerate(self.neighbours):
                print(str(neighbour_idx + 1), ".Neighbour: ", neighbour.idx)


def calculate_distance_between_two_points(point1, point2, distance_mode):
    if distance_mode == 'euclidian':
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)
    elif distance_mode == 'x_distance':
        return np.abs(point1.x - point2.x)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def add_gaussians(gauss_1, gauss_2):
    new_mean = (gauss_1.variance**2 * gauss_2.mean + gauss_2.variance**2 * gauss_1.mean) / \
               (gauss_1.variance**2 + gauss_2.variance**2)
    new_variance = 1 / (1 / gauss_1.variance**2 + 1 / gauss_2.variance**2)
    return GaussianDistribution(new_mean, new_variance)


def convert_point_x_list_to_gaussian(point_list, default_variance):
    list_of_gaussian_points = []
    for point in point_list:
        new_gaussian_point = GaussianDistribution(mean=point.x, variance=default_variance)
        list_of_gaussian_points.append(new_gaussian_point)
    return list_of_gaussian_points


def sort_list_of_points_along_x_axis(unsorted_list_of_points):
    list_of_points = np.zeros((len(unsorted_list_of_points)))
    for pointindex, point in enumerate(unsorted_list_of_points):
        list_of_points[pointindex] = point.x

    sorted_array = np.sort(list_of_points)
    sorted_list_of_points = []
    for x_value in sorted_array:
        for point in unsorted_list_of_points:
            if x_value == point.x:
                sorted_list_of_points.append(point)
                # unsorted_list_of_points.remove(point)

    return sorted_list_of_points


def get_gaussian_distribution(point_list, x_values,  variance):
    gaussian_distribution = np.zeros(x_values.size)
    for point in point_list:
        for x in x_values:
            gaussian_distribution[int(x - x_values[0])] += gaussian(x, point.x * 100, variance)

    return gaussian_distribution


def move_gaussian_distribution(distribution, delta):
    for value in distribution:
        value += delta
    return distribution


def print_list_of_points(list_of_points):
    for point in list_of_points:
        point.print_point()


def update_point_list_with_neighbours(point_list, only_use_following_points, distance_mode):
    for new_point in point_list:
        for neighbour_point in point_list:
            if new_point.idx == neighbour_point.idx:
                continue

            if only_use_following_points and new_point.x > neighbour_point.x:
                continue

            distance_between_new_and_neighbour = calculate_distance_between_two_points(new_point, neighbour_point,
                                                                                       distance_mode)

            if new_point.distance_to_nearest_neighbour is None:
                new_point.distance_to_nearest_neighbour = distance_between_new_and_neighbour
                new_point.nearest_neighbour = neighbour_point.idx
            else:
                if distance_between_new_and_neighbour < new_point.distance_to_nearest_neighbour:
                    new_point.distance_to_nearest_neighbour = distance_between_new_and_neighbour
                    new_point.nearest_neighbour = neighbour_point.idx


def transform_to_point_class_list(point_list_as_array):
    point_list_as_class_points = []
    for selected_point_idx, selected_point in enumerate(point_list_as_array):
        point = Point(x=selected_point[0], y=selected_point[1], z=selected_point[2], idx=selected_point_idx,
                      nearest_neighbour=None, distance_to_nearest_neighbour=None)
        point_list_as_class_points.append(point)
    return point_list_as_class_points


def get_boundaries_of_points(points):
    left_boarder = points[0].x
    right_boarder = points[-1].x

    return left_boarder, right_boarder


def plot_list_of_points(right_points, left_points, with_numbers_as_text, style):

    def execute_plot(point_list):
        if with_numbers_as_text:
            for point in point_list:
                plt.plot(point.x, point.y, style)
                plt.text(point.x, point.y, str(point.idx))

        else:
            for point in point_list:
                plt.plot(point.x, point.y, style)

    plt.figure('Points with index numbers')
    plt.subplot(121)
    execute_plot(left_points)
    plt.title('Left')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(122)
    execute_plot(right_points)
    plt.title('Right')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_points(points, style):
    for point in points:
        plt.plot(point.x, point.y, style)


def plot_bewertungskriterium_result(all_pts, left_pts, right_pts, filtered_left, filtered_right):
    fig, gs = plot_divided_sides(all_pts=all_pts, left_pts=left_pts, right_pts=right_pts)

    left_plot_autocorr = fig.add_subplot(gs[1, 0])
    left_plot_autocorr.set_xlabel('distance')
    plot_bewertungskriterium(filtered_left[:1000], 'left')
    left_value_to_take = np.argmax(filtered_left[:1000])
    plt.title('left: {}'.format(left_value_to_take))

    right_plot_autocorr = fig.add_subplot(gs[1, 1])
    right_plot_autocorr.set_xlabel('distance')
    plot_bewertungskriterium(filtered_right[:1000], 'right')
    right_value_to_take = np.argmax(filtered_right[:1000])
    plt.title('right: {}'.format(right_value_to_take))




def plot_list_of_gaussian_points(list_of_points, color, linestyle, linewidth):
    for gaussian_point in list_of_points:
        gaussian_point.plot_point(color, linestyle, linewidth)


def get_near_points(point, point_list, distance_threshold, only_use_following_points, distance_mode):
    for point_list_element in point_list:
        distance = calculate_distance_between_two_points(point, point_list_element, distance_mode)
        # don't add self
        if distance == 0:
            continue
        # only add points in front
        if only_use_following_points and point.x < point_list_element.x:
            continue
        if distance < distance_threshold:
            point.neighbours.append(point_list_element)


def get_points_from_one_rail(all_points, side):
    if side == 'left':
        mask = all_points[:, 1] < 0
    elif side == 'right':
        mask = all_points[:, 1] > 0
    else:
        mask = 0

    masked_points = all_points[mask]
    return transform_to_point_class_list(masked_points)


def list_distances_for_histogram_for_bewertungskriterium(selected_points, only_use_following_points, distance_thr,
                                                         distance_mode, multipliers, weights):
    list_of_raw_distances = []
    list_of_distances = np.zeros(5000)
    for selected_point in selected_points:
        get_near_points(point=selected_point, point_list=selected_points, distance_threshold=distance_thr,
                        only_use_following_points=only_use_following_points, distance_mode=distance_mode)
        if len(selected_point.neighbours) != 0:
            for neighbour in selected_point.neighbours:
                distance = calculate_distance_between_two_points(selected_point, neighbour, distance_mode)
                list_of_raw_distances.append(distance)

                for multiplier, weight in zip(multipliers, weights):
                    list_of_distances[int(np.around(distance * 100 * multiplier))] += weight
    return list_of_distances, list_of_raw_distances


def plot_statsmodel_signal_correlation(all_points, left, right, left_corr, right_corr, normalized=False):
    fig, gs = plot_divided_sides(all_pts=all_points, left_pts=left, right_pts=right)

    left_plot_autocorr = fig.add_subplot(gs[1, 0])
    left_plot_autocorr.set_xlabel('distance')
    plt.plot(left_corr[:1000], color='r')

    left_value_to_take = np.argmax(left_corr[20:1000]) + 20
    if normalized:
        left_peak_threshold_value = get_peak_value(left_corr)
        plt.title('left: argmax: {} and Peak_threshold: {}'.format(left_value_to_take, left_peak_threshold_value))
    else:
        plt.title('left: argmax: {}'.format(left_value_to_take))



    right_plot_autocorr = fig.add_subplot(gs[1, 1])
    right_plot_autocorr.set_xlabel('distance')
    plt.plot(right_corr[:1000], color='g')

    right_value_to_take = np.argmax(right_corr[20:1000]) + 20
    if normalized:
        right_peak_threshold_value = get_peak_value(right_corr)
        plt.title('right: argmax: {} and Peak_threshold: {}'.format(right_value_to_take, right_peak_threshold_value))
    else:
        plt.title('left: argmax: {}'.format(left_value_to_take))


def get_peak_value(values):
    # delete first peak for autocorrelation with delta = 0
    for value_idx, value in enumerate(values):
        if value < 0.01:
            values = values[value_idx:1000]
            break
    peaks = values[values < 0.24]
    idxs = np.arange(len(values))[values > 0.24] + value_idx
    results = []
    sequence = []
    for i in range(len(idxs) - 1):
        if idxs[i + 1] - idxs[i] == 1:
            sequence.append(idxs[i])
        else:
            results.append(int(np.mean(sequence)))
            sequence = []
    try:
        results.append(int(np.mean(sequence)))
    except ValueError:
        pass

    return results



def plot_bewertungskriterium(distance_list, side):
    final_distance_max_method = np.argmax(distance_list)

    if side == 'left':
        color = 'r'
    else:
        color = 'g'

    # fig = plt.figure(side + ' points')
    plt.plot(distance_list, color=color)
    plt.title('final_distance_max_method: ' + str(final_distance_max_method) + 'cm')
    plt.xlabel('distance [cm]')
    plt.ylabel('weight')


def convert_gaussian_to_array(gs):
    array = np.zeros(shape=len(gs))
    for point_idx, point in enumerate(gs):
        array[point_idx] = point.mean
    return array


def get_list_and_array_from_gaussian_with_movement(gs, movement=None):
    moved_point_list = []
    moved_point_array = np.zeros(len(gs))
    for gaussian_point_idx, gaussian_point in enumerate(gs):
        if movement is None:
            moved_point_list.append(
                GaussianDistribution(mean=gaussian_point.mean, variance=gaussian_point.variance))
            moved_point_array[gaussian_point_idx] = gaussian_point.mean
        else:
            moved_point_list.append(
                GaussianDistribution(mean=gaussian_point.mean + movement, variance=gaussian_point.variance))
            moved_point_array[gaussian_point_idx] = gaussian_point.mean + movement

    return moved_point_list, moved_point_array


def add_two_gaussian_lists(list_1, list_2):
    added_point_list = []
    added_point_array = np.zeros(len(list_1))
    for added_point_idx, (point_1, point_2) in enumerate(zip(list_1, list_2)):
        added_point_list.append(add_gaussians(point_1, point_2))
        added_point_array[added_point_idx] = add_gaussians(point_1, point_2).mean
    return added_point_list, added_point_array


def plot_xcorr(mean_right, mean_left, moved_right, moved_left, delta=0):
    plt.figure('plt.xcorr with delta {}'.format(delta))
    plt.subplot(121)
    plt.xcorr(mean_left, moved_left)
    plt.title('Left')
    plt.xlabel('lags')
    plt.ylabel('partial autocorrelation')
    plt.subplot(122)
    plt.xcorr(mean_right, moved_right)
    plt.title('Right')
    plt.xlabel('lags')
    plt.ylabel('partial autocorrelation')

    plt.show()


def plot_moved_and_added_point_lists(gaussian_left, moved_left, added_left, gaussian_right, moved_right, added_right,
                                     delta, color=0):
    plt.figure('autocorrelation with delta {}'.format(delta))

    ax1 = plt.subplot(321)
    plot_list_of_gaussian_points(gaussian_left, color='g', linestyle='-', linewidth=1.)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Left')

    plt.subplot(323, sharex=ax1)
    plot_list_of_gaussian_points(moved_left, color='r', linestyle=':', linewidth=1.)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(325, sharex=ax1)
    plot_list_of_gaussian_points(gaussian_left, color='g', linestyle='--', linewidth=.5)
    plot_list_of_gaussian_points(moved_left, color='r', linestyle=':', linewidth=.5)
    plot_list_of_gaussian_points(added_left, color=plt.cm.yjet(color), linestyle='-', linewidth=1.)
    plt.xlabel('x')
    plt.ylabel('y')

    ax3 = plt.subplot(322)
    plot_list_of_gaussian_points(gaussian_right, color='g', linestyle='-', linewidth=1.)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Right')

    plt.subplot(324, sharex=ax3)
    plot_list_of_gaussian_points(moved_right, color='r', linestyle=':', linewidth=1.)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(326, sharex=ax3)
    plot_list_of_gaussian_points(gaussian_right, color='g', linestyle='--', linewidth=.5)
    plot_list_of_gaussian_points(moved_right, color='r', linestyle=':', linewidth=.5)
    plot_list_of_gaussian_points(added_right, color=plt.cm.jet(color), linestyle='-', linewidth=1.)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_divided_sides(all_pts, left_pts, right_pts):
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    point_cloud = fig.add_subplot(gs[0, :])
    plot_points(left_pts, 'rx')
    plot_points(right_pts, 'gx')
    plot_points(all_pts, 'b.')

    plt.title('point_cloud')
    point_cloud.set_xlabel('x')
    point_cloud.set_ylabel('y')

    return fig, gs


def plot_results(all_points, left, right, left_distances, right_distances):
    fig, gs = plot_divided_sides(all_pts=all_points, left_pts=left, right_pts=right)

    left_histogram = fig.add_subplot(gs[1, 0])
    left_histogram.hist(left_distances, color='r')
    plt.title('left histogram')
    left_histogram.set_xlabel('distance')
    left_histogram.set_ylabel('num of values')

    right_histogram = fig.add_subplot(gs[1, 1])
    right_histogram.hist(right_distances, color='g')
    plt.title('right histogram')
    right_histogram.set_xlabel('distance')
    right_histogram.set_ylabel('num of values')

    plt.show()


def process_file(filename_):
    with open(filename_) as f:
        lines = f.readlines()
    points_ = np.zeros((len(lines), 3), dtype=np.float)

    for line_idx, line in enumerate(lines):
        line_elements = line.split(" ")
        if len(line_elements) == 4:
            points_[line_idx] = [float(line_elements[0]), float(line_elements[1]), float(line_elements[2])]
    return points_


def autocorr(gaussian_distribution):
    correlation = signal.correlate(gaussian_distribution, gaussian_distribution, 'full')
    # print('correlation {}'.format(correlation))
    return correlation[int(correlation.size / 2):]
