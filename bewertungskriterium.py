from utils import *

DISTANCE_THRESHOLD = 10
DEFAULT_VARIANCE = 1
# chose distance calculation 'euclidian' or 'x_distance'
DISTANCE_MODE = 'euclidian'

ONLY_FOLLOWING_POINTS = True

PLOT_RESULT = False
PLOT_BEWERTUNGSKRITERIUM = True

# iterate over all files
for i in range(11):
    # filename = './VehiclePoints/example' + str(3) + '.txt'
    filename = './VehiclePoints/example' + str(i) + '.txt'
    # read point coordinates from file
    points = process_file(filename)

    # divide left & right points and transform to point class
    left_points = get_points_from_one_rail(points, 'left')
    right_points = get_points_from_one_rail(points, 'right')
    transformed_points = transform_to_point_class_list(points)

    # calculate neighbours and create histogram of distances between points
    distance_list_right, raw_distances_right = list_distances_for_histogram_for_bewertungskriterium(
        selected_points=right_points, only_use_following_points=ONLY_FOLLOWING_POINTS, distance_thr=DISTANCE_THRESHOLD,
        distance_mode=DISTANCE_MODE, default_variance=DEFAULT_VARIANCE)
    distance_list_left, raw_distances_left = list_distances_for_histogram_for_bewertungskriterium(
        selected_points=left_points, only_use_following_points=ONLY_FOLLOWING_POINTS, distance_thr=DISTANCE_THRESHOLD,
        distance_mode=DISTANCE_MODE, default_variance=DEFAULT_VARIANCE)

    # plot bewertungskriterium
    if PLOT_BEWERTUNGSKRITERIUM:
        filtered_right = ndimage.gaussian_filter(distance_list_right, 20)
        filtered_left = ndimage.gaussian_filter(distance_list_left, 20)

        # plot_bewertungskriterium(distance_list_right, 'right')
        # plot_bewertungskriterium(distance_list_left, 'left')



        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        point_cloud = fig.add_subplot(gs[0, :])
        plot_points(left_points, 'rx')
        plot_points(right_points, 'gx')
        plot_points(transformed_points, 'b.')

        plt.title('point_cloud')
        point_cloud.set_xlabel('x')
        point_cloud.set_ylabel('y')

        left_plot_autocorr = fig.add_subplot(gs[1, 0])
        left_plot_autocorr.set_xlabel('distance')
        plot_bewertungskriterium(filtered_left[0:1000], 'left')
        plt.title('left')

        right_plot_autocorr = fig.add_subplot(gs[1, 1])
        right_plot_autocorr.set_xlabel('distance')
        plot_bewertungskriterium(filtered_right[0:1000], 'right')
        plt.title('right')



    if PLOT_RESULT:
        plot_results(all_points=transformed_points, left=left_points, right=right_points,
                     left_distances=raw_distances_left, right_distances=raw_distances_right)



plt.show()