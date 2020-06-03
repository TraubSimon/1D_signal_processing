from utils import *

DISTANCE_THRESHOLD = 10
DEFAULT_VARIANCE = 1
# chose distance calculation 'euclidian' or 'x_distance'
DISTANCE_MODE = 'euclidian'
list_of_multipliers = [1./4., 1./3., 1./2., 1, 2, 3, 4]
list_of_weights = [1./5., 1./5., 1./5., 1., -2., -2., -2.]
save_location = '/home/simon/Documents/plots/paramter_tuning_bewertKrit/'

ONLY_FOLLOWING_POINTS = True

PLOT_RESULT = False
PLOT_BEWERTUNGSKRITERIUM = True
SAVE_FIGURES = False
SHOW_PLOTS = False

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
        distance_mode=DISTANCE_MODE, multipliers=list_of_multipliers, weights=list_of_weights)
    distance_list_left, raw_distances_left = list_distances_for_histogram_for_bewertungskriterium(
        selected_points=left_points, only_use_following_points=ONLY_FOLLOWING_POINTS, distance_thr=DISTANCE_THRESHOLD,
        distance_mode=DISTANCE_MODE, multipliers=list_of_multipliers, weights=list_of_weights)

    # plot bewertungskriterium
    if PLOT_BEWERTUNGSKRITERIUM:
        filtered_right = ndimage.gaussian_filter(distance_list_right, 20)
        filtered_left = ndimage.gaussian_filter(distance_list_left, 20)

        # plot_bewertungskriterium(distance_list_right, 'right')
        # plot_bewertungskriterium(distance_list_left, 'left')

        plot_bewertungskriterium_result(all_pts=transformed_points, left_pts=left_points, right_pts=right_points,
                                        filtered_left=filtered_left, filtered_right=filtered_right)
        if SAVE_FIGURES:
            fig_name = '/{}/bewertungskrit_5_{}'.format(i, i)
            plt.savefig(save_location + fig_name)

    if PLOT_RESULT:
        plot_results(all_points=transformed_points, left=left_points, right=right_points,
                     left_distances=raw_distances_left, right_distances=raw_distances_right)

if SHOW_PLOTS:
    plt.show()
