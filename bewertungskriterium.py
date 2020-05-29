from utils import *

DISTANCE_THRESHOLD = 10
DEFAULT_VARIANCE = 1

ONLY_FOLLOWING_POINTS = True

PLOT_RESULT = False
PLOT_BEWERTUNGSKRITERIUM = True

# iterate over all files
for i in range(11):
    # filename = '../VehiclePoints/example' + str(3) + '.txt'
    filename = '../VehiclePoints/example' + str(i) + '.txt'
    # read point coordinates from file
    points = process_file(filename)

    # divide left & right points and transform to point class
    left_points = get_points_from_one_rail(points, 'left')
    right_points = get_points_from_one_rail(points, 'right')
    transformed_points = transform_to_point_class_list(points)

    # calculate neighbours and create histogram of distances between points
    distance_list_right, raw_distances_right = list_distances_for_histogram_for_bewertungskriterium(
        selected_points=right_points, only_use_following_points=ONLY_FOLLOWING_POINTS, distance_thr=DISTANCE_THRESHOLD,
        default_variance=DEFAULT_VARIANCE)
    distance_list_left, raw_distances_left = list_distances_for_histogram_for_bewertungskriterium(
        selected_points=left_points, only_use_following_points=ONLY_FOLLOWING_POINTS, distance_thr=DISTANCE_THRESHOLD,
        default_variance=DEFAULT_VARIANCE)

    # plot bewertungskriterium
    if PLOT_BEWERTUNGSKRITERIUM:
        plot_bewertungskriterium(distance_list_right, 'right')
        plot_bewertungskriterium(distance_list_left, 'left')

    if PLOT_RESULT:
        plot_results(all_points=transformed_points, left=left_points, right=right_points,
                     left_distances=raw_distances_left, right_distances=raw_distances_right)



