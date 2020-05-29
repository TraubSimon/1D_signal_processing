from utils import *

DISTANCE_THRESHOLD = 10
PLOT_RESULT = True
ONLY_FOLLOWING_POINTS = True


# iterate over all files
for i in range(11):
    filename = '../VehiclePoints/example' + str(i) + '.txt'
    # filename = '../VehiclePoints/example' + str(3) + '.txt'
    # read point coordinates from file
    points = process_file(filename)

    # divide left & right points and transform to point class
    left_points = get_points_from_one_rail(points, 'left')
    right_points = get_points_from_one_rail(points, 'right')
    transformed_points = transform_to_point_class_list(points)


    # calculate neighbours and create histogram of all points general
    distance_list_right = list_distances_for_histogram_in_general(right_points, ONLY_FOLLOWING_POINTS)
    distance_list_left = list_distances_for_histogram_in_general(left_points, ONLY_FOLLOWING_POINTS)


    if PLOT_RESULT:
        plot_results(all_points=transformed_points, left=left_points, right=right_points,
                     left_distances=distance_list_left, right_distances=distance_list_right)








