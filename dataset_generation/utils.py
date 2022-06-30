import numpy as np

def compute_midpoints(coords):

    min_length = min(len(coords[0]), len(coords[1]))
    max_length = max(len(coords[0]), len(coords[1]))
    # Determine short and long rail (number of points in polygon)
    if len(coords[0]) < len(coords[1]):
        short_side = 0
        long_side = 1
        left_side = "long"
    else:
        short_side = 1
        long_side = 0
        left_side = "short"

    # For each point in the long rail polygon, find the corresponding point on the short rail polygon that
    # has the closest y-offset (such that the two points are connected by an approximately horizontal line)
    # Compute the midpoints and distances between these corresponding points
    short = list()
    long = list()
    distances = list()
    midpoints = list()
    i_short = 0
    for i_long in range(max_length):
        curr_y_offset = 999999999
        # Search for local minimum y offset
        while True:
            if i_short >= min_length:
                # End of polygon reached, break
                i_short = min_length - 1
                break
            x_short = coords[short_side][i_short][0]
            x_long = coords[long_side][i_long][0]
            y_short = coords[short_side][i_short][1]
            y_long = coords[long_side][i_long][1]
            y_offset = np.sqrt((y_short - y_long) ** 2)
            if y_offset > curr_y_offset:
                # Minimum found in previous iteration, break
                i_short -= 1
                break
            if i_long == 0:
                # Special case: in the beginning, ensure that no points are ommited
                midpoint = ((x_short + x_long) // 2, (y_short + y_long) // 2)
                distance = np.sqrt((x_short - x_long) ** 2 + (y_short - y_long) ** 2)
                midpoints.append(midpoint)
                short.append((x_short, y_short))
                long.append((x_long, y_long))
                distances.append(distance)
            curr_y_offset = y_offset
            i_short += 1
        # Minimum y_offset was found, now store corresponding points, midpoint and distance
        x_short = coords[short_side][i_short][0]
        x_long = coords[long_side][i_long][0]
        y_short = coords[short_side][i_short][1]
        y_long = coords[long_side][i_long][1]
        distance = np.sqrt((x_short - x_long) ** 2 + (y_short - y_long) ** 2)
        midpoint = ((x_short + x_long) // 2, (y_short + y_long) // 2)
        midpoints.append(midpoint)
        short.append((x_short, y_short))
        long.append((x_long, y_long))
        distances.append(distance)
    return midpoints, distances, short, long, left_side
