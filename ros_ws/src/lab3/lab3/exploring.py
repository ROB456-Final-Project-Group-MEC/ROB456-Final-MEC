#!/usr/bin/env python3

# This assignment lets you both define a strategy for picking the next point to explore and determine how you
#  want to chop up a full path into way points. You'll need path_planning.py as well (for calculating the paths)
#
# Note that there isn't a "right" answer for either of these. This is (mostly) a light-weight way to check
#  your code for obvious problems before trying it in ROS. It's set up to make it easy to download a map and
#  try some robot starting/ending points
#
# Given to you:
#   Image handling
#   plotting
#   Some structure for keeping/changing waypoints and converting to/from the map to the robot's coordinate space
#
# Slides

# The ever-present numpy
import numpy as np
from scipy.ndimage import binary_dilation
import os

# Your path planning code
try:
    import lab3.path_planning as path_planning
except:
    import path_planning as path_planning


# -------------- Plot just the image ---------------
def plot_image(im_threshhold, zoom=1.0):
    """Show the map plus, optionally, the robot location and points marked as ones to explore/use as end-points
    @param im - the image of the SLAM map
    @param im_threshhold - the image of the SLAM map
    @param robot_loc - the location of the robot in pixel coordinates
    @param best_pt - The best explore point (tuple, i,j)
    @param explore_points - the proposed places to explore, as a list"""

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1)
    axs.imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs.set_title("threshold image")

    # Implements a zoom - set zoom to 1.0 if no zoom
    width = im_threshhold.shape[1]
    height = im_threshhold.shape[0]
    axs.axis('equal')

    axs.set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
    axs.set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)

# -------------- Plot just the image ---------------
def plot_goal_pts_explore_pts(im_threshhold, zoom=1.0, robot_loc=None, goal_loc=None, path=None, goal_pts=None, explore_points=None, best_pt=None):
    """Show the map plus, optionally, the robot location and goal location and proposed path
    @param im - the image of the SLAM map (numpy array)
    @param im_threshhold - the image of the SLAM map, threshholded
    @param zoom - how much to zoom into the map (value between 0 and 1)
    @param robot_loc - the location of the robot in pixel coordinates
    @param goal_loc - the location of the goal in pixel coordinates
    @param path - the proposed path in pixel coordinates"""

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1)
    axs.imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs.set_title("threshold image")
    # axs.axis('equal')

    # Implements a zoom - set zoom to 1.0 if no zoom
    width = im_threshhold.shape[1]
    height = im_threshhold.shape[0]

    axs.set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
    axs.set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)

    # Show original and thresholded image
    if explore_points is not None:
        for p in explore_points:
            axs.plot(p[0], p[1], '.b', markersize=2)
    if robot_loc is not None:
        axs.plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
    if path is not None:
        for p, q in zip(path[0:-1], path[1:]):
            axs.plot([p[0], q[0]], [p[1], q[1]], '-g', markersize=2)
            axs.plot(p[0], p[1], '.g', markersize=2)
    if goal_pts is not None:
        # axs.plot([goal_loc[0], goal_pts[0][0]], [goal_loc[1], goal_pts[0][1]], color='cyan', ls='-', markersize=2)
        # axs.plot(goal_pts[0][0], goal_pts[0][1], color='magenta', marker='o', markersize=2)
        for p, q in zip(goal_pts[0:-1], goal_pts[1:]):
            axs.plot([p[0], q[0]], [p[1], q[1]], color='cyan', ls='-', markersize=2)
            axs.plot(p[0], p[1], color='magenta', marker='o', markersize=2)
    if best_pt is not None:
        axs.plot(best_pt[0], best_pt[1], color='pink', marker='x', markersize=10)
    if goal_loc is not None:
        axs.plot(goal_loc[0], goal_loc[1], color='gold', marker='*', markersize=10)



# -------------- Showing start and end and path ---------------
def plot_with_explore_points(im_threshhold, zoom=1.0, robot_loc=None, explore_points=None, best_pt=None):
    """Show the map plus, optionally, the robot location and points marked as ones to explore/use as end-points
    @param im - the image of the SLAM map
    @param im_threshhold - the image of the SLAM map
    @param robot_loc - the location of the robot in pixel coordinates
    @param best_pt - The best explore point (tuple, i,j)
    @param explore_points - the proposed places to explore, as a list"""

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1)
    axs.imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs.set_title("threshold image")

    # Show original and thresholded image
    if explore_points is not None:
        for p in explore_points:
            axs.plot(p[0], p[1], '.b', markersize=2)

    if robot_loc is not None:
        axs.plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
    if best_pt is not None:
        axs.plot(best_pt[0], best_pt[1], '*y', markersize=10)
    # axs.axis('equal')

    # Implements a zoom - set zoom to 1.0 if no zoom
    width = im_threshhold.shape[1]
    height = im_threshhold.shape[0]

    axs.set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
    axs.set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)


# -------------- For converting to the map and back ---------------
def convert_pix_to_x_y(im_size, pix, size_pix):
    """Convert a pixel location [0..W-1, 0..H-1] to a map location (see slides)
    Note: Checks if pix is valid (in map)
    @param im_size - width, height of image
    @param pix - tuple with i, j in [0..W-1, 0..H-1]
    @param size_pix - size of pixel in meters
    @return x,y """
    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Pixel {pix} not in image, image size {im_size}")

    return [size_pix * pix[i] / im_size[1-i] for i in range(0, 2)]


def convert_x_y_to_pix(im_size, x_y, size_pix):
    """Convert a map location to a pixel location [0..W-1, 0..H-1] in the image/map
    Note: Checks if x_y is valid (in map)
    @param im_size - width, height of image
    @param x_y - tuple with x,y in meters
    @param size_pix - size of pixel in meters
    @return i, j (integers) """
    pix = [int(x_y[i] * im_size[1-i] / size_pix) for i in range(0, 2)]

    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Loc {x_y} not in image, image size {im_size}")
    return pix


def is_reachable(im, pix):
    """ Is the pixel reachable, i.e., has a neighbor that is free?
    Used for
    @param im - the image
    @param pix - the pixel i,j"""

    # GUIDE: Returns True (the pixel is adjacent to a pixel that is free)
    #  False otherwise
    # You can use four or eight connected - eight will return more points
    # YOUR CODE HERE

    # ---- this is my original way of doing it but it is really slow, so I switched to the method below ----
    # pixels_to_check = []
    # for ix in range(-1, 2):
    #     for iy in range(-1, 2):
    #         pixels_to_check.append((pix[0] + ix, pix[1] + iy))

    # return np.any([path_planning.is_free(im, p) for p in pixels_to_check])

    # --------------------- this is a much faster way to do it -------------------------
    """ Checks if a pixel has a free neighbor (255) using a 3x3 slice. """
    x, y = pix
    
    # Grab a 3x3 window around the target pixel instantly
    window = im[y-1:y+2, x-1:x+2]
    
    # Let NumPy check if 255 exists anywhere in that 3x3 grid
    return np.any(window == 255)



def find_all_possible_goals(im):
    """ Find all of the places where you have a pixel that is unseen next to a pixel that is free
    It is probably easier to do this, THEN cull it down to some reasonable places to try
    This is because of noise in the map - there may be some isolated pixels
    @param im - thresholded image
    @return list of possible pixel (x,y) locations"""

    # YOUR CODE HERE

    # ---- this is my original way of doing it but it is really slow, so I switched to the method below ----
    # # create a emplty list to hold the possible goals
    # all_possible_goals = []

    # # this is y,x
    # unseen_points = np.argwhere(im == 128)
    # # print(f"Found {len(unseen_points)} unseen points")
    # # print(unseen_points)

    # for unseen_point in unseen_points:
    #     if not 10 < unseen_point[0] < im.shape[0] - 10 or not 10 < unseen_point[1] < im.shape[1] - 10:
    #         continue
    #     if is_reachable(im, (unseen_point[1], unseen_point[0])):
    #         all_possible_goals.append((unseen_point[1], unseen_point[0]))


    # return all_possible_goals

    # --------------------- this is a much faster way to do it -------------------------
    # Create a mask of all free spaces
    free_spaces = (im == 255)
    
    # "Grow" the free spaces by 1 pixel in all 8 directions
    # A 3x3 matrix of True values tells it to use 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    adjacent_to_free = binary_dilation(free_spaces, structure=structure)
    
    # Create a mask of all unseen spaces
    unseen_spaces = (im == 128)
    
    # Create a mask for valid boundaries (the 10 pixel margin)
    valid_boundaries = np.zeros_like(im, dtype=bool)
    valid_boundaries[16:-16, 16:-16] = True
    
    # Combine all conditions using fast boolean math (&)
    # A goal must be unseen AND adjacent to free space AND within boundaries
    valid_goals_mask = unseen_spaces & adjacent_to_free & valid_boundaries
    
    # Extract the Y, X coordinates of the True values
    y_coords, x_coords = np.where(valid_goals_mask)
    
    # Return as a list of (x, y) tuples
    return list(zip(x_coords, y_coords))



def find_best_points(im, possible_points : list, robot_loc=None):
    """ Pick one of the unseen points to go to
    @param im - thresholded image
    @param possible_points - possible points to chose from (list of tuples)
    @param robot_loc - location of the robot (in case you want to factor that in)
    """
    # YOUR CODE HERE
    better_pts = []
    for p in possible_points:
        count_free = 0
        count_unseen = 0
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                if path_planning.is_free(im, (p[0] + ix, p[1] + iy)):
                    count_free += 1
                elif path_planning.is_unseen(im, (p[0] + ix, p[1] + iy)):
                    count_unseen += 1
        if count_free < 3:
            continue
        if count_free + count_unseen != 9:
            continue
        better_pts.append(p)

    better_filtered2 = []
    for p in better_pts:
        count_better_pts = 0
        for ix in range(-2, 3):
            for iy in range(-2, 3):
                if (p[0] + ix, p[1] + iy) in better_pts:
                    count_better_pts += 1
        if count_better_pts >= 5:
            better_filtered2.append(p)

    return better_filtered2

def find_best_point(im, possible_points : list, robot_loc, search_dist=80):
    """ Pick one of the unseen points to go to
    @param im - thresholded image
    @param possible_points - possible points to chose from (list of tuples)
    @param robot_loc - location of the robot (in case you want to factor that in)
    """
    # YOUR CODE HERE

    better_pts = []
    # min_dist = np.hypot(im.shape[1], im.shape[0])/
    min_dist_goal = (-1,-1)
    for p in possible_points:
        count_free = 0
        count_unseen = 0
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                if path_planning.is_free(im, (p[0] + ix, p[1] + iy)):
                    count_free += 1
                elif path_planning.is_unseen(im, (p[0] + ix, p[1] + iy)):
                    count_unseen += 1
        if count_free < 3:
            continue
        if count_free + count_unseen != 9:
            continue
        better_pts.append(p)

    better_filtered2 = []
    for p in better_pts:
        count_better_pts = 0
        for ix in range(-2, 3):
            for iy in range(-2, 3):
                if (p[0] + ix, p[1] + iy) in better_pts:
                    count_better_pts += 1
        if count_better_pts >= 5:
            better_filtered2.append(p)

    at_least_search_dist = []
    for p in better_filtered2:
        dist = np.hypot(p[0] - robot_loc[0], p[1] - robot_loc[1])
        if dist > search_dist:
            at_least_search_dist.append(p)



    min_dist_goal = (-1,-1)
    if len(at_least_search_dist) > 0:

        min_dist = np.hypot(im.shape[1], im.shape[0])
        for p in at_least_search_dist:
            
            dist = np.hypot(p[0] - robot_loc[0], p[1] - robot_loc[1])
            if dist < min_dist:
                min_dist_goal = p 
                min_dist = dist

    else:

        min_dist = np.hypot(im.shape[1], im.shape[0])
        for p in at_least_search_dist:
            
            dist = np.hypot(p[0] - robot_loc[0], p[1] - robot_loc[1])
            if dist < min_dist:
                min_dist_goal = p 
                min_dist = dist


    return min_dist_goal


def find_waypoints(im, path, distance_between_points=5):
    """ Place waypoints along the path
    @param im - the thresholded image
    @param path - the initial path
    @ return - a new path"""

    # Again, no right answer here
    # YOUR CODE HERE

    # waypoints = []

    # # This is a simple way to do it - just take every Nth point. 
    # # TODO: I could also do something fancier, like only add a point if the path changes direction by more than some amount
    # for i in range(0, len(path), distance_between_points):
    #     waypoints.append(path[i])

    # return waypoints

    new_path = path.copy()
    new_path.reverse()

    waypoints = [new_path[0]]

    curr_direct = -np.pi*3
    last_pt_indx = 0
    last_pt = new_path[0]

    for i in range(len(new_path)-1):
        # divide by 36 for a 5 degree allowable tolerance for changing direction
        if i+1 > last_pt_indx + distance_between_points*2 and not np.isclose(curr_direct, np.arctan2(new_path[i+1][1] - last_pt[1], new_path[i+1][0] - last_pt[0]), atol=np.pi/36):
            waypoints.append(new_path[i+1])
            try:
                curr_direct = np.arctan2(new_path[i+distance_between_points][1] - new_path[i+1][1], new_path[i+distance_between_points][0] - new_path[i+1][0])
            except IndexError:
                break
            last_pt_indx = i
            last_pt = new_path[i+1]

    waypoints.append(new_path[-1])

    # waypoints.reverse()
    return waypoints


def test_unseen(im, pts):
    for pt in pts:
        count_free = 0
        count_unseen = 0
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                if path_planning.is_free(im, (pt[0] + ix, pt[1] + iy)):
                    count_free += 1
                elif path_planning.is_unseen(im, (pt[0] + ix, pt[1] + iy)):
                    count_unseen += 1
        if count_free == 0 or count_unseen == 0:
            return False
    return True


def test_best(im, pt):
    """ Check that the selected point has at least 3 free neighbors"""
    count_free = 0
    count_unseen = 0
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            if path_planning.is_free(im, (pt[0] + ix, pt[1] + iy)):
                count_free += 1
            elif path_planning.is_unseen(im, (pt[0] + ix, pt[1] + iy)):
                count_unseen += 1
    if count_free < 3:
        return False
    if count_free + count_unseen != 9:
        return False
    return True


if __name__ == '__main__':
    _, im_thresh = path_planning.open_image("map.pgm")

    robot_start_loc = (60, 40)

    all_unseen = find_all_possible_goals(im_thresh)
    best_unseen = find_best_point(im_thresh, all_unseen, robot_loc=robot_start_loc)

    assert test_unseen(im=im_thresh, pts=all_unseen)
    assert test_best(im=im_thresh, pt=best_unseen)

    plot_with_explore_points(im_thresh, zoom=1.0, robot_loc=robot_start_loc, explore_points=all_unseen, best_pt=best_unseen)

    path = path_planning.dijkstra(im_thresh, robot_start_loc, best_unseen)
    waypoints = find_waypoints(im_thresh, path)
    path_planning.plot_with_path(im_thresh, zoom=1.0, robot_loc=robot_start_loc, goal_loc=best_unseen, path=waypoints)

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt
    plt.show()

    print("Done")
