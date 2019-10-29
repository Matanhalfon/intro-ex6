##############################################################################
# ex6.py
# Omri Yavne
# login-omriyavne
# id-316520097
# description- image processing
##############################################################################

import sys
from ex6_helper import *
import os
from PIL import Image
import PIL
import sys
import copy
import math
NUM_OF_ARGUMENTS = 3
FIRST_ARG = 1
SECOND_ARG = 2
THIRD_ARG = 3
WHITE = 255
BLACK = 0


def otsu(image):
    """This function gives grades to each potential
    threshold value, from 0 to 255,
    and compares each one to each other.
    The function returns the highest
    threshold value by using otsu method"""
    best_threshold = 0
    best_grade = 0
    for tsh in range(256):
        black_pixels = 0
        white_pixels = 0
        black_sum = 0
        white_sum = 0
        for pixel in image:
            for number in pixel:
                if number < tsh:
                    black_sum += number
                    black_pixels += 1
                else:
                    white_sum += number
                    white_pixels += 1
        mean_black = 0
        if black_pixels == 0:
            continue
        else:
            mean_black = black_sum / black_pixels
        mean_white = 0
        if white_pixels == 0:
            continue
        else:
            mean_white = white_sum / white_pixels
        tsh_grade = int(black_pixels * white_pixels *
                        (mean_black - mean_white) ** 2)
        if tsh_grade > best_grade:
            best_threshold = tsh
            best_grade = tsh_grade
    return best_threshold


def threshold_filter(image):
    """This function receives a picture,
    in a grey scale, and turns it into a black and
    white picture, by replacing each
    pixel in the list to either black, or white
    using the best threshold"""
    new_image = []
    best_threshold = otsu(image)
    for pixel in image:
        pixel_list = []
        for number in pixel:
            if number == 0 or number < best_threshold:
                pixel_list.append(BLACK)
            else:
                pixel_list.append(WHITE)
        new_image.append(pixel_list)
    return new_image


def is_valid(i, j, mat):
    """This function checks if one number
    in the mat is not in the the edge of it,
    and has 'neighbours'"""
    if i < 0 or j < 0:
        return False
    elif j >= len(mat[0]) or i >= len(mat):
        return False
    else:
        return True


def single_value(row, col, mat, filter):
    """This function calculates
    which value the new mat should be given,
    by using convolusion with a 3*3 mat.
    If a number in the mat, is at
    one of the edges of it, the number itself
    will be considered as its own neighbour
    for any value, that the number doesn't
    have a neighbour in it"""
    top = row - 1
    left = col - 1
    sum = 0
    for i in range(3):
        for j in range(3):
            if is_valid(top + i, left + j, mat):
                sum += mat[top + i][left + j] * filter[i][j]
            else:
                sum += mat[row][col] * (filter[i][j])
    return sum


def apply_filter(image, filter):
    """This function returns a new
    mat, with the new values, by replacing
    each value from 'single_value',
    in the mew mat, and receiving a new
    picture"""
    new_mat = copy.deepcopy(image)
    for row in range(len(image)):
        for col in range(len(image[0])):
            value = single_value(row, col, image, filter)
            new_mat[row][col] = value
    return new_mat


# print (apply_filter(mat,filter_mat))


def neighbours_check(row, col, mat):
    """This function returns in a list,
    for a numbee -all of his 8 neighbours,
    including the option, of having
    less than than 8 neighbours, in
    case it is at one of the edges.
    In that case, the neighbours
    will be the number itself,
    for every time the number
    doesn't have a neighbour."""
    neighbours = []
    top = row - 1
    left = col - 1
    for i in range(3):
        for j in range(3):
            if is_valid(top + i, left + j, mat):
                neighbours.append(mat[top + i][left + j])
            else:
                neighbours.append(mat[row][col])
    neighbours.remove(mat[row][col])
    return neighbours



def avg_check(row, col, mat):
    """This function calculates
    the average of one pixel's neighbours
    average.by summing all neighbours
     and divide it by 8,
    because each number has 8 neighbours
    no matter what"""
    sum1 = sum(neighbours_check(row, col, mat))
    avg = round(sum1 / 8)
    return avg


def detect_edges(image):
    """This function detects the edges
    of a picture, by taking for each
    pixel, its original value, and
    take off from it the average
    of its neighbours.
    adn we reacieve a new mat with
    the new values"""
    new_mat = copy.deepcopy(image)
    for row in range(len(image)):
        for col in range(len(image[0])):
            new_value = abs(new_mat[row][col] -
                            (avg_check(row, col, image)))
            new_mat[row][col] = new_value
    return new_mat


def downsample_by_3(image):
    """This function is making
    a picture smaller by 3.
    we calcualte the average of
    each "sqaure" of pixel,
    and making it smaller by 3.
    the function does it by
    calling single_value function,
    that comparing the mat and a 3*3 mat,
    and multiply each value in the mat, to its
    matching value in the 3*3 mat, and sum them
    all together.
    we define our 3*3 mat to 1/9,
    so instead of dividing by 9
    (number of numbers)we divide
    each number by 9,that way,
    which is the same"""
    small_image = []
    filter = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9]]
    value = 0
    for row in range(1, len(image) - 1, 3):
        small_image.append([])
        for col in range(1, len(image[0]) - 1, 3):
            small_image[value].append\
                (round(single_value(row, col, image, filter)))
        value += 1
    return small_image


def downsample(image, max_diagonal_size):
    """This function checks how many
    time we should make our image smaller by 3,
    by comparing the suitable diagonal
    value, to the length of the rows
    and cols, using pythagoras"""
    new_image = copy.deepcopy(image)
    pythagoras = get_pythagoras(len(new_image), len(new_image[0]))
    while pythagoras > max_diagonal_size:
        new_image = downsample_by_3(new_image)
        pythagoras = get_pythagoras(len(new_image), len(new_image[0]))

    return new_image


def get_pythagoras(x, y):
    """This function calculates the length
    of a line using pythagoras"""
    return int(math.sqrt(x ** 2 + y ** 2))


mat9 = [[0, 0, 0, 0, 0, 0, 255], [0, 0, 0, 0, 0, 255, 0], [255, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0], [0, 0, 255, 0, 0, 0, 0], [0, 255, 0, 0, 0, 0, 0],
            [255, 0, 0, 0, 0, 0, 0]]

p_list = [[6, 0], [5, 1], [4, 1], [4, 2], [3, 2], [3, 3], [2, 3], [2, 4], [1, 4], [1, 5], [0, 5], [0, 6]]

def score_line(line, image):
    """ Get the score for a given 'line'
    which is a list of cordinates, we recieve
    from the ex6 helper function "pixels_on_line.
    we check the length of every segment of
    white pixels are they, for a white line,
    pow 2, and sum them.
    then we calculate their scores,
    and can tell which line has the highest score"""
    first_white_pixel = None
    last_white_pixel = None
    black_pixel = None
    sub_line_dist = 0
    is_first_white_pixel = True
    line_score = 0
    for pixel in line:
        pixel_value = image[pixel[0]][pixel[1]]
        if pixel_value == WHITE:
            if is_first_white_pixel:
                first_white_pixel = pixel
                is_first_white_pixel = False
            else:
                last_white_pixel = pixel
        if pixel_value == BLACK or pixel == line[-1]:
            if pixel_value == BLACK:
                black_pixel = pixel
            if last_white_pixel is None or last_white_pixel is None:
                continue
            if get_pythagoras(last_white_pixel[0] - black_pixel[0],
                              last_white_pixel[1] - black_pixel[1]) <= 2:
                continue
            sub_line_dist = get_pythagoras(last_white_pixel[0]\
                                           - first_white_pixel[0],
                                           last_white_pixel[1]\
                                           - first_white_pixel[1])
            line_score += sub_line_dist ** 2
            is_first_white_pixel = True
            first_white_pixel = None
            last_white_pixel = None
    return line_score
print (score_line(mat9,p_list))



def get_angle(image):
    """This function returns the most
    dominant angle of an image.
    by calculating the angles of the
    the line that has the highest score"""
    max_cross = get_pythagoras(len(image),len(image[0]))
    angle_to_return = 0
    max_line_score = 0
    current_line_score = 0
    for angle in range(180):
        for distance in range(max_cross):
            rad_angle = math.radians(angle)
            if angle == 0:
                top_line = pixels_on_line(image, rad_angle,distance, False)
                current_line_score += score_line(top_line,image)
            else:
                if angle < 90:
                    top_line = pixels_on_line\
                        (image, rad_angle,distance, True)
                    bottom_line = pixels_on_line\
                        (image, rad_angle,distance, False)
                    current_line_score += score_line(top_line,image)\
                                          + score_line(
                        bottom_line,image)
                if angle >= 90:
                    bottom_line = pixels_on_line\
                        (image, rad_angle,distance, False)
                    current_line_score += score_line(bottom_line,image)
        if max_line_score < current_line_score:
            max_line_score = current_line_score
            angle_to_return = angle
    return angle_to_return

image_matrix = load_image(r"sudoku20corrected.jpg")
# print (get_angle(image_matrix))

def rotate(image, angle):
    pass


def make_correction(image, max_diagonal):
    """This function makes the final
    image corrections by calling all
    the functions"""
    new_image = downsample(image, max_diagonal)
    new_image = threshold_filter(new_image)
    new_image = detect_edges(new_image)
    new_image = threshold_filter(new_image)
    new_image = get_angle(new_image)
    return new_image


if __name__ == '__main__':
    # This main function runs
    # the script, and by recieving three
    # arguments, it saves it into an output_file
    if len(sys.argv) != NUM_OF_ARGUMENTS + 1:
        print ("Wrong number of parameters.  The correct usage is:\
                ex6.py <image_source> <output > <max_diagonal>")
    else:
        image_source = (sys.argv[FIRST_ARG])
        output_file = (sys.argv[FIRST_ARG])
        max_diagonal = (sys.argv[THIRD_ARG])
        image = load_image(image_source)
        new_image = make_correction(image, int(max_diagonal))
        save(new_image, output_file)