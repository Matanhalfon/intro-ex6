###################################
# ex6.py
# matan halfon,matan.halfon,205680648
# intro2cs ex6 2017-2018
# Describe : A program פ
###################################

import PIL
import ex6_helper as HELPER
import math
import sys
import os

NUM_OF_BLACK = 0
NUM_OF_WHITE = 1
EADGE_filter = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
CLEAR_filter = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
BLACK = 0
WHITE = 255


def black_white_histogram(image):
    '''A function that return an histogram of the black hue in the image'''
    histogram = [0] * 256
    for i in range(len(image)):
        for j in range(len(image[i])):
            histogram[image[i][j]] += 1
    return histogram


def get_variant(hue, histogram):
    '''A function that get the inatra variant of the selected hue'''
    num_of_black = 0
    num_of_white = 0
    sum_of_white = 0
    sum_of_black = 0
    mean_white = 0
    mean_black = 0
    for i in range(len(histogram)):
        if i < hue:
            num_of_black += histogram[i]
            sum_of_black += histogram[i] * i
        elif hue <= i:
            num_of_white += histogram[i]
            sum_of_white += histogram[i] * i
    if num_of_black != 0:
        mean_black = sum_of_black / num_of_black
    if num_of_white != 0:
        mean_white = sum_of_white / num_of_white
    variant = num_of_black * num_of_white * (mean_black - mean_white) ** 2
    return variant


def otsu(image):
    '''A function that finds the otsu'''
    histogram = black_white_histogram(image)
    otsu = 0
    variant = 0
    for i in range(256):
        the_inta_var = get_variant(i, histogram)
        if variant < the_inta_var:
            variant = the_inta_var
            otsu = i
    return otsu


def threshold_filter(image):
    ''' The function cheaks the otsu of the image and set the pixels that have lower vule set to 0(black) \
    and the pixels higher then the otu set to  white (255)'''
    optimal_threshold = otsu(image)
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < optimal_threshold:
                image[i][j] = BLACK
            if optimal_threshold <= image[i][j]:
                image[i][j] = WHITE
    return image


def filterd_table(image, filter, row, pixel):
    '''A function that get a spcific pixel and return the table of close pixels multiplyed by the\
    filter(if the close pixels are a pading thay replaced with the pixel itself '''
    sum_of_pixel = 0
    for j in range(len(filter)):
        for k in range(len(filter[j])):
            if (row - 1 + j < 0 or pixel - 1 + k < 0) or (
                                len(image) - 1 < row - 1 + j or len(image[0]) - 1 < pixel - 1 + k):
                sum_of_pixel += (image[row][pixel])
                continue
            sum_of_pixel += (filter[j][k] * image[row - 1 + j][pixel - 1 + k])
    return sum_of_pixel


def apply_filter(image, filter):
    '''A function that output a new image that for each pixel in the orginal pic get the sum of values of \
    the near pixels multiplayed by the filter matrix'''
    image_output = []
    for row in range(len(image)):
        image_output.append([])
        for pixel in range(len(image[row])):
            new_pixel = int(filterd_table(image, filter, row, pixel))
            if new_pixel < BLACK:
                new_pixel = -new_pixel
            elif WHITE < new_pixel:
                new_pixel = WHITE
            image_output[row].append(new_pixel)
    return image_output


def detect_edges(image):
    '''A function  create a new image which in each pixel get the vule of the origenal pixel minous the\
    average of the near pixels'''
    image_output = []
    for row in range(len(image)):
        image_output.append([])
        for pixel in range(len(image[row])):
            sum_of_nibers = filterd_table(image, EADGE_filter, row, pixel)
            new_pixel = int(image[row][pixel] - sum_of_nibers / 8)
            if new_pixel < BLACK:
                new_pixel = -new_pixel
            elif WHITE < new_pixel:
                new_pixel = WHITE
            image_output[row].append(new_pixel)
    return image_output


def downsample_by_3(image):
    '''A function that downsample the image by 3 and give eanch pixel the average of 3x3 pixels around the/
    original pix '''
    downsample = []
    filter = CLEAR_filter
    for row in range(1, len(image), 3):
        if row > len(image):
            break
        small_row = []
        for pixel in range(1, len(image[row]), 3):
            if pixel > len(image[row]):
                break
            sum_of_near_pix = filterd_table(image, filter, row, pixel)
            new_pixel = int(sum_of_near_pix / 9)
            if new_pixel < BLACK:
                new_pixel = -new_pixel
            elif WHITE < new_pixel:
                new_pixel = WHITE
            small_row.append(new_pixel)
        downsample.append(small_row[:])
        small_row.clear()
    return downsample


def pitaguras(widgh, lengh):
    '''A function that get widgh and the hight the the finds the diagonal if the square  '''
    diagonal = math.sqrt(math.pow(widgh, 2) + math.pow(lengh, 2))
    return diagonal


def downsample(image, max_diagonl):
    '''A function that get the a max diagonal value and keep on downsmaple the image until hiting the /
    wanted value'''
    image_to_return = image
    while max_diagonl < pitaguras(len(image_to_return), len(image_to_return[0])):
        if len(image_to_return) == 1 or len(image_to_return[0]) == 1:
            return image_to_return
        image_to_return = downsample_by_3(image_to_return)
    return image_to_return


def get_dictance(cor_a, cor_b):
    '''A function that measure the distance between two cordints '''
    the_dictance = math.sqrt(math.pow((cor_a[0] - cor_b[0]), 2) + math.pow((cor_a[1] - cor_b[1]), 2))
    return the_dictance


def get_distance_squre(cord_a, cord_b):
    '''A function that get the cord and find the distance raised by 2'''
    dictance = get_dictance(cord_a, cord_b)
    dictance_squre = math.pow(dictance, 2)
    return dictance_squre


def line_grade(pix_line, pic):
    '''A function that get the grade of each line, the function find a white pixel that followed by white pixel/
    mesuare the distance between them and return the line grade'''
    lines_dictance = []
    last_call = 0
    for i in range(len(pix_line) - 1):
        if i + 1 > len(pix_line):
            break
        if i < last_call:
            continue
        if pic[pix_line[i][0]][pix_line[i][1]] == BLACK:
            continue
        elif pic[pix_line[i][0]][pix_line[i][1]] == WHITE:
            if ((pic[pix_line[i + 1][0]][pix_line[i + 1][1]] == WHITE)) and \
                    ((get_dictance(pix_line[i], pix_line[i + 1]) < 2)):
                for j in range(i, len(pix_line)):
                    if j == len(pix_line) - 1:
                        last_call = j
                        lines_dictance.append(get_distance_squre(pix_line[i], pix_line[j]))
                        break
                    elif (pic[pix_line[j + 1][0]][pix_line[j + 1][1]] == WHITE) \
                            and ((get_dictance(pix_line[i], pix_line[i + 1]) < 2)):
                        continue
                    elif (pic[pix_line[j + 1][0]][pix_line[j + 1][1]] == BLACK):
                        last_call = j
                        lines_dictance.append(get_distance_squre(pix_line[i], pix_line[j]))
                        break
            continue
    return sum(lines_dictance)


def get_angle(image):
    '''A function that finds the dominate angle ,the function use the "line_grade" function to grade each \
     distance for each degree and return the max garde and angle '''
    pic = threshold_filter(image)
    diagonnal = int(pitaguras(len(image), len(image[0])))
    degree_to_return = 0
    max_digree = 0
    for i in range(180):
        the_rad = math.radians(i)
        degree_grade = 0
        for j in range(0, diagonnal, 1):
            the_line = HELPER.pixels_on_line(pic, the_rad, j)
            degree_grade += line_grade(the_line, pic)
            if 0 < i < 90:
                the_line = HELPER.pixels_on_line(pic, the_rad, j, False)
                degree_grade += line_grade(the_line, pic)
        if degree_grade > max_digree:
            max_digree = degree_grade
            degree_to_return = i
    return degree_to_return


def black_backround(image, angle):
    ''''A function that create the a black backround in the size of the image with the right angle '''
    backround_image = []
    pic_hight = len(image)
    pic_widght = len(image[0])
    if angle <= 90:
        backround_width = int(abs((pic_widght * math.cos(angle))) + abs(pic_hight * math.sin(angle)))
        backround_hight = int(abs(pic_widght * math.sin(angle)) + abs(pic_hight * math.cos(angle)))
        if backround_width < 0:
            backround_width = -backround_width
        if backround_hight < 0:
            backround_hight = -backround_hight
        for i in range(backround_hight):
            backround_image.append([])
            for j in range(backround_width):
                backround_image[i].append(0)
        return backround_image
    if 90 < angle:
        backround_width = int(abs((pic_hight * math.cos(angle))) + abs(pic_widght * math.sin(angle)))
        backround_hight = int(abs(pic_hight * math.sin(angle)) + abs(pic_widght * math.cos(angle)))
        if backround_width < 0:
            backround_width = -backround_width
        if backround_hight < 0:
            backround_hight = -backround_hight
        for i in range(backround_hight):
            backround_image.append([])
            for j in range(backround_width):
                backround_image[i].append(20)
        return backround_image


def move_by_degree(cords, angle):
    '''A function that move a cordinte it get by a spcific degree'''
    y, x = cords[0], cords[1]
    new_y, new_x = (math.sin(angle) * x + math.cos(angle) * y, math.cos(angle) * x - math.sin(angle) * y)
    return new_y, new_x


def rotate(image, angle):
    '''A function that rotate an image by the angle the function get'''
    angle = math.radians(angle)
    back = black_backround(image, angle)
    back_hight = len(back)
    back_wight = len(back[0])
    image_hight = len(image)
    image_wigth = len(image[0])
    image_center = [image_hight / 2, image_wigth / 2]
    back_center = [back_hight / 2, back_wight / 2]
    for i in range(len(back)):
        for j in range(len(back[0])):
            new_i = i - back_center[0]
            new_j = j - back_center[1]
            moved_i, moved_j = move_by_degree([new_i, new_j], angle)
            fixed_i = int(moved_i + image_center[0])
            fixed_j = int(moved_j + image_center[1])
            if fixed_i >= len(image) - 1 or fixed_j >= len(image[0]) - 1:
                continue
            if fixed_i < 0 or fixed_j < 0:
                continue
            back[i][j] = image[fixed_i][fixed_j]
    return back


def make_correction(image, max_diagonal):
    '''A function that get a pic mesure the dominate angle and correct the image by that degree'''
    pic_to_change = downsample(image, max_diagonal)
    pic_to_change = threshold_filter(detect_edges(threshold_filter(pic_to_change)))
    dominete_angle = get_angle(pic_to_change)
    return rotate(image, dominete_angle)


def main():
    if len(sys.argv) != 4:
        print("Wrong number of parameters. The correct usage is:\
        ex6.py <image_source> <output > <max_diagonal>")
    else:
        _, image_source, output_name, max_diagonal =sys.argv
        if not os.path.isfile(image_source):
            print("no image file")
        else:
            max_diagonal=float(max_diagonal)
            image = HELPER.load_image(image_source)
            fixed_image = make_correction(image, max_diagonal)
            HELPER.save(fixed_image, output_name)


if __name__ == '__main__':
    main()
