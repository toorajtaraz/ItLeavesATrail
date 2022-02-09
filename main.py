import cv2
import numpy as np
from time import gmtime, strftime
from math import pi, atan2

#NEEDED DATA
TWO_PI = 2 * pi

#Remember to change this to the correct path
data_folder_path = '$HOME/Documents/ItLeavesATrail/data/'

books = [
    'Angels and Demons', 
    # 'Anne of Green Gables & Anne of Avonlea', 
    'David Copperfield', 
    # 'Dracula', 
    # 'Pickwick Papers', 
    # 'To Kill a Mockingbird',
    # 'Tom Sawyer and Huckleberry Finn',
    # 'Twenty Thousand Leagues Under the Sea',
    # 'Twilight - New Moon',
    # 'Twilight - Eclipse'
]

format_extentions = [
    '.jpg',
    '.mp4'
]

base_clips = [
    '0.MOV',
    '1.MOV'
]

#LOADING BINARIES
book_covers = []

for book in books:
    path = data_folder_path + book + format_extentions[0]
    book_covers.append(cv2.imread(path))

trailers = []

for trailer in books:
    path = data_folder_path + trailer + format_extentions[1]
    trailers.append(path)


# for cover in book_covers:
#     cv2.imshow("img", cover)
#     cv2.waitKey(0)