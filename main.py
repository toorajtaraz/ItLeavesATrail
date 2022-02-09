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