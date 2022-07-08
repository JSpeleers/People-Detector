import threading

import cvlib    # high level module, uses YOLO model with the find_common_objects method
import cv2      # image/video manipulation, allows us to pass frames to cvlib
from argparse import ArgumentParser
import os
import sys
from datetime import datetime
# from twilio.rest import Client  # used for texting if you'd like, flag is optional
import smtplib, ssl # for sending email alerts
from email.message import EmailMessage
import imghdr


# these will need to be fleshed out to not miss any formats
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp']
VID_EXTENSIONS = ['.mov', '.mp4', '.avi', '.mpg', '.mpeg', '.m4v', '.mkv']

# function takes a file name(full path), checks that file for human shaped objects
# saves the frames with people detected into directory named 'save_directory'
def human_checker(video_file_name, save_directory, yolo='yolov4', continuous=False, nth_frame=10, confidence=.65,
                  gpu=False, no_images=False):
    # tracking if we've found a human or not
    is_human_found = False
    analyze_error = False
    is_valid = False

    # we'll need to increment every time a person is detected for file naming
    person_detection_counter = 0

    # check if image
    if os.path.splitext(video_file_name)[1] in IMG_EXTENSIONS:
        frame = cv2.imread(video_file_name)  # our frame will just be the image
        # make sure it's a valid image
        if frame is not None:
            frame_count = 8  # this is necessary so our for loop runs below
            nth_frame = 1
            is_valid = True
            print(f'Image')
        else:
            is_valid = False
            analyze_error = True
    # check if video
    elif os.path.splitext(video_file_name)[1] in VID_EXTENSIONS:
        vid = cv2.VideoCapture(video_file_name)
        # get approximate frame count for video
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # make sure it's a valid video
        if frame_count > 0:
            is_valid = True
            print(f'{frame_count} frames')
        else:
            is_valid = False
            analyze_error = True
    else:
        print(f'\nSkipping {video_file_name}')

    if is_valid:
        # look at every nth_frame of our video file, run frame through detect_common_objects
        # Increase 'nth_frame' to examine fewer frames and increase speed. Might reduce accuracy though.
        # Note: we can't use frame_count by itself because it's an approximation and could lead to errors
        for frame_number in range(1, frame_count - 6, nth_frame):

            # if not dealing with an image
            if os.path.splitext(video_file_name)[1] not in IMG_EXTENSIONS:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, frame = vid.read()

            # feed our frame (or image) in to detect_common_objects
            try:

                bbox, labels, conf = cvlib.detect_common_objects(frame, model=yolo, confidence=confidence,
                                                                 enable_gpu=gpu)
            except Exception as e:
                print(e)
                analyze_error = True
                break

            if 'person' in labels:
                person_detection_counter += 1
                is_human_found = True

                if not no_images:
                    # create image with bboxes showing people and then save
                    marked_frame = cvlib.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)
                    save_file_name = os.path.basename(os.path.splitext(video_file_name)[0]) + '-' + str(
                        person_detection_counter) + '.jpeg'
                    cv2.imwrite(save_directory + '/' + save_file_name, marked_frame)

                if continuous is False:
                    break

    return is_human_found, analyze_error


# takes a directory and returns all files and directories within
def list_dir(dir_name):
    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # ignore hidden files and directories
        if entry[0] != '.':
            # Create full path
            full_path = os.path.join(dir_name, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(full_path):
                all_files = all_files + list_dir(full_path)
            else:
                all_files.append(full_path)
    return all_files


if __name__ == "__main__":

    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)

    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='Path to video folder')
    parser.add_argument('-f', default='', help='Used to select an individual file')
    parser.add_argument('--tiny_yolo', action='store_true',
                        help='Flag to indicate using YoloV4-tiny model instead of the full one. Will be faster but less accurate.')
    parser.add_argument('--continuous', action='store_true',
                        help='This option will go through entire video file and save all frames with people. Default behavior is to stop after first person sighting.')
    parser.add_argument('--confidence', type=int, choices=range(1, 100), default=65,
                        help='Input a value between 1-99. This represents the percent confidence you require for a hit. Default is 65')
    parser.add_argument('--frames', type=int, default=10, help='Only examine every nth frame. Default is 10')
    parser.add_argument('--gpu', action='store_true',
                        help='Attempt to run on GPU instead of CPU. Requires Open CV compiled with CUDA enables and Nvidia drivers set up correctly.')
    parser.add_argument('--no-images', action='store_true',
                        help='Do not create images/frames of detected people.')

    args = vars(parser.parse_args())

    # decide which model we'll use, default is 'yolov3', more accurate but takes longer
    if args['tiny_yolo']:
        yolo_string = 'yolov4-tiny'
    else:
        yolo_string = 'yolov4'

    # check our inputs, can only use either -f or -d but must use one
    if args['f'] == '' and args['directory'] == '':
        print('You must select either a directory with -d <directory> or a file with -f <file name>')
        sys.exit(1)
    if args['f'] != '' and args['directory'] != '':
        print('Must select either -f or -d but can''t do both')
        sys.exit(1)

    every_nth_frame = args['frames']
    confidence_percent = args['confidence'] / 100

    gpu_flag = False
    if args['gpu']:
        gpu_flag = True

    # create a directory to hold snapshots and log file
    time_stamp = datetime.now().strftime('%m%d%Y-%H%M%S')
    os.mkdir(time_stamp)

    print('Beginning Detection')
    print(f'Directory {time_stamp} has been created')
    print(f"Confidence threshold set to {args['confidence']}%")
    print(f'Examining every {every_nth_frame} frames.')
    print(f"Continous examination is set to {args['continuous']}")
    print(f"GPU is set to {args['gpu']}")
    print('\n\n')
    print(datetime.now().strftime('%m%d%Y-%H:%M:%S'))

    # open a log file and loop over all our video files
    with open('_' + time_stamp + '/' + time_stamp + '.txt', 'w') as log_file:
        if args['f'] == '':
            media_directory_list = list_dir(args['directory'] + '/')
        else:
            media_directory_list = [args['f']]

        # what video we are on
        working_on_counter = 1

        for media_file in media_directory_list:
            print(
                f'Examining {media_file}: {working_on_counter} of {len(media_directory_list)}: '
                f'{int((working_on_counter / len(media_directory_list) * 100))}%', end='')

            # check for people
            human_detected, error_detected = human_checker(str(media_file), time_stamp, yolo=yolo_string,
                                                           nth_frame=every_nth_frame, confidence=confidence_percent,
                                                           continuous=args['continuous'], gpu=gpu_flag,
                                                           no_images=args['no_images'])

            if human_detected:
                print(f'Human detected in {media_file} ({os.path.getsize(media_file)})')
                log_file.write(f'{media_file} \n')

            if error_detected:
                print(f'\nError in analyzing {media_file}')
                log_file.write(f'Error in analyzing {media_file} \n')

            working_on_counter += 1
