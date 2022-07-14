import os
import sys
import threading
from argparse import ArgumentParser
from datetime import datetime

import cv2  # image/video manipulation, allows us to pass frames to cvlib
import cvlib  # high level module, uses YOLO model with the find_common_objects method

# these will need to be fleshed out to not miss any formats
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp']
VID_EXTENSIONS = ['.mov', '.mp4', '.avi', '.mpg', '.mpeg', '.m4v', '.mkv']


def get_file_extension(file_name):
    return os.path.splitext(file_name)[1]


def is_video(file_name):
    return get_file_extension(file_name) in VID_EXTENSIONS


def is_image(file_name):
    return get_file_extension(file_name) in IMG_EXTENSIONS


# function takes a file name(full path), checks that file for human shaped objects
# saves the frames with people detected into directory named 'save_directory'
def person_checker(file_name, save_directory, yolo='yolov4', nth_frame=10, confidence=.65, gpu=False,
                   no_images=False):
    # tracking if we've found a human or not
    is_person_found = False
    analyze_error = False
    is_valid = False

    # check if image
    if is_image(file_name):
        frame = cv2.imread(file_name)  # our frame will just be the image
        # make sure it's a valid image
        if frame is not None:
            frame_count = 8  # this is necessary so our for loop runs below
            nth_frame = 1
            is_valid = True
        else:
            is_valid = False
            analyze_error = True
    # check if video
    elif is_video(file_name):
        vid = cv2.VideoCapture(file_name)
        # get approximate frame count for video
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # make sure it's a valid video
        if frame_count > 0:
            is_valid = True
        else:
            is_valid = False
            analyze_error = True
    else:
        print(f'Skipping {file_name}')

    if not is_valid:
        return is_person_found, analyze_error

    # look at every nth_frame of our video file, run frame through detect_common_objects
    # Increase 'nth_frame' to examine fewer frames and increase speed. Might reduce accuracy though.
    # Note: we can't use frame_count by itself because it's an approximation and could lead to errors
    for frame_number in range(1, frame_count - 6, nth_frame):
        # if not dealing with an image
        if is_video(file_name):
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

        is_person_found = 'person' in labels
        print(('Person detected' if is_person_found else 'No person detected')
              + f' in frame {frame_number}/{frame_count}')
        if is_person_found:
            break

    # Person detected or went through all frames without person detected
    image_path = ''
    if not no_images:
        # create image with bboxes showing people and then save
        marked_frame = cvlib.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)

        person_found_dir = 'person' if is_person_found else 'no_person'
        image_dir = save_directory + '/' + person_found_dir
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        save_file_name = f'{os.path.basename(file_name)}.{working_on_counter}.jpg'
        image_path = image_dir + '/' + save_file_name
        cv2.imwrite(image_path, marked_frame)
        print(f'Wrote debug image to {image_path}')

    detection[is_person_found].append((file_name, image_path))

    return is_person_found, analyze_error


# takes a directory and returns all files and directories within
def list_dir(dir_name, img_only=False, vid_only=False):
    all_files = list()
    # Iterate over all the entries
    for entry in os.listdir(dir_name):
        # ignore hidden files and directories
        if entry[0] != '.':
            # Create full path
            full_path = os.path.join(dir_name, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(full_path):
                all_files = all_files + list_dir(full_path)
            # only add video file if not img_only, only add image file if not vid_only
            elif (is_video(full_path) and not img_only) or (is_image(full_path) and not vid_only):
                all_files.append(full_path)
    return all_files


def human_size(file_bytes, units=[' bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']):
    """ Returns a human readable string representation of bytes """
    return str(file_bytes) + units[0] if file_bytes < 1024 else human_size(file_bytes >> 10, units[1:])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='Path to video folder')
    parser.add_argument('-f', default='', help='Used to select an individual file')
    parser.add_argument('--tiny_yolo', action='store_true',
                        help='Flag to indicate using YoloV4-tiny model instead of the full one. Will be faster but less accurate.')
    parser.add_argument('--confidence', type=int, choices=range(1, 100), default=65,
                        help='Input a value between 1-99. This represents the percent confidence you require for a hit. Default is 65')
    parser.add_argument('--frames', type=int, default=10, help='Only examine every nth frame. Default is 10')
    parser.add_argument('--gpu', action='store_true',
                        help='Attempt to run on GPU instead of CPU. Requires Open CV compiled with CUDA enables and Nvidia drivers set up correctly.')
    parser.add_argument('--no-images', action='store_true',
                        help='Do not create images/frames of detected people.')
    parser.add_argument('--debug-amount', type=int, default=-1, help='Only examine first n files.')
    parser.add_argument('--img-only', action='store_true', help='Only examine image files.')
    parser.add_argument('--vid-only', action='store_true', help='Only examine video files.')

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
    log_dir = '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
    os.mkdir(log_dir)

    print('Beginning Detection')
    print(f'Directory {log_dir} has been created')
    print(f"Confidence threshold set to {args['confidence']}%")
    print(f'Examining every {every_nth_frame} frames.')
    print('Not creating debug images' if args['no_images'] else 'Creating debug images')
    print(f"Debug amount is set to {args['debug_amount']}")
    print(f"GPU is set to {args['gpu']}")
    print('\n')

    # open a log file and loop over all our video files
    if args['f'] == '':
        media_directory_list = list_dir(args['directory'] + '/', img_only=args['img_only'], vid_only=args['vid_only'])
    else:
        media_directory_list = [args['f']]

    # what file we are on
    working_on_counter = 1

    # bytes counters
    total_bytes = 0
    bytes_to_delete = 0

    # file dictionary
    detection = {True: [], False: []}

    for media_file in media_directory_list:
        media_file_size = os.path.getsize(media_file)
        total_bytes += media_file_size
        print(f'Examining {media_file}: {working_on_counter} of {len(media_directory_list)} '
              f'({human_size(media_file_size)})')

        # check for people
        human_detected, error_detected = person_checker(str(media_file), log_dir, yolo=yolo_string,
                                                        nth_frame=every_nth_frame, confidence=confidence_percent,
                                                        gpu=gpu_flag, no_images=args['no_images'])

        if not human_detected:
            bytes_to_delete += os.path.getsize(media_file)

        if error_detected:
            print(f'\nError in analyzing {media_file}')

        if working_on_counter == args['debug_amount']:
            print(f'Debug amount of {working_on_counter} files reached')
            break

        working_on_counter += 1
        print('\n')

    print(f'Total file size without people detected: {human_size(bytes_to_delete)}/{human_size(total_bytes)}\n')
    print('Next action:\n\t[0] do nothing'
          '\n\t[1] delete all files with a person detected'
          '\n\t[2] delete all files with a person detected and still in debug dir'
          '\n\t[3] delete all files with no person detected'
          '\n\t[4] delete all files with no person detected and still in debug dir')
    action = input('> ')
    print(f'Chosen action {action}')

