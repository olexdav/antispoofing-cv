from pytube import YouTube
import cv2
import numpy as np
import os

FOLDERS_PATH = "C:\\DataSets\\AntispoofingRare8\\"
frames_number = 20
video_ids_file = "video_ids1.txt"


def save_images(video_id="zV1zK8zRCPo"):
    try:
        yt = YouTube('https://youtu.be/' + video_id)
    except Exception as e:
        return False

    while True:
        try:
            yt.streams.filter(subtype='mp4').first().download(filename=video_id)
        except ConnectionResetError as e:
            continue
        except Exception as e:
            break
        break

    # print(yt.title)
    vidcap = cv2.VideoCapture(os.path.dirname(__file__) + "/{}.mp4".format(video_id))

    success, image = vidcap.read()
    success = True

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    mu, sigma = total_frames / 2 / 1000, 0.58
    frames_normal_distrib = (abs(np.random.normal(mu, sigma, frames_number)) * 1000).astype(int)

    try:
        os.mkdir(FOLDERS_PATH + "{}".format(video_id))

        for count in frames_normal_distrib:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
            success, image = vidcap.read()
            if not success:
                break
            cv2.imwrite(FOLDERS_PATH + "{}\\frame{}.jpg".format(video_id, count), image)
            print('Read a new frame: ', success, count)
        print("Folder Done.\n\n\n\n")
    except FileExistsError as e:
        print("Something went wrong... ", e)

    return True


def get_finished_ids():
    f_ids = set()
    with open("processed_ids.txt", 'r') as f:
        for line in f:
            f_ids.add(line.strip())
    return f_ids


def get_video_ids():
    v_ids = []
    with open(video_ids_file, 'r') as f:
        for line in f:
            v_ids.append(line.strip())
    return set(v_ids)


def rewrite_finished_ids(vid_id):
    with open("processed_ids.txt", 'a') as f:
        f.write(vid_id + '\n')


if __name__ == '__main__':
    finished_ids = get_finished_ids()
    video_ids = get_video_ids()

    for v_id in video_ids:
        print(v_id)
        if v_id not in finished_ids:
            finished_ids.add(v_id)
            rewrite_finished_ids(v_id)
            if not save_images(v_id):
                continue
            os.remove(os.path.dirname(__file__) + "/{}.mp4".format(v_id))
    print("All links are used.\nAdd new video links file in current folder.")
