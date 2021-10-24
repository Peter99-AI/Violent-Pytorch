import os
import cv2
import split_data

outpath = r"img"

path_videos = ['video/fight', 'video/nofight']
path_imgs = [['/train/fight/', '/valid/fight/'], ['/train/nofight/', '/valid/nofight/']]

for ind, path in enumerate(path_videos):
    for i in os.listdir(path):
        if "avi" in i or "mp4" in i:
            imgs = []
            count = 0

            cap = cv2.VideoCapture(os.path.join(path, i))
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if count % 5 == 0:
                    imgs.append(frame)

                count += 1

            while len(imgs) % 15 != 0:
                imgs.pop(-1)

            for (j, k) in enumerate(imgs):
                jj = "{:=06d}".format(j)

                file_name = os.path.join(outpath, i.split(".")[0] + "_" + jj + ".png")
                cv2.imwrite(file_name, k)
        if len(os.listdir("img")) > 15000:
            break
    print(ind)
    print(path_imgs[ind][0])
    print(path_imgs[ind][1])
    split_data.split_data(path_imgs[ind][0], path_imgs[ind][1])

