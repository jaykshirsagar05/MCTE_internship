import mtcnn as mt
from imutils.video import VideoStream
from keras.applications.resnet50 import preprocess_input
from numpy import asarray
from PIL import Image
import numpy as np
import time
import cv2
import os
import face_recognition
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from buildvgg import vgg_face
import bloc as bloc

median = []
detector = mt.MTCNN()


def create_dataset(roll):  # name will act as a label.
    total = 0
    print("Starting video stream")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    temp = r"C:\Users\Jay Kshirsagar\Desktop\MCTE\My_module\Dataset" + '/' + roll
    os.makedirs(temp)
    mean_encod = []
    while True:
        frame = vs.read()
        orig = frame.copy()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("k"):

            p = os.path.sep.join([temp, "{}_{}.png".format(roll, str(total))])
            print(p)
            image = frame
            pixels = asarray(image)
            results = detector.detect_faces(pixels)

            try:
                x1, y1, width, height = results[0]['box']
            except IndexError as error:
                print('FACE NOT FOUND')

            x1, y1 = abs(x1), abs(y1)

            x2, y2 = x1 + width, y1 + height

            face = pixels[y1:y2, x1:x2]
            cv2.imshow("face", face)
            cv2.imwrite(p, face)  # image stored in directory.

            image = Image.fromarray(face)
            image = image.resize((224, 224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            img_encod = vgg_face(image)
            print("encodings {}".format(img_encod))
            mean_encod.append(img_encod)
            # print(face_array)
            total += 1

        elif key == ord("q"):
            print("all encods {}".format(mean_encod))
            mean_encod = np.array(mean_encod)
            media = np.median(mean_encod, axis=0)  # send this median to database with labels as name
            global median
            median = np.array(media)
            bloc.add_feature(roll, median)
            print("median {}".format(median))
            break

    print("Total {} faces of {} stored".format(total, roll))
    cv2.destroyAllWindows()
    vs.stop()


def compare(path=r'C:\Users\Jay Kshirsagar\Desktop\poli.mp4'):
    bloc.set_date()
    print(median)
    # detector = mt.MTCNN()
    # vs = VideoStream(path).start()
    vs = cv2.VideoCapture(path)
    time.sleep(2.0)
    frame_cnt = 0
    while True:
        ret, frame = vs.read()
        frame_cnt += 1
        if ret:
            if frame_cnt % 5 != 0:
                continue
            cv2.imshow("video", frame)
            # cv2.waitKey(1)
            image = frame
            pixels = asarray(image)
            results = detector.detect_faces(pixels)
            for i in range(len(results)):
                try:
                    x1, y1, width, height = results[i]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    face = pixels[y1:y2, x1:x2]

                    image = Image.fromarray(face)
                    image = image.resize((224, 224))
                    image = np.expand_dims(image, axis=0)
                    image = preprocess_input(image)
                    img_encoding = vgg_face(image)
                    img_encoding = np.array(img_encoding)
                    img_detected = img_encoding
                    # print("ima_encod" + str(img_encod))
                    feature_list = bloc.get_all_features()
                    dist = []
                    for j in feature_list:
                        dist.append(np.linalg.norm(img_detected - j))
                    if min(dist) <= 85.0:
                        index = dist.index(min(dist))
                        # print("Index" + str(index))
                        naam = bloc.get_roll_from_feature(feature_list[index])
                        bloc.changekey(naam)
                    else:
                        naam = 'unknown'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                    cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    text = f"face: " + naam
                    cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
                    cv2.imshow("video", frame)
                    # cv2.waitKey(1)
                except IndexError as error:
                    print('FACE NOT FOUND')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

        # print(min(dist))
        # if cv2.waitKey(1) & 0xFF == ord('q'):

    # print("distance {}".format(dist))
    cv2.destroyAllWindows()
    vs.release()


def create_ds(name, rn):
    train_images = list()

    image_path = r"C:\Users\Jay Kshirsagar\Desktop\MCTE\My_module\Dataset" + '/' + name
    for image in os.walk(image_path):
        train_images.append(image[2])

    print(train_images)
    mean_encod = []
    for i in train_images[0]:

        path = image_path + '/' + str(i)
        print(path)
        data = pyplot.imread(path)
        pyplot.imshow(data)
        ax = pyplot.gca()
        # cv2.imshow('modi', image_path+str(i))
        # image = cv2.imread(path)
        pixels = data
        results = detector.detect_faces(pixels)
        # cv2.imshow('modi', image)
        # time.sleep(10.0)
        try:
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            # #
            x2, y2 = x1 + width, y1 + height
            # #
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
            # pyplot.show()
            face = pixels[y1:y2, x1:x2]
            # cv2.imshow("face", face)
            # cv2.imwrite(p, face)  # image stored in directory.
            #
            image = Image.fromarray(face)
            image = image.resize((224, 224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            img_encod = vgg_face(image)
            print("encodings {}".format(img_encod))
            mean_encod.append(img_encod)
        except IndexError as error:
            print('FACE NOT FOUND')
        # #
        #

    print("all encods {}".format(mean_encod))
    mean_encod = np.array(mean_encod)
    media = np.median(mean_encod, axis=0)  # send this median to database with labels as name
    # global median
    # median = np.array(media)
    print(str(rn))
    bloc.add_feature(str(rn), media)
    print("median {}".format(media))


def compare_1(path):
    input_video = cv2.VideoCapture(path)
    time.sleep(2.0)
    # length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    known_faces = bloc.get_all_features()

    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    while True:
        ret, frame = input_video.read()
        cv2.imshow('video', frame)
        frame_number += 1

        pixels = asarray(frame)
        results = detector.detect_faces(pixels)

        for i in range(len(results)):
            x, y, width, height = results[i]['box']
            x1 = x + width
            y1 = y + width
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            # cv2.imshow('video', frame)

            # cv2.waitKey()
            face = pixels[y:y1, x:x1]

            image = Image.fromarray(face)
            image = image.resize((224, 224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            encoding = vgg_face(image)
            match = face_recognition.compare_faces(known_faces, encoding, 0.6)
            for i in match:
                if match[i]:
                    naam = bloc.get_roll_from_feature(known_faces[i])
            cv2.rectangle(frame, (x, y - 20), (x1, y1), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"face: " + naam
            cv2.putText(frame, text, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
            cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(frame_number)


if __name__ == '__main__':
    #     # print("[INFO] first creating student")
    #     # bloc.add_student(str(input('Enter name:')), str(input('Enter roll no:')))
    #     # print("[INFO] Saving facial features")
    #     # create_dataset(input('Enter name:'))
    #     # print(type(median))
    #     # compare()
    #     # create_ds(input('Enter name:'), input('Enter roll_no:'))
    compare(r'C:\Users\Jay Kshirsagar\Desktop\poli.mp4')
