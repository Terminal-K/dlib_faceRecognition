import cv2 as cv
import os
import numpy as np
import csv
import config
from tqdm import tqdm
import shutil

def write2csv(data, mode):                      # 更新库内所有数据
    with open(config.csv_base_path, mode, newline='') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(data)

def get_features_from_csv():
    features_in_csv = []
    with open(config.csv_base_path, 'r') as rf:
        csv_reader = csv.reader(rf)
        for row in csv_reader:
            for i in range(0, 128):
                row[i] = float(row[i])

            features_in_csv.append(row)
        return features_in_csv

def save_select_in_csv(data):                   # 库内数据选择更新
    features_in_csv = get_features_from_csv()

    with open(config.csv_base_path, 'w', newline='') as wf:
        csv_writer = csv.writer(wf)
        for index, i in enumerate(config.face_needTo_update):
            features_in_csv[i] = data[index]

        csv_writer.writerow(features_in_csv[0])

    with open(config.csv_base_path, 'a+', newline='') as af:
        csv_writer = csv.writer(af)
        for j in range(1, len(features_in_csv)):
            csv_writer.writerow(features_in_csv[j])

    print("csv文件更新完成!!")

def get_128_features(person):                    # person代表第几个人脸数据文件夹
    num = 0
    features = []
    imgs_folder = config.imgs_folder_path[person]
    points_faceImage_path = config.points_faceData_path + imgs_folder

    imgs_path = config.faceData_path + imgs_folder + '/'
    list_imgs = os.listdir(imgs_path)
    imgs_num = len(list_imgs)

    if os.path.exists(config.points_faceData_path + imgs_folder):
        shutil.rmtree(points_faceImage_path)
    os.makedirs(points_faceImage_path)
    print("人脸点图文件夹建立成功!!")

    with tqdm(total=imgs_num) as pbar:
        pbar.set_description(str(imgs_folder))
        for j in range(imgs_num):
            image = cv.imread(os.path.join(imgs_path, list_imgs[j]))

            faces = config.detector(image, 1)           # 经查阅资料，这里的1代表采样次数
            if len(faces) != 0:
                for z, face in enumerate(faces):
                    shape = config.predictor(image, face)       # 获取68点的坐标

                    w, h = (face.right() - face.left()), (face.bottom() - face.top())
                    left, right, top, bottom = face.left() - w // 4, face.right() + w // 4, face.top() - h // 2, face.bottom() + h // 4
                    im = image

                    cv.rectangle(im, (left, top), (right, bottom), (0, 0, 255))
                    cv.imwrite(points_faceImage_path + '/{}.png'.format(j), im)

                    if config.get_points_faceData_flag == True:
                        for p in range(0, 68):
                            cv.circle(image, (shape.part(p).x, shape.part(p).y), 2, (0,0,255))
                        cv.imwrite(points_faceImage_path + '/{}.png'.format(j), image)

                    the_features = list(config.recognition_model.compute_face_descriptor(image, shape)) # 获取128维特征向量
                    features.append(the_features)
                    #print("第{}张图片，第{}张脸,特征向量为:{}".format(j+1, z+1, the_features))
                    num += 1
            pbar.update(1)

    np_f = np.array(features)
    #res = np.mean(np_f, axis=0)
    res = np.median(np_f, axis=0)

    return res

def main():
    if config.import_all_features_flag == True:
        res = get_128_features(person=0)
        write2csv(res, 'w')
        for i in range(1, config.num_of_person_in_lib):
            res = get_128_features(person=i)
            write2csv(res, 'a+')
            #print("人脸特征向量为：{}".format(res))
    else:
        select_res = []
        for i in config.face_needTo_update:
            res = get_128_features(person=i)
            select_res.append(res)
        save_select_in_csv(select_res)

if __name__ == '__main__':
    main()
