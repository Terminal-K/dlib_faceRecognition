import dlib
import os

# 各路径
shape_predictor_path = 'data_dlib/shape_predictor_68_face_landmarks.dat'
recognition_model_path = 'data_dlib/dlib_face_recognition_resnet_model_v1.dat'
csv_base_path = 'data/csv/features.csv'                                             # 存储人脸特征的csv路径
faceData_path = 'data/faceData/'                                                    # 存放各人脸图像的文件夹
points_faceData_path = 'data/faceData_points/'                                      # 存放人脸点图的文件夹
faceName_path = 'data/faceName.txt'                                                 # 存放人脸名字的txt

imgs_folder_path = os.listdir(faceData_path)                                        # 所有人的人脸文件夹路径

# 各标志位
get_points_faceData_flag = True                                                     # 是否获取人脸点图

import_all_features_flag = True                                                     # 是否更新库内所有数据
face_needTo_update = [x for x in range(3, 5)]                                       # 选择性更新时，需要更新库的人脸序号(从0开始)

num_of_person_in_lib = len(imgs_folder_path)                                        # 有多少个人的脸

recognition_threshold = 0.43                                                        # 人脸识别阈值，过小会难以识别到

detector = dlib.get_frontal_face_detector()                                         # 人脸检测器
predictor = dlib.shape_predictor(shape_predictor_path)                              # 人脸68点提取器
recognition_model = dlib.face_recognition_model_v1(recognition_model_path)          # 128特征向量提取器

