import cv2 as cv
import time
import os
import config

class face_detect():
    def __init__(self):
        self.start_time = 0                     # 用于计算帧率
        self.fps = 0                            # 帧率

        self.image = None
        self.face_img = None

        self.face_num = 0                       # 这一帧的人脸个数
        self.last_face_num = 0                  # 上一帧的人脸个数

        self.face_num_change_flag = False       # 当前帧人脸数量变化的标志位，用于后续人脸识别提高帧率
        self.quit_flag = False                  # 退出程序标志位
        self.buildNewFolder = False             # 按下"n"新建文件夹标志位
        self.save_flag = False                  # 按下“s”保存人脸数据标志位
        self.face_flag = False                  # 人脸检测标志位

        self.img_num = 0                        # 人脸数据文件夹内的图像个数

        self.collect_face_data = True           # 是否进行人脸数据的采集，只有为真时才会进行采集

    def get_fps(self):
        now = time.time()
        time_period = now - self.start_time
        self.fps = 1.0 / time_period
        self.start_time = now
        color = (0,255,0)
        if self.fps < 15:
            color = (0,0,255)
        cv.putText(self.image, str(self.fps.__round__(2)), (20, 50), cv.FONT_HERSHEY_DUPLEX, 1, color)

    def key_scan(self, key):
        if self.collect_face_data == True:
            if self.save_flag == True and self.buildNewFolder == True:
                if self.face_img.size > 0:
                    cv.imwrite(
                        config.faceData_path + 'person_{}/{}.png'.format(config.num_of_person_in_lib - 1, self.img_num),
                        self.face_img)
                    self.img_num += 1

            if key == ord('s'):
                self.save_flag = not self.save_flag

            if key == ord('n'):
                os.makedirs(config.faceData_path + 'person_{}'.format(config.num_of_person_in_lib))
                config.num_of_person_in_lib += 1
                print("新文件夹建立成功!!")
                self.buildNewFolder = True
        if key == ord('q'): self.quit_flag = True

    def face_detecting(self):
        face_location = []
        all_face_location = []

        faces = config.detector(self.image, 0)
        self.face_num = len(faces)

        if self.face_num != self.last_face_num:
            self.face_num_change_flag = True
            print("脸数改变，由{}张变为{}张".format(self.last_face_num, self.face_num))
            self.check_times = 0
            self.last_face_num = self.face_num
        else:
            self.face_num_change_flag = False

        if len(faces) != 0:
            self.face_flag = True

            for i, face in enumerate(faces):
                face_location.append(face)
                w, h = (face.right() - face.left()), (face.bottom() - face.top())
                left, right, top, bottom = face.left() - w//4, face.right() + w//4, face.top() - h//2, face.bottom() + h//4

                all_face_location.append([left, right, top, bottom])

            return face_location, all_face_location
        else:
            self.face_flag = False

        return None

    def show(self, camera):
        while camera.isOpened() and not self.quit_flag:
            val, self.image = camera.read()
            if val == False: continue

            key = cv.waitKey(1)

            res = self.face_detecting()
            if res is not None:
                _, all_face_location = res

                for i in range(self.face_num):
                    [left, right, top, bottom] = all_face_location[i]
                    self.face_img = self.image[top:bottom, left:right]
                    cv.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255))

                    if self.collect_face_data == True:
                        cv.putText(self.image, "Face", (int((left + right) / 2) - 50, bottom + 20), cv.FONT_HERSHEY_COMPLEX, 1,
                                   (255, 255, 255))

                self.key_scan(key)

            self.get_fps()

            cv.namedWindow('camera', 0)
            cv.imshow('camera', self.image)

        camera.release()
        cv.destroyAllWindows()

def main():
    try:
        cam = cv.VideoCapture(0)
        face_detect().show(cam)
    finally:
        cam.release()
        cv.destroyAllWindows()
        print("程序退出!!")

if __name__ == '__main__':
    main()