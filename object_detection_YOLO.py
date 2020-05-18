import tensorflow.compat.v1 as tf
import tensornets as nets
import cv2
import numpy as np
import time
from create_folder import createFolder

tf.disable_v2_behavior()

class YoloObjectDetection():

    def __init__(self):

        self.url1 = "black.mp4"
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.model = nets.YOLOv3COCO(self.inputs, nets.Darknet19)
        # model = nets.YOLOv2(inputs, nets.Darknet19)

        # frame=cv2.imread("D://pyworks//yolo//truck.jpg",1)
        self.count = 0
        self.classes = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'bike', '5': 'bus', '7': 'truck', '8': 'chair'}
        self.list_of_classes = [0, 1, 2, 3, 5, 7, 8]

    def main(self):

        with tf.Session() as sess:
            sess.run(self.model.pretrained())
            # "D://pyworks//yolo//videoplayback.mp4"
            cap = cv2.VideoCapture(self.url1)

            while (cap.isOpened()):

                ret, frame = cap.read()
                img = cv2.resize(frame, (416, 416))
                copy_img = img.copy()
                imge = np.array(img).reshape(-1, 416, 416, 3)

                start_time = time.time()
                preds = sess.run(self.model.preds, {self.inputs: self.model.preprocess(imge)})

                print("--- %s seconds ---" % (time.time() - start_time))
                boxes = self.model.get_boxes(preds, imge.shape[1:3])
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)

                cv2.resizeWindow('image', 700, 700)
                # print("--- %s seconds ---" % (time.time() - start_time))
                boxes1 = np.array(boxes)

                for j in self.list_of_classes:
                    count = 0
                    if str(j) in self.classes:
                        lab = self.classes[str(j)]
                    if len(boxes1) != 0:

                        for i in range(len(boxes1[j])):

                            box = boxes1[j][i]

                            if boxes1[j][i][4] >= 0.5:

                                count += 1
                                #crop_img = copy_img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

                                print(box)

                                #crop_img = copy_img[int(box[1]):int(box[1] + int(box[3])), int(box[0]):int(box[0] + int(box[2]))]
                                #(xLeftBottom, yLeftBottom), (xRightTop, yRightTop)

                                crop_img = copy_img[int(box[1]):int(box[3] ), int(box[0]):int(box[2])]

                                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                                cv2.putText(img, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),
                                            lineType=cv2.LINE_AA)

                                # crop_img = copy_img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
                                #crop_img = cv2.resize(crop_img, (416, 416))

                                cv2.imshow("cropped_image", crop_img)
                                cv2.waitKey(1)

                                dir = "temp/" + lab + "/"
                                createFolder(dir)
                                s1 = dir + '{}.jpg'.format(self.count)
                                self.count = self.count + 1
                                cv2.imwrite(s1, crop_img)

                    print(lab, ": ", count)

                cv2.imshow("image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

objectDetetcion = YoloObjectDetection()
objectDetetcion.main()
