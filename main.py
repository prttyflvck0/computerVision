import numpy as np
import cv2
from collections import Counter, defaultdict

# путь к певому кадру
firstframe_path = r'firstFrames\firstFrame.jpg'

firstframe = cv2.imread(firstframe_path)
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray, (21, 21), 0)

# ---------------------------------
# ресайз окон вручную
# ---------------------------------
cv2.namedWindow('CannyEdgeDet', cv2.WINDOW_NORMAL)
cv2.namedWindow('Abandoned Object Detection', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Morph_CLOSE', cv2.WINDOW_NORMAL)

# путь к видео
file_path = r'videos\video2.mp4'

cap = cv2.VideoCapture(file_path)

consecutiveframe = 20 # на скольких кадрах должен оставаться объект

track_temp = []
track_master = []
track_temp2 = []

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)

frameno = 0

try:
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('main', frame)

        if ret == 0:
            break

        frameno = frameno + 1
        cv2.putText(frame,'%s%.f'%('Frameno:',frameno), (400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

        # RGB -> оттенки серого
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # блюр по гауссу
        frame_blur = cv2.GaussianBlur(frame_gray, (15, 15), 0)
        # абсолютная разница между первым и текущим кадром
        frame_diff = cv2.absdiff(firstframe, frame)

        # Canny Edge Detection
        edged = cv2.Canny(frame_diff, 160, 200)
        cv2.imshow('CannyEdgeDet', edged)
        kernel2 = np.ones((3, 3), np.uint8)  # чем больше ядро-kernel, например (10,10), тем больше будет значение eroded или dilated
        thresh2 = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel2, iterations=2)
        cv2.imshow('Morph_Close', thresh2)

        # создаем копию для нахождения контуров
        cnts, hierarchy = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(cnts)

        mycnts = []

        for c in cnts:

            # координаты центроида(центра фигуры)
            M = cv2.moments(c)
            if M['m00'] == 0:
                pass
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # ----------------------------------------------------------------
                # критерии размеров контура
                # ----------------------------------------------------------------

                if cv2.contourArea(c) < 300 or cv2.contourArea(c) > 20000:
                    pass
                else:
                    mycnts.append(c)

                    #координаты баундинг бокса,
                    # обновление текста
                    (x, y, w, h) = cv2.boundingRect(c)

                    #---------------------------------------------------------
                    # вывод всех найденных контуров с координатами
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(frame,'C %s,%s,%.0f'%(cx,cy,cx+cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
                    # ---------------------------------------------------------

                    sumcxcy = cx + cy
                    track_master.append([sumcxcy, frameno])
                    countuniqueframe = set(j for i, j in track_master)

                    if len(countuniqueframe) > consecutiveframe or False:
                        minframeno = min(j for i, j in track_master) # самый старый кадр
                        for i, j in track_master:
                            if j != minframeno:
                                track_temp2.append([i, j])

                        track_master = list(track_temp2)
                        track_temp2 = []



                    countcxcy = Counter(i for i, j in track_master)
                    # print countcxcy
                    # если тот же sumcxcy появляется на всех кадрах, сохраняем его в конечный список контуров, добавляем 1
                    for i, j in countcxcy.items():
                        if j >= consecutiveframe:
                            top_contour_dict[i] += 1

                    if sumcxcy in top_contour_dict:
                        if top_contour_dict[sumcxcy] > 100:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                            cv2.putText(frame, '%s' % ('Abandoned Obj'), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255), 2)
                            print('Detected : ', frameno)

        cv2.imshow('Abandoned Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except:
    print("Видео закончилось")

cap.release()
cv2.destroyAllWindows()
