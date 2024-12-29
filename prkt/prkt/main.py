import os
import cv2
import numpy as np
import hydra

from omegaconf import DictConfig
from utils import *

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Тестовое видео
    video_path = cfg.paths.video

    # Инициализируем работу с видео
    video_capture = cv2.VideoCapture(video_path)

    # Пока не нажата клавиша q функция будет работать
    #Определяем параметры модели
    #Загружаем конфигурацию и веса модели скаченные ранее
    net = cv2.dnn.readNetFromDarknet(cfg.paths.config, cfg.paths.weights)
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    #Парковочные места
    first_frame_parking_spaces = None

    free_parking_timer = 0
    free_parking_timer_bag1 = 0
    free_parking_count = 0
    first_parking_timer = 0
    free_parking_space = False
    free_parking_space_box = None
    check_det_frame = None

    #Инициализируем работу с видео
    video_capture = cv2.VideoCapture(video_path)

    #Пока не нажата клавиша q функция будет работать
    while video_capture.isOpened():
        
        ret, image_to_process = video_capture.read()

        #Препроцессинг изображения и работа YOLO
        height, width, _ = image_to_process.shape
        blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (cfg.model.input_size, cfg.model.input_size),
                                    (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_layers)
        class_indexes, class_scores, boxes = ([] for i in range(3))

        #Обнаружение объектов в кадре
        for out in outs:
            for obj in out:
                scores = obj[5:]
                class_index = np.argmax(scores)

                #В классе 2 (car) только автомобили
                if class_index == cfg.CLASS_IDX: 
                    class_score = scores[class_index]
                    if class_score > 0:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        obj_width = int(obj[2] * width)
                        obj_height = int(obj[3] * height)
                        box = [center_x - obj_width // 2, center_y - obj_height // 2,
                                obj_width, obj_height]
                        #BBoxes
                        boxes.append(box)
                        class_indexes.append(class_index)
                        class_scores.append(float(class_score))
            
            
        ###ПЕРВЫЙ КАДР ОПРЕДЕЛЯЕМ ПАРКОМЕСТА
        if not first_frame_parking_spaces:
            #Предполагаем, что под каждой машиной будет парковочное место
            first_frame_parking_spaces = boxes
            first_frame_parking_score = class_scores
            print(boxes)
        else:
            chosen_cars_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, cfg.model.confidence_threshold, cfg.model.nms_threshold)
            cars_area = []

            ###МАШИНЫ
            for box_index in chosen_cars_boxes:
                car_box = boxes[box_index]
                cars_area.append(car_box)

                x, y, w, h = car_box
                parking_text = 'Car'
                final_image = draw_bbox(image_to_process, x, y, w, h, parking_text, (255, 255, 0))
                
            #Теперь зная парковочные места, определим когда место освободится
            cars_boxes = cars_area        
            
            ###IoU     
            overlaps = compute_overlaps(np.array(parking_spaces), np.array(cars_boxes))
            
            for parking_space_one, area_overlap in zip(parking_spaces, overlaps):
                
                max_IoU = max(area_overlap)
                sort_IoU = np.sort(area_overlap[area_overlap > 0])[::-1]      
                
                if free_parking_space == False:
                    
                    if 0.0 < max_IoU < cfg.parking.iou_threshold_free:

                        #Количество паркомест по условию 1: 0.0 < IoU < 0.4
                        len_sort = len(sort_IoU)

                        #Количество паркомест по условию 2: IoU > 0.15
                        sort_IoU_2 = sort_IoU[sort_IoU > 0.15]
                        len_sort_2 = len(sort_IoU_2)

                        #Смотрим чтобы удовлятворяло условию 1 и условию 2
                        if (check_det_frame == parking_space_one) & (len_sort != len_sort_2):
                            #Начинаем считать кадры подряд с пустыми координатами
                            free_parking_timer += 1

                        elif check_det_frame == None:
                            check_det_frame = parking_space_one

                        else:
                            #Фильтр от чехарды мест (если место чередуется, то "скачет")
                            free_parking_timer_bag1 += 1
                            if free_parking_timer_bag1 == 2:
                                #Обнуляем счётчик, если паркоместо "скачет"
                                check_det_frame = parking_space_one
                                free_parking_timer = 0

                        #Если более N кадров подряд, то предполагаем, что место свободно
                        if free_parking_timer == cfg.parking.free_space_timer:
                            #Помечаем свободное место
                            free_parking_space = True
                            free_parking_space_box = parking_space_one
                            #Отрисовываем рамку парковочного места 
                            x_free, y_free, w_free, h_free = parking_space_one
                            
                else:
                    #Если место занимают, то помечается как отсутствие свободных мест
                    overlaps = compute_overlaps(np.array([free_parking_space_box]), np.array(cars_boxes))
                    for area_overlap in overlaps:                
                        max_IoU = max(area_overlap)
                        if max_IoU > cfg.parking.iou_threshold_occupied:
                            free_parking_space = False             
        
        ###ПАРКОВОЧНЫЕ МЕСТА
        chosen_boxes = cv2.dnn.NMSBoxes(first_frame_parking_spaces, 
                                        first_frame_parking_score, cfg.model.confidence_threshold, cfg.model.nms_threshold)
        parking_spaces = []
        
        for box_index in chosen_boxes:
            box = first_frame_parking_spaces[box_index]
            
            #Если определилось пустое место, то отрисуем его в кадре
            if free_parking_space:
                if box == [x_free, y_free, w_free, h_free]:
                    parking_text = 'FREE SPACE!!!'
                    final_image = draw_bbox(image_to_process, x_free, y_free, w_free, h_free, parking_text, (0, 0, 255))
                else:
                    x, y, w, h = box
                    parking_text = 'No parking'
                    final_image = draw_bbox(image_to_process, x, y, w, h, parking_text)
                
            else:
                #Координаты и размеры BB
                x, y, w, h = box
                parking_text = 'No parking'
                final_image = draw_bbox(image_to_process, x, y, w, h, parking_text)

            #Координаты парковочных мест с первого кадры
            parking_spaces.append(box)
        #Показать результат работы
        cv2.imshow("Parking Space", final_image)

        #Прерывание работы клавишей q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #Очищаем всё после завершения.
            video_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    import os
    print("Текущий рабочий каталог:", os.path.isdir('/Users/nikita/study/python/prkt/prkt/prkt/config'))
    main()
