import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:

    hsv_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)


    obstacle_mask = detect_obstacles(hsv_img)
    line_mask = detect_lines(hsv_img)
    car_mask = detect_cars(hsv_img)


    combined_mask = obstacle_mask | line_mask | car_mask
    img_height, img_width = combined_mask.shape

    left_clear = 0
    right_clear = 0


    for col in range(img_width):
        mid_row = img_height // 2
        if combined_mask[mid_row, col] != 0:
            if right_clear != 0:
                break
            left_clear += 1

        if combined_mask[mid_row, col] == 0 and left_clear != 0:
            right_clear += 1


    total_roads_count = round((img_height - left_clear) / (left_clear + right_clear))
    road_obstacle_status = np.zeros(total_roads_count)

    offset = int(((img_width - left_clear) / total_roads_count) // 2)
    for road_idx in range(total_roads_count):
        for row in range(img_height // 2):
            if combined_mask[row, offset + 2 * offset * road_idx] in [255, 254]:
                road_obstacle_status[road_idx] = -1
                break

    car_presence_status = np.zeros(total_roads_count)
    for road_idx in range(total_roads_count):
        for row in range(img_height // 2, img_height):
            if combined_mask[row, offset + 2 * offset * road_idx] in [255, 254]:
                car_presence_status[road_idx] = -1
                break

    current_road_index = next((i for i, status in enumerate(car_presence_status) if status == -1), -1)
    available_road_index = next((i for i, status in enumerate(road_obstacle_status) if status == 0), -1)

    if current_road_index == available_road_index:
        return -1

    return available_road_index


def detect_obstacles(hsv_img):
    return cv2.inRange(hsv_img, (1, 250, 250), (255, 255, 255))


def detect_cars(hsv_img):
    return cv2.inRange(hsv_img, (109, 0, 0), (112, 255, 255))


def detect_lines(hsv_img):
    return cv2.inRange(hsv_img, (25, 0, 0), (32, 255, 255))