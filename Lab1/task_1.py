import cv2
import numpy as np
from collections import deque


def find_first_and_last_indices(arr, value):
    first_index = -1
    last_index = -1

    for idx in range(len(arr)):
        if arr[idx] == value:
            if first_index == -1:
                first_index = idx
            last_index = idx


    if first_index == -1:
        return (None, None)
    else:
        return (first_index, last_index)


def bfs_maze_solver(maze, start, goal):
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for direction in directions:
            next_position = (current[0] + direction[0], current[1] + direction[1])

            if next_position == goal:
                print("Path found!")
                return path + [next_position]


            if (0 <= next_position[0] < len(maze) and
                    0 <= next_position[1] < len(maze[0]) and
                    maze[next_position[0]][next_position[1]] != 0 and
                    next_position not in visited):
                queue.append((next_position, path + [next_position]))
                visited.add(next_position)

    return None


def find_way_from_maze(image: np.ndarray) -> tuple:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

    top_row_left, top_row_right = find_first_and_last_indices(binarized_img[0], 255)
    start_point = (0, (top_row_left + top_row_right) // 2)

    bottom_row_left, bottom_row_right = find_first_and_last_indices(binarized_img[-1], 255)
    end_point = (binarized_img.shape[0] - 1, (bottom_row_left + bottom_row_right) // 2)

    path = bfs_maze_solver(binarized_img, start_point, end_point)

    if path is None:
        print("No path found.")
        return np.array([]), np.array([])

    x_coords, y_coords = zip(*path)
    return np.array(x_coords), np.array(y_coords)