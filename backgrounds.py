import random
import numpy as np


def rand_background(shape):
    rand_int = random.randint(0, 7)
    if rand_int == 0:
        return noise(shape)
    elif rand_int == 1:
        return white(shape)
    elif rand_int == 2:
        return checker(shape)
    elif rand_int == 3:
        return horz_stripe(shape)
    elif rand_int == 4:
        return vert_stripe(shape)
    elif rand_int == 5:
        return gradient_vert(shape)
    elif rand_int == 6:
        return gradient_horz(shape)
    elif rand_int == 7:
        return triangles(shape)

def noise(shape):
    return np.random.randint(0, 256, shape).astype(np.uint8)
def white(shape):
    return np.full(shape, 255).astype(np.uint8) * 255
def checker(shape):
    checker_height = (shape[0] // 8) + 1
    checker_width = (shape[1] // 8) + 1
    white_tile = np.full((checker_height, checker_width), 255)
    black_tile = np.zeros((checker_height, checker_width))
    checker_half_1 = np.concatenate([black_tile, white_tile], axis=0)
    checker_half_2 = np.concatenate([white_tile, black_tile], axis=0)
    complete_checker = np.concatenate([checker_half_1, checker_half_2], axis=1)
    checker_board = np.tile(complete_checker, (4,4))
    return checker_board[:shape[0],:shape[1]].astype(np.uint8) * 255
def horz_stripe(shape):
    indicies = np.indices(shape)[0]
    return ((indicies % 2) ).astype(np.uint8) * (255*255) 
def vert_stripe(shape):
    indicies = np.indices(shape)[1]
    return ((indicies % 2) * 255).astype(np.uint8) * 255
def gradient_vert(shape):
    indicies = np.indices(shape)[0]
    indicies = indicies / np.max(indicies)
    return (indicies * 255).astype(np.uint8)
def gradient_horz(shape):
    indicies = np.indices(shape)[1]
    indicies = indicies / np.max(indicies)
    return (indicies * 255).astype(np.uint8)
def triangles(shape):
    tile_height = (shape[0] // 8) + 1
    tile_width = (shape[1] // 8) + 1
    tile = np.zeros((tile_height, tile_width))
    indices = np.indices((tile_height, tile_width))
    tile[indices[0] > indices[1]] = 255
    full_board = np.tile(tile, (8,8))
    return full_board[:shape[0],:shape[1]].astype(np.uint8) * 255
