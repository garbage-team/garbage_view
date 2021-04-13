def depth_volume(depth):


    r=(38, 32, 146, 124)
    depth=depth[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    return np.sum(depth)/np.sum(depth.size)