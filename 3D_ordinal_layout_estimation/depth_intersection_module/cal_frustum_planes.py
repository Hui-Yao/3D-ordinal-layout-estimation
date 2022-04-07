import numpy as np


def calc_frustum_planes(h, w, K):
    '''
    Calculate frustum planes
    :return: A list of frustum planes
    '''

    K_inv = np.linalg.inv(K)
    c1 = np.array([0, 0, 1])      # c1------c2
    c2 = np.array([w - 1, 0, 1])  # |        |
    c3 = np.array([0, h - 1, 1])  # c3------c4
    c4 = np.array([w - 1, h - 1, 1])

    v1 = K_inv.dot(c1)
    v2 = K_inv.dot(c2)
    v3 = K_inv.dot(c3)
    v4 = K_inv.dot(c4)
    n12 = np.cross(v1, v2)  # 这里计算的是abc
    n12 = n12 / np.sqrt(n12[0] ** 2 + n12[1] ** 2 + n12[2] ** 2)    # shang
    n13 = -np.cross(v1, v3)
    n13 = n13 / np.sqrt(n13[0] ** 2 + n13[1] ** 2 + n13[2] ** 2)    # zuo
    n24 = -np.cross(v2, v4)
    n24 = n24 / np.sqrt(n24[0] ** 2 + n24[1] ** 2 + n24[2] ** 2)    # you
    n34 = -np.cross(v3, v4)
    n34 = n34 / np.sqrt(n34[0] ** 2 + n34[1] ** 2 + n34[2] ** 2)    # xia
    plane1 = np.concatenate(([-1], n12, [0]))     # 这里拼接上的是d
    plane2 = np.concatenate(([-2], n24, [0]))
    plane3 = np.concatenate(([-3], n34, [0]))
    plane4 = np.concatenate(([-4], n13, [0]))
    frustum_planes = [plane1, plane2, plane3, plane4]
    frustum_planes = np.array(frustum_planes)

    return frustum_planes