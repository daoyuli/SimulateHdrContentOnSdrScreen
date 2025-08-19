import numpy as np
import matplotlib.pyplot as plt

# sRGB / Rec.709
sRGB_xy = {
    'red':   (0.6400, 0.3300),
    'green': (0.3000, 0.6000),
    'blue':  (0.1500, 0.0600),
    'white': (0.3127, 0.3290)  # D65
}

# Display P3
display_p3_xy = {
    'red':   (0.6800, 0.3200),
    'green': (0.2650, 0.6900),
    'blue':  (0.1500, 0.0600),
    'white': (0.3127, 0.3290)  # D65
}

# BT.2020
bt2020_xy = {
    'red':   (0.7080, 0.2920),
    'green': (0.1700, 0.7970),
    'blue':  (0.1310, 0.0460),
    'white': (0.3127, 0.3290)  # D65
}


def xyToXYZ(xy):
    """将 xy 色度值转换为 XYZ"""
    x, y = xy
    if y == 0:
        return (0., 0., 0.)
    Y = 1.0
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return (X, Y, Z)


def RgbToXYZMatirx(r_xy, g_xy, b_xy, w_xy):
    xr, yr = r_xy
    xg, yg = g_xy
    xb, yb = b_xy
    Xr, Yr, Zr = xyToXYZ((xr, yr))
    Xg, Yg, Zg = xyToXYZ((xg, yg))
    Xb, Yb, Zb = xyToXYZ((xb, yb))

    M = np.array([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ])

    W = np.array(xyToXYZ(w_xy))
    S = np.linalg.solve(M, W)

    Sr, Sg, Sb = S
    return np.array([
        [Xr * Sr, Xg * Sg, Xb * Sb],
        [Yr * Sr, Yg * Sg, Yb * Sb],
        [Zr * Sr, Zg * Sg, Zb * Sb]
    ])


def RgbToXYZ(rgb, gamut):
    if gamut == 'sRGB':
        gamut_basis = sRGB_xy
    elif gamut == 'Display P3':
        gamut_basis = display_p3_xy
    elif gamut == 'BT.2020':
        gamut_basis = bt2020_xy
    else:
        print('Unknown gamut, use sRGB as default')
        gamut_basis = sRGB_xy
    r_xy = gamut_basis['red']
    g_xy = gamut_basis['green']
    b_xy = gamut_basis['blue']
    w_xy = gamut_basis['white']
    matrix = RgbToXYZMatirx(r_xy, g_xy, b_xy, w_xy)
    rgb = np.asarray(rgb)
    xyz = rgb @ matrix.T
    return xyz


def XYZToRgb(xyz, gamut):
    if gamut == 'sRGB':
        gamut_basis = sRGB_xy
    elif gamut == 'Display P3':
        gamut_basis = display_p3_xy
    elif gamut == 'BT.2020':
        gamut_basis = bt2020_xy
    else:
        print('Unknown gamut, use sRGB as default')
        gamut_basis = sRGB_xy
    r_xy = gamut_basis['red']
    g_xy = gamut_basis['green']
    b_xy = gamut_basis['blue']
    w_xy = gamut_basis['white']
    matrix = RgbToXYZMatirx(r_xy, g_xy, b_xy, w_xy)
    inv_matrix = np.linalg.inv(matrix)
    xyz = np.asarray(xyz)
    rgb = xyz @ inv_matrix.T
    return rgb


def RgbToRgb(rgb, src_gamut, dst_gamut):
    # return XYZToRgb(RgbToXYZ(rgb, src_gamut), dst_gamut)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    if src_gamut == 'BT.2020' and dst_gamut == "sRGB":
        matrix = [
            [1.660, -0.5876, -0.0728],
            [-0.1245, 1.1329, 0.0083],
            [-0.0181, -0.1006, 1.1187]
        ]
    elif src_gamut == "BT.2020" and dst_gamut == "Display P3":
        matrix = [

        ]
    elif src_gamut == "Display P3" and dst_gamut == "sRGB":
        matrix = [

        ]
    elif src_gamut == dst_gamut:
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    r_out = matrix[0][0]*r + matrix[0][1]*g + matrix[0][2]*b
    g_out = matrix[1][0]*r + matrix[1][1]*g + matrix[1][2]*b
    b_out = matrix[2][0]*r + matrix[2][1]*g + matrix[2][2]*b
    return np.dstack((r_out, g_out, b_out))

def RgbToLuma(rgb, gamut):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    if gamut == 'sRGB':
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif gamut == 'Display P3':
        y = 0.22897 * r + 0.69174 * g + 0.07929 * b
    elif gamut == 'BT.2020':
        y = 0.2627 * r + 0.6780 * g + 0.0593 * b
    else:
        raise ValueError(f"Unsupported: {gamut}.")
    return y


def PlotColorGamut():
    gamuts = {
        'sRGB': sRGB_xy,
        'Display P3': display_p3_xy,
        'BT.2020': bt2020_xy
    }

    plt.figure(figsize=(8, 8))
    for name, cs in gamuts.items():
        r = cs['red']
        g = cs['green']
        b = cs['blue']

        plt.plot([r[0], g[0], b[0], r[0]],
                 [r[1], g[1], b[1], r[1]], label=name)

    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Color Gamut Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    rgb = (1.0, 0.0, 0.0)

    print("原始 RGB:", rgb)

    # sRGB -> XYZ
    xyz = RgbToXYZ(rgb, 'sRGB')
    print("sRGB -> XYZ:", xyz)

    # XYZ -> Display P3
    p3_rgb = XYZToRgb(xyz, 'Display P3')
    print("XYZ -> Display P3:", p3_rgb)

    # sRGB -> Display P3
    converted = RgbToRgb(rgb, 'sRGB', 'Display P3')
    print("sRGB -> Display P3:", converted)

    # Display P3 -> BT.2020
    bt2020_rgb = RgbToRgb(converted, 'Display P3', 'BT.2020')
    print("Display P3 -> BT.2020:", bt2020_rgb)

    # BT.2020 -> XYZ
    xyz_bt2020 = RgbToXYZ(bt2020_rgb, 'BT.2020')
    print("BT.2020 -> XYZ:", xyz_bt2020)

    PlotColorGamut()