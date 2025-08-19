import cv2
import numpy as np
import matplotlib.pyplot as plt


class Cam16:
    def __init__(
            self,
            XYZ_w=[95.047, 100.0, 108.883],
            LA=100,
            Y_b=20,
            D=None,
            condition='average'
    ):
        """
        :param XYZ_w: 参考白点XYZ值
        :param LA: 适应亮度(cd/m²)
        :param Y_b: 背景亮度(相对Y值)
        :param F: 周围参数
        :param c: 影响因子
        :param Nc: 色诱导因子
        :param D: 适应因子(可选)
        """
        # 定义转换矩阵
        self.M_CAT16 = np.array([
            [0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414, 0.045854],
            [-0.002079, 0.048952, 0.953127]
        ], dtype=np.float32)

        self.M_CAT16_INV = np.linalg.inv(self.M_CAT16)

        self.M_HPE = np.array([
            [0.38971, 0.68898, -0.07868],
            [-0.22981, 1.18340, 0.04641],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.M_HPE_INV = np.linalg.inv(self.M_HPE)

        self.XYZ_w = np.asarray(XYZ_w, dtype=np.float32)
        self.LA = LA
        self.Y_b = Y_b

        if condition == 'average':
            self.F = 1.0
            self.c = 0.69
            self.Nc = 1.0
        elif condition == 'dim':
            pass
        elif condition == 'dark':
            pass
        else:
            self.F = 1.0
            self.c = 0.69
            self.Nc = 1.0

        self.D = self.F * (1 - (1 / 3.6) * np.exp((-self.LA - 42) / 92)) if D is None else D

        self.M_sRGB2XYZ = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

        self.M_XYZ2sRGB = np.linalg.inv(self.M_sRGB2XYZ)

    def XYZToCIECam16(self, XYZ):
        XYZ = XYZ.astype(np.float32)

        # RGB转换
        RGB = np.dot(XYZ, self.M_CAT16.T)
        RGB_w = np.dot(self.XYZ_w, self.M_CAT16.T)
        Y_w = self.XYZ_w[1]

        # 色适应调整
        Yw_over_RGBw = np.divide(Y_w, RGB_w, where=RGB_w != 0)
        RGB_c = (self.D * Yw_over_RGBw + (1 - self.D)) * RGB

        # 转换到LMS空间
        LMS = np.dot(RGB_c, self.M_HPE.T)

        # 亮度适应计算
        k = 1 / (5 * self.LA + 1)
        F_L = 0.2 * (k ** 4) * (5 * self.LA) + 0.1 * (1 - k ** 4) ** 2 * (5 * self.LA) ** (1 / 3)

        # 非线性压缩
        LMSp = 100 * np.power((F_L * LMS) / 100, 0.42)

        # 色相计算
        a = LMSp[..., 0] - (12 / 11) * LMSp[..., 1] + (1 / 11) * LMSp[..., 2]
        b = (1 / 9) * LMSp[..., 0] + (1 / 9) * LMSp[..., 1] - (2 / 9) * LMSp[..., 2]
        h = np.degrees(np.arctan2(b, a)) % 360

        # 中间参数计算
        n = self.Y_b / Y_w
        N_bb = 0.725 * n ** -0.2
        z = 1.48 + np.sqrt(n)

        # 明度响应
        A = (2 * LMSp[..., 0] + LMSp[..., 1] + 0.05 * LMSp[..., 2] - 0.305) * N_bb

        # 参考白点处理
        LMSp_w = 100 * np.power((F_L * np.dot((self.D * Yw_over_RGBw + (1 - self.D)) * RGB_w, self.M_HPE.T)) / 100, 0.42)
        A_w = (2 * LMSp_w[0] + LMSp_w[1] + 0.05 * LMSp_w[2] - 0.305) * N_bb

        # 最终计算结果
        J = 100 * (A / A_w) ** (self.c * z)
        e_t = (np.cos(np.radians(h) + 2) + 3.8) / 4
        t = (50000 / 13) * self.Nc * N_bb * e_t * np.hypot(a, b) / (LMSp.sum(axis=-1) + 0.05 * LMSp[..., 2])
        C = t ** 0.9 * np.sqrt(J / 100) * (1.64 - 0.29 ** n) ** 0.73

        return np.dstack((J, C, h))


    def CIECam16ToXYZ(self, JCh):
        J, C, h = JCh[0], JCh[1], JCh[2]

        Y_w = self.XYZ_w[1]
        n = self.Y_b / Y_w
        z = 1.48 + np.sqrt(n)
        N_bb = 0.725 * n ** -0.2

        # 步骤1：计算明度响应A
        RGB_w = np.dot(self.XYZ_w, self.M_CAT16.T)
        A_w = (2 * (RGB_w[0] * self.D / RGB_w[0] + (1 - self.D)) * RGB_w[0] +
               (RGB_w[1] * self.D / RGB_w[1] + (1 - self.D)) * RGB_w[1] +
               0.05 * (RGB_w[2] * self.D / RGB_w[2] + (1 - self.D)) * RGB_w[2]) * N_bb
        A = (J / 100) ** (1 / (self.c * z)) * A_w

        # 步骤2：计算t参数
        e_t = (np.cos(np.radians(h) + 2) + 3.8) / 4
        t = (C / (np.sqrt(J / 100) * (1.64 - 0.29 ** n) ** 0.73)) ** (1 / 0.9)
        t /= (50000 / 13) * self.Nc * N_bb * e_t

        # 步骤3：计算色相分量a,b
        a = t * np.cos(np.radians(h))
        b = t * np.sin(np.radians(h))

        # 步骤4：计算LMSp分量
        LMSp = np.empty_like(a.shape + (3,))
        LMSp[..., 0] = (11 / 23) * a + (11 / 23) * (108 / 23) * b + (A / N_bb + 0.305) / 2.0
        LMSp[..., 1] = LMSp[..., 0] - (11 / 23) * a - (1 / 11) * b
        LMSp[..., 2] = (1 / 0.05) * (A / N_bb + 0.305 - 2 * LMSp[..., 0] - LMSp[..., 1])

        # 步骤5：逆非线性压缩
        k = 1 / (5 * self.LA + 1)
        F_L = 0.2 * (k ** 4) * (5 * self.LA) + 0.1 * (1 - k ** 4) ** 2 * (5 * self.LA) ** (1 / 3)
        LMS = np.power(LMSp / 100, 1 / 0.42) * 100 / F_L

        # 步骤6：逆HPE转换
        RGB_c = np.dot(LMS, self.M_HPE_INV.T)

        # 步骤7：逆色适应调整
        Yw_over_RGBw = Y_w / RGB_w
        RGB = RGB_c / (self.D * Yw_over_RGBw + (1 - self.D))

        # 步骤8：逆CAT16转换
        XYZ = np.dot(RGB, self.M_CAT16_INV.T)

        return XYZ

# refer to Safdar et al. Opt. Express Vol. 25, Issue 13, pp. 15131-15151 (2017)
class Jabz:
    def __init__(self):
        self.b = 1.15
        self.g = 0.66
        self.d = -0.56
        self.d0 = 1.6295499532821566e-11
        self.PQ_LUMINANCE_RANGE = 10000
        self.m1 = 2610 / 16384
        self.m2 = 2523 / 32 * 1.7  # different from real PQ curve
        self.c1 = 3424 / 4096
        self.c2 = 2413 / 128
        self.c3 = 2392 / 128

    def simulatedPQ(self, x):
        x = np.clip(x, 0, 1)
        temp = x ** self.m1
        numerator = self.c1 + self.c2 * temp
        denominator = 1 + self.c3 * temp
        base = numerator / np.where(denominator != 0, denominator, np.finfo(float).eps)
        base = np.clip(base, 0, None)
        return base ** self.m2

    def simulatedInvPQ(self, x):
        x = np.clip(x, 0, 1)
        temp = x ** (1 / self.m2)
        numerator = temp - self.c1
        denominator = self.c2 - self.c3 * temp
        base = numerator / np.where(denominator != 0, denominator, np.finfo(float).eps)
        base = np.clip(base, 0, None)
        return base ** (1 / self.m1)

    def XYZToJChz(self, XYZ):
        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
        X, Y, Z = max(0, X), max(0, Y), max(0, Z)
        X2 = self.b * X - (self.b - 1) * Z
        Y2 = self.g * Y - (self.g - 1) * X
        L = 0.41478972 * X2 + 0.579999 * Y2 + 0.0146480 * Z
        M = -0.2015100 * X2 + 1.120649 * Y2 + 0.0531008 * Z
        S = -0.0166008 * X2 + 0.264800 * Y2 + 0.6684799 * Z
        L2 = self.simulatedPQ(L/self.PQ_LUMINANCE_RANGE)
        M2 = self.simulatedPQ(M/self.PQ_LUMINANCE_RANGE)
        S2 = self.simulatedPQ(S/self.PQ_LUMINANCE_RANGE)
        Iz = 0.5 * L2 + 0.5 * M2
        Az = 3.524000 * L2 - 4.066708 * M2 + 0.542708 * S2
        Bz = 0.199076 * L2 + 1.096799 * M2 - 1.295875 * S2
        Jz = ((1 + self.d) * Iz) / (1 + self.d * Iz) - self.d0
        Cz = np.sqrt(Az ** 2 + Bz ** 2)
        hz = np.arctan2(Bz, Az)
        return np.array([Jz, Cz, hz])

    def XYZToJChzMap(self, XYZMap):
        X, Y, Z = XYZMap[:, :, 0], XYZMap[:, :, 1], XYZMap[:, :, 2]
        X, Y, Z = np.maximum(0, X), np.maximum(0, Y), np.maximum(0, Z)
        X2 = self.b * X - (self.b - 1) * Z
        Y2 = self.g * Y - (self.g - 1) * X
        L = 0.41478972 * X2 + 0.579999 * Y2 + 0.0146480 * Z
        M = -0.2015100 * X2 + 1.120649 * Y2 + 0.0531008 * Z
        S = -0.0166008 * X2 + 0.264800 * Y2 + 0.6684799 * Z
        L2 = self.simulatedPQ(L/self.PQ_LUMINANCE_RANGE)
        M2 = self.simulatedPQ(M/self.PQ_LUMINANCE_RANGE)
        S2 = self.simulatedPQ(S/self.PQ_LUMINANCE_RANGE)
        Iz = 0.5 * L2 + 0.5 * M2
        Az = 3.524000 * L2 - 4.066708 * M2 + 0.542708 * S2
        Bz = 0.199076 * L2 + 1.096799 * M2 - 1.295875 * S2
        Jz = ((1 + self.d) * Iz) / (1 + self.d * Iz) - self.d0
        Cz = np.sqrt(Az ** 2 + Bz ** 2)
        hz = np.arctan2(Bz, Az)
        return np.dstack((Jz, Cz, hz))

    def JChzToXYZ(self, JChz):
        Jz, Cz, hz = JChz[0], JChz[1], JChz[2]
        Az = Cz * np.cos(hz)
        Bz = Cz * np.sin(hz)
        Iz = (Jz + self.d0) / (1 + self.d - self.d * (Jz + self.d0))
        L2 = Iz + 0.13860504 * Az + 0.05804732 * Bz
        M2 = Iz - 0.13860504 * Az - 0.05804732 * Bz
        S2 = Iz - 0.09601924 * Az - 0.81189190 * Bz
        L = self.PQ_LUMINANCE_RANGE * self.simulatedInvPQ(L2)
        M = self.PQ_LUMINANCE_RANGE * self.simulatedInvPQ(M2)
        S = self.PQ_LUMINANCE_RANGE * self.simulatedInvPQ(S2)
        X2 = 1.92422644 * L - 1.00479231 * M + 0.03765140 * S
        Y2 = 0.35031676 * L + 0.72648119 * M - 0.06538442 * S
        Z = -0.09098281 * L - 0.31272829 * M + 1.52276656 * S
        X = (X2 + (self.b - 1) * Z) / self.b
        Y = (Y2 + (self.g - 1) * X) / self.g
        return np.array([X, Y, Z])

    def JChzToXYZMap(self, JChzMap):
        Jz, Cz, hz = JChzMap[:, :, 0], JChzMap[:, :, 1], JChzMap[:, :, 2]
        Az = Cz * np.cos(hz)
        Bz = Cz * np.sin(hz)
        Iz = (Jz + self.d0) / (1 + self.d - self.d * (Jz + self.d0))
        L2 = Iz + 0.13860504 * Az + 0.05804732 * Bz
        M2 = Iz - 0.13860504 * Az - 0.05804732 * Bz
        S2 = Iz - 0.09601924 * Az - 0.81189190 * Bz
        L = self.PQ_LUMINANCE_RANGE * self.simulatedInvPQ(L2)
        M = self.PQ_LUMINANCE_RANGE * self.simulatedInvPQ(M2)
        S = self.PQ_LUMINANCE_RANGE * self.simulatedInvPQ(S2)
        X2 = 1.92422644 * L - 1.00479231 * M + 0.03765140 * S
        Y2 = 0.35031676 * L + 0.72648119 * M - 0.06538442 * S
        Z = -0.09098281 * L - 0.31272829 * M + 1.52276656 * S
        X = (X2 + (self.b - 1) * Z) / self.b
        Y = (Y2 + (self.g - 1) * X) / self.g
        return np.dstack((X, Y, Z))


if __name__ == "__main__":
    jab = Jabz()
    XYZ = np.array([30, 100.0, 120])
    JChz = jab.XYZToJChz(XYZ)
    restored_XYZ = jab.JChzToXYZ(JChz)
    print(f"XYZ = {XYZ}")
    print(f"JChz = {JChz}")
    print(f"restored XYZ = {restored_XYZ}")