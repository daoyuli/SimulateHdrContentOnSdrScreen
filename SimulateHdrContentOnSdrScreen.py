import cv2
import numpy as np
import matplotlib.pyplot as plt
from Gamut import XYZToRgb, RgbToXYZ, RgbToRgb, RgbToLuma
from TransferFunction import Gamma22Oetf, Gamma22Eotf, PqEotf, SDR_DIFFUSION_WHITE, PQ_LUMINANCE_RANGE
from ColourAppearanceModel import Jabz
JabzConverter = Jabz()


SCREEN_WHITE = 200  # nits
ALTERNATE_PATH = "alternate.png"
ALTERNATE_GAMUT = "sRGB"
ALTERNATE_TRANSFER = "BT.2020"
BASE_PATH = "base.jpg"
BASE_GAMUT = "sRGB"
BASE_TRANSFER = "Gamma2.2"


HdrBgrImage = cv2.imread(ALTERNATE_PATH, cv2.IMREAD_UNCHANGED)
HdrRgbImage = HdrBgrImage[:, :, ::-1]
HdrRgbLuminance = PqEotf(HdrRgbImage / 65535.0) * PQ_LUMINANCE_RANGE
HdrLuminance = RgbToLuma(HdrRgbLuminance, ALTERNATE_GAMUT)
HdrLightestPoint = np.where(HdrLuminance == np.max(HdrLuminance))
MaxHdrRgbLuminance = HdrRgbLuminance[HdrLightestPoint]
MaxHdrLuminance = HdrLuminance[HdrLightestPoint]

SdrBgrImage = cv2.imread(BASE_PATH)
SdrRgbImage = SdrBgrImage[:, :, ::-1]
SdrRgbLuminance = Gamma22Eotf(SdrRgbImage / 255.0) * SDR_DIFFUSION_WHITE
SdrLuminance = RgbToLuma(SdrRgbLuminance, BASE_GAMUT)
SdrLightestPoint = np.where(SdrLuminance == np.max(SdrLuminance))
MaxSdrRgbLuminance = SdrRgbLuminance[SdrLightestPoint]
MaxSdrLuminance = SdrLuminance[SdrLightestPoint]


MaxHdrXYZ = RgbToXYZ(MaxHdrRgbLuminance[0], "sRGB")
MaxHdrJChz = JabzConverter.XYZToJChz(MaxHdrXYZ)
DiffusionWhiteXYZ = SDR_DIFFUSION_WHITE / 100.0 * np.array([95.047, 100.0, 108.883])  # D65
DiffusionWhiteJChz = JabzConverter.XYZToJChz(DiffusionWhiteXYZ)
ScreenWhiteXYZ = SCREEN_WHITE / 100.0 * np.array([95.047, 100.0, 108.883])  # D65
ScreenWhiteJChz = JabzConverter.XYZToJChz(ScreenWhiteXYZ)
#    MaxHdrJ / DiffusionWhiteJ = ScreenWhiteJ / BackgroundJ
# -> BackgroundJ = ScreenWhiteJ / (MaxHdrJ / DiffusionWhiteJ)
BackgroundJChz = np.array([ScreenWhiteJChz[0] / (MaxHdrJChz[0] / DiffusionWhiteJChz[0]),
                             ScreenWhiteJChz[1], ScreenWhiteJChz[2]])
BackgroundXYZ = JabzConverter.JChzToXYZ(BackgroundJChz)
BackgroundLuminance = XYZToRgb(BackgroundXYZ, "sRGB")
BackgroundLinear = BackgroundLuminance / SCREEN_WHITE
DimmingFactor = np.mean(BackgroundLinear)
print(f"Dimming factor: {DimmingFactor:.2f}, Background: {BackgroundLinear[0]:.2f}")
# compress Hdr luminance to SCREEN_WHITE, and Sdr Luminance to Background
SimuHdrRgbLinear = HdrRgbLuminance / HdrRgbLuminance.max()
SimuSdrRgbLinear = SdrRgbLuminance / SdrRgbLuminance.max() * DimmingFactor
"""
# pixel-wise compression
HdrXYZ = RgbToXYZ(HdrRgbLuminance, "sRGB")
HdrJChz = JabzConverter.XYZToJChzMap(HdrXYZ)
SdrXYZ = RgbToXYZ(SdrRgbLuminance, "sRGB")
SdrJChz = JabzConverter.XYZToJChzMap(SdrXYZ)
SimuHdrRgbJChz = np.dstack((BackgroundJChz[0] * HdrJChz[:, :, 0] / DiffusionWhiteJChz[0],
                         HdrJChz[:, :, 1], HdrJChz[:, :, 2]))
SimuSdrRgbJChz = np.dstack((BackgroundJChz[0] * SdrJChz[:, :, 0] / DiffusionWhiteJChz[0],
                         SdrJChz[:, :, 1], SdrJChz[:, :, 2]))
SimuHdrRgbXYZ = JabzConverter.JChzToXYZMap(SimuHdrRgbJChz)
SimuSdrRgbXYZ = JabzConverter.JChzToXYZMap(SimuSdrRgbJChz)
SimuHdrRgbLuminance = np.maximum(0, XYZToRgb(SimuHdrRgbXYZ, "sRGB"))
SimuSdrRgbLuminance = np.maximum(0, XYZToRgb(SimuSdrRgbXYZ, "sRGB"))
SimuHdrRgbLinear = np.clip(SimuHdrRgbLuminance / SCREEN_WHITE, 0, 1)
SimuSdrRgbLinear = np.clip(SimuSdrRgbLuminance / SCREEN_WHITE, 0, 1)
"""
BackgroundEncode = Gamma22Oetf(BackgroundLinear)
BackgroundColor = BackgroundEncode
SimuHdrRgbImage = np.uint8(255 * Gamma22Oetf(SimuHdrRgbLinear))
SimuSdrRgbImage = np.uint8(255 * Gamma22Oetf(SimuSdrRgbLinear))
plt.figure(facecolor=BackgroundColor)
plt.subplot(121)
plt.imshow(SimuSdrRgbLinear)
plt.title("Simu. SDR Image")
plt.axis("off")
plt.subplot(122)
plt.imshow(SimuHdrRgbImage)
plt.title("Simu. HDR Image")
plt.axis("off")
# plt.show(block=True)
plt.savefig("./Simu.png")

