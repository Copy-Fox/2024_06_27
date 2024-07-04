from skimage import feature
import numpy as np
#LBP 히스토그램을 만들어주는 클래스 생성
#LBP 반지름, 지정가능
#describe함수에서 직접적으로 이미지를 회색으로 변환
#히스토그램 생성
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

