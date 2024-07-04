from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True, help="path to the testing images")
#python recognize.py --training images/training --testing images/testing 를 터미널에 치면
#args["training"]: --training 다음에 입력한 경로
#args["testing"]: --testing 다음에 입력한 경로
#args에 저장
args = vars(ap.parse_args())

#wallpaper와 tile이 구분이 안되기 때문에 LBP의 값을 구분이 잘되도록 변경 (24,8)->(24,3)
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# 학습 이미지를 순회합니다.

for imagePath in paths.list_images(args["training"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
    print(hist)

# Linear SVC을 학습시킵니다.
#하이퍼 파라미터
#각 모델별 변수를 조절하는 것
#GridSearch를 써서 C값을 변경해야함.
#model = LinearSVC(C=0.1, random_state=42)
#C:과적합 방지 파라미터
model = LinearSVC(C=100,max_iter=15000, random_state=42)
model.fit(data, labels)

# 테스트 이미지를 순회합니다.
for imagePath in paths.list_images(args["testing"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))

    # 이미지와 예측 결과를 표시합니다.
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

