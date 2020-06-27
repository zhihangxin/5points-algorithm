import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 5)
        img1 = cv2.circle(img1, tuple(pt1), 15, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 15, color, -1)
    return img1, img2

img1 = cv2.imread('aaaa.JPG', 0)
img2 = cv2.imread('bbbb.jpg', 0)
'''
# get the point from img
def OnMouse(event, x, y, flags, param):
    #EVENT_LBUTTONDOWN 左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_2d.append([x, y])
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)


if __name__ == '__main__':
    # index = sys.argv[1]
    pts_2d = []

    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)  # 窗口大小保持比例
    cv2.resizeWindow("image", 1000, 1000);
    # setMouseCallback 用来处理鼠标动作的函数
    # 当鼠标事件触发时，OnMouse()回调函数会被执行
    cv2.setMouseCallback('image', OnMouse)

    while 1:
        cv2.imshow("image", img2)
        k = cv2.waitKey(1)
        if k == 27:
            break

    print(pts_2d)

'''
points1 = [[2582, 2035], [1006, 1152], [1315, 961], [332, 1614], [1663, 2624]]
points2 = [[1660, 2243], [1941, 1082], [2579, 879], [710, 1502], [347, 2779]]

cameraMat = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

pts1 = np.float32(points1)
pts2 = np.float32(points2)
E, mask = cv2.findEssentialMat(pts1, pts2, method = cv2.RANSAC)


F = np.linalg.inv(cameraMat.T).dot(E[0:3,:]).dot(np.linalg.inv(cameraMat))
print(F)
print(E)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()


