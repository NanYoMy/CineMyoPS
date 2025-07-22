import cv2



from visulize.helper import extract_semantic_contour

img = cv2.imread('../../data/test/a.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours=extract_semantic_contour(binary)
draw_img0 = cv2.drawContours(img.copy(), contours[0], 0, (0, 255, 255), 1)
draw_img1 = cv2.drawContours(img.copy(), contours[0], 1, (255, 0, 255), 1)
draw_img2 = cv2.drawContours(img.copy(), contours[0], 2, (255, 255, 0), 1)
draw_img3 = cv2.drawContours(img.copy(), contours[0], -1, (0, 0, 255), 1)

print("contours:类型：", type(contours))
print("第0 个contours:", type(contours[0]))
print("contours 数量：", len(contours))

# print("contours[0]点的个数：", len(contours[0]))
# print("contours[1]点的个数：", len(contours[1]))

# cv2.imshow("img", img)
# cv2.imshow("draw_img0", draw_img0)
# cv2.imshow("draw_img1", draw_img1)
# cv2.imshow("draw_img2", draw_img2)
cv2.imshow("draw_img3", draw_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()