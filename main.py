import matplotlib.pyplot as plt

from cv2.cv2 import HOUGH_GRADIENT
from .core_processing import *


def processing():

    print("---------------------- Objective 1 ---------------------\n")
    image_1 = cv2.imread(r'balls1.tif')
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    threshold = 100
    sobel_edge_map, sobel_gradient_direction = sobel_edge_detection(image_1, threshold)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('original')
    plt.imshow(image_1, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('edge magnitude')
    plt.imshow(sobel_edge_map, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('edge direction')
    plt.imshow(sobel_gradient_direction, cmap='gray')

    print("---------------------- Objective 2 ---------------------\n")
    image_2 = cv2.imread(r'coins1.tif')
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    image_2_manipulation = np.copy(image_2)
    image_2_manipulation[np.where(image_2_manipulation < 255)] = 0

    canny_th_low = 1000
    canny_th_high = 1400
    # canny_th_low = 1019
    # canny_th_high = 1100

    image_canny = cv2.GaussianBlur(image_2_manipulation, (9, 9), 0)
    image_canny = cv2.Canny(image_2_manipulation, canny_th_low, canny_th_high)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('original image')
    plt.imshow(image_2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.title('canny image')
    plt.imshow(image_canny, cmap='gray')

    print("---------------------- Objective 3 ---------------------\n")
    image_3 = cv2.imread(r'balls1.tif')
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    th_low = 90
    th_high = 160
    # th_low = 100
    # th_high = 200

    # image_3_canny = cv2.GaussianBlur(image_3, (9, 9), 0)
    image_3_canny = cv2.Canny(image_3, th_low, th_high)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_3_canny, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Objective 4 ---------------------\n")
    image_4 = cv2.imread(r'coins3.tif')
    image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(image_4, HOUGH_GRADIENT, dp=1, minDist=1, param1=45, param2=100, minRadius=5,
                               maxRadius=80)
    # circles = cv2.HoughCircles(image_4, HOUGH_GRADIENT, dp=1, minDist=1, param1=23, param2= 114.5, minRadius=10,
    #                            maxRadius=0)
    image_4_circles = np.copy(image_4)
    for circle in circles[0, :]:

        # draw the outer circle
        circle = circle.astype('int')
        image_4_circles = cv2.circle(image_4_circles, (circle[0], circle[1]), circle[2], 0, 5)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_4_circles, cmap='gray', vmin=0, vmax=255)

    print("Describing the problem with the image and the method/solution: \n")
    print("The problem with the image is that we have coins with different radius, and there's a compensation between "
          "catching all the small circles and yet not emphasising\\creating new, not existing, ones\n")

    print("---------------------- Objective 5 ---------------------\n")
    image_5 = cv2.imread(r'boxOfchocolates1.tif')
    image_5 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)

    image_5_lines = create_hough_lines_image(image_5, 200, 500, 150, 1, 10)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_5)
    plt.subplot(1, 2, 2)
    plt.imshow(image_5_lines)

    print("Describing the problem with the image and the method/solution: \n")
    print("The problem is that there're lots of close lines which the hough algorithm can't separate efficiently. Our "
          "method is to help the algorithm by pre-processing and finding the edges (which the lines are included in)"
          ", and after, transferring it to Hough algorithm for line detecting. Still, for the above mentioned problem,"
          " we pass the results to a filtered method which eliminates close lines (by encountering the distance and "
          "the angle of the line)\n")

    print("---------------------- Objective 6 ---------------------\n")
    image_6 = cv2.imread(r'boxOfchocolates2.tif')
    image_6 = cv2.cvtColor(image_6, cv2.COLOR_BGR2GRAY)

    blur_mask = np.ones([5, 5])
    blur_mask = blur_mask / blur_mask.sum()
    image_6_lines = create_hough_lines_image(image_6, 60, 90, 100, 1.5, 30, blur_mask)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_6)
    plt.subplot(1, 2, 2)
    plt.imshow(image_6_lines)

    print("Describing the problem with the image and the method/solution: \n")
    print("The problem, in addition to the above mentioned points, is that there are also non-continues inner lines. "
          "Our method is to help the algorithm by pre-processing and finding the edges (which the lines are included"
          " in), and after, transferring it to Hough algorithm for line detecting. Still, for the above mentioned "
          "problem, we pass the results to a filtered method which eliminates close lines (by encountering the distance"
          " and the angle of the line)\n")

    print("---------------------- Objective 7 ---------------------\n")
    image_7 = cv2.imread(r'boxOfchocolates2rot.tif')
    image_7 = cv2.cvtColor(image_7, cv2.COLOR_BGR2GRAY)

    image_7_lines = create_hough_lines_image(image_7, 0, 0, 130, 1, 10)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_7)
    plt.subplot(1, 2, 2)
    plt.imshow(image_7_lines)

    print("Describing the problem with the image and the method/solution: \n")
    print("The problem, in addition to the above mentioned points, is that there are also a slight rotation to the "
          "lines. Our method is to help the algorithm by pre-processing and finding the edges (which the lines are "
          "included in), and after, transferring it to Hough algorithm for line detecting. Still, for the above "
          "mentioned problem, we pass the results to a filtered method which eliminates close lines (by encountering "
          "the distance and the angle of the line)\n")

    plt.show()


if __name__ == "__main__":
    processing()
