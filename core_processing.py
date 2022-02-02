import cv2.cv2
import numpy as np

from scipy.signal import convolve2d


def image_blurring(image: np.ndarray) -> np.ndarray:
    """
    A helper function

    Args:
        image: A numpy array representing the input image

    Returns: A blurred image (after applying a 2D convolution with a gaussian manually built mask)

    """

    gaussian_mask = (1 / 164) * np.array([[1, 8, 1],
                                         [8, 128, 8],
                                         [1, 8, 1]])

    image = convolve2d(image, gaussian_mask, mode='same')

    return image


def sobel_edge_detection(image: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """

    Args:
        image: A numpy array representing the input image
        threshold: An integer representing the threshold hyper-parameter in the Sobel Edge Detection algorithm

    Returns: Two images. One represents the edges in the input image (after applying the Sobel Edge Detection algorithm
    on). And the second, an image which represents the gradient orientation of of the edges in the image

    """

    image = image_blurring(image)

    sobel_x_mask = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
    sobel_y_mask = sobel_x_mask.T
    
    sobel_edge_x = convolve2d(image, sobel_x_mask, mode='same')
    sobel_edge_y = convolve2d(image, sobel_y_mask, mode='same')
    
    sobel_edge_magnitude = (sobel_edge_x * sobel_edge_x + sobel_edge_y*sobel_edge_y)**0.5
    sobel_edge_map = sobel_edge_magnitude > threshold

    # Calculating the gradient direction
    sobel_gradient_direction = np.arctan2(sobel_edge_y, sobel_edge_x) * (180 / np.pi)
    sobel_gradient_direction += 180
    sobel_gradient_direction *= sobel_edge_map

    # Coloring the different gradient orientations
    red = np.array([0, 0, 255])
    cyan = np.array([255, 255, 0])
    green = np.array([0, 255, 0])
    yellow = np.array([0, 255, 255])

    sobel_grad_orien = np.zeros((sobel_gradient_direction.shape[0], sobel_gradient_direction.shape[1], 3),
                                dtype=np.uint8)

    # setting the colors, maybe there is a better way, my numpy skills are rusty
    # it checks that magnitude is above the threshold and that the orientation is in range
    sobel_grad_orien[sobel_gradient_direction < 90] = red
    sobel_grad_orien[(sobel_gradient_direction > 90) & (sobel_gradient_direction < 180)] = cyan
    sobel_grad_orien[(sobel_gradient_direction > 180) & (sobel_gradient_direction < 270)] = green
    sobel_grad_orien[(sobel_gradient_direction > 270)] = yellow

    # Discarding the background gradient orientation
    sobel_grad_orien[:, :, 0] *= sobel_edge_map
    sobel_grad_orien[:, :, 1] *= sobel_edge_map
    sobel_grad_orien[:, :, 2] *= sobel_edge_map

    return sobel_edge_map, sobel_grad_orien


def create_hough_lines_image(image: np.ndarray, canny_p1: int, canny_p2: int, hough_th: int, min_angle: int, min_r: int,
                             blur_mask: np.ndarray = None) -> np.ndarray:
    """

    Args:
        image: A numpy array representing the input image
        canny_p1: A canny edge detector upper hysteresis threshold
        canny_p2: A canny edge detector lower hysteresis threshold
        hough_th: A Hough algorithm intersection threshold
        min_angle: The minimum threshold angle for which a line would be counted as a Hough edge line (above the
                   minimum angle)
        min_r: The minimum threshold 'radius' for which a line would be counted as a Hough edge line (above the
                   minimum radius)
        blur_mask: A numpy array representing a mask which implies whether to perform a blurring filter (through a
                   convolution with the blurring mask). Default None.

    Returns: An image which represent the original image with Hough lines detected (drew on)

    """
    
    if blur_mask is None:
        image_blur = image
    else:
        image_blur = convolve2d(image, blur_mask, mode='same')
    
    image_canny = cv2.Canny(image_blur.astype('uint8'), canny_p1, canny_p2, None, 3)
    lines = cv2.HoughLines(image_canny, 1, np.pi / 180, hough_th, None, 0, 0)
    image_diag_size = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5
    
    min_angle_dif = min_angle
    min_r_dif = min_r

    # Find all 'Unique' Hough lines
    filtered_lines = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        
        exist = False
        for line in filtered_lines:
            if np.abs(line[0] - rho) < min_r_dif and np.abs(line[1] - theta) < min_angle_dif:
                exist = True
                break
        
        if not exist:
            filtered_lines.append([rho, theta])
    
    image_copy = np.copy(image)
    for rho, theta in filtered_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + image_diag_size*(-b)), int(y0 + image_diag_size * a))
        pt2 = (int(x0 - image_diag_size*(-b)), int(y0 - image_diag_size * a))
        cv2.line(image_copy, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        
    return image_copy
