#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import HTML
# %matplotlib inline
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
imshape = image.shape
#printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image) 

def grayscale(img):
    # Applies the Grayscale transform
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    # """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    # """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    
    # Applies an image mask.
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    # NOTE: this is the function you might want to use as a starting point once you want to 
    # average/extrapolate the line segments you detect to map out the full
    # extent of the lane (going from the result shown in raw-lines-example.mp4
    # to that shown in P1_example.mp4).  
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    for line in lines:
            for x1,y1,x2,y2 in line:
                gradient = (y2-y1)/(x2-x1)
                ymin_global = min(min(y1, y2), ymin_global)
                if gradient >0:
                    all_left_grad.append(gradient)
                    all_left_y += [y1,y2]
                    all_left_x += [x1,x2]
                elif gradient <0:
                    all_right_grad.append(gradient)
                    all_right_y += [y1,y2]
                    all_right_x += [x1,x2]
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)                

    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global), 
                      (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), 
                      (lower_right_x, ymax_global), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
 
    # Returns an image with hough lines drawn.

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.7, β=1., γ=0.):
    # initial_img * α + img * β + γ
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    gray = grayscale(image)
#     plt.imshow(gray)
    
    kernel_size = 5
    blur_gray_image = gaussian_blur(gray, kernel_size)
#     plt.imshow(blur_gray_image)
    
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(blur_gray_image,low_threshold,high_threshold)
#     plt.imshow(canny_edges)
    
    vertices = np.array([[(0,imshape[0]),(460, 315), (490, 315), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_image = region_of_interest(canny_edges, vertices)
#     plt.imshow(masked_image)
    
    rho = 1
    theta = np.pi/180
    threshold = 5
    min_line_length = 55
    max_line_gap = 25
    line_image = np.copy(image)*0 #creating a blank to draw lines on
    line_image = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
#     plt.imshow(line_image)
    
    color_edges = np.dstack((canny_edges, canny_edges, canny_edges)) 
    line_edges = weighted_img(line_image,color_edges) 
#     plt.imshow(line_edges)
    
    result = weighted_img(line_image, image) 
#     plt.imshow(line_edges)
#     print(result.shape)
    return result


