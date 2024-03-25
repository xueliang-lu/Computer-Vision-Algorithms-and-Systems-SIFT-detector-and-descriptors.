# CSCI935/CSCI435 (S223) Computer Vision Algorithms and Systems
# Assignment 2
# Student ID: 8097471
# UOW login name: xl340

import cv2
import numpy as np
import sys

def resize_image(image, max_height, max_width):
    """
    Resize the input image based on the maximum height and width, preserving the aspect ratio.
    :param image: Input image.
    :param max_height: Maximum allowable height for the image.
    :param max_width: Maximum allowable width for the image.
    :return: Resized image.
    """
    image_height, image_width = image.shape[:2]
    aspect_ratio = image_width / image_height
    if image_width > max_width or image_height > max_height:
        # checks which dimension (width or height) is "more out of bounds" relative to its max value 
        if aspect_ratio > max_width / max_height:
            return cv2.resize(image, (max_width, int(max_width / aspect_ratio)),interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(image, (int(max_height * aspect_ratio), max_height),interpolation=cv2.INTER_LINEAR)
        
    return image
 
def sift_y_component(image,option):
    """
    Extract SIFT keypoints and descriptors from the Y channel of an image.
    :param image: Input image.
    :param option: either "kp_only" for just keypoints or "kp_des" for both keypoints and descriptors.
    :return: Keypoints and possibly descriptors.
    """
    # get iamge Y component
    image_y = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
    # use sift to get keypoints and descriptors
    sift = cv2.SIFT_create()
    if option == "kp_only":
        return sift.detect(image_y,None)
    elif option == "kp_des":
        return sift.detectAndCompute(image_y,None)
    else:
        print("function sift_y_component()- second argument invild!")
    
def draw_cross(image,keypoint,cross_color,cross_rr):
    """
    Draw crosses on the image at the location of the keypoints.
    :param image: Image on which keypoints are to be drawn.
    :param keypoint: List of detected keypoints.
    :param cross_color: Color of the cross.
    :param cross_rr: Relative radius ratio of the cross to the keypoint size.
    :return: Image with crosses.
    """
    for kp in keypoint:
        x, y = map(int, kp.pt)
        scale = int(kp.size * cross_rr)
        cv2.line(image, (x - scale, y), (x + scale, y), cross_color)
        cv2.line(image, (x, y - scale), (x, y + scale), cross_color)

    return image

def get_imageinfo_list(image_list):
    """
    Process a list of images to extract keypoints and descriptors.
    :param image_list: List of paths to input images.
    :return: List of counts of keypoints and list of descriptors for all images.
    """
    kp_count_list=[]
    des_full_list=[]
       
    for image_path in image_list:
        image=cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}. Please check the file path and file format.")
            return
        image_VGA = resize_image(image,480,600)
        kp, des = sift_y_component(image_VGA,"kp_des")
        #count each image keypoints and put them in a list
        kp_count_list.append(len(kp))
        # put all images descriptors in one list
        des_full_list.extend(des)

    return kp_count_list,des_full_list

def get_kclusters_labels(kp_count_list,des_full_list,K):
    """
    Perform KMeans clustering on descriptors.
    :param kp_count_list: List of counts of keypoints.
    :param des_full_list: List of descriptors.
    :param K: Number of clusters for KMeans.
    :return: List of labels corresponding to descriptors for each image.
    """
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    _,labels,_ = cv2.kmeans(np.float32(des_full_list),K,None,criteria,10,flags)

    # based on keypoints number of each image to slice out their labels from the all images list
    image_label_list = []
    start = 0
    for kp_num in kp_count_list:
        end = start + kp_num
        image_label_list.append(labels[start:end].flatten())
        start = end
    
    return image_label_list

def get_histogram_list(image_list,K,image_label_list,kp_count_list):
    """
    Create normalized histograms based on labels for descriptors.
    :param image_list: List of paths to input images.
    :param K: Number of clusters.
    :param image_label_list: List of labels corresponding to descriptors for each image.
    :param kp_count_list: List of counts of keypoints.
    :return: List of histograms.
    """
    # construct the blank histogram and use a list to hold each image histogram separatelly 
    histogram_list = []
    for image in image_list:
        histogram = np.zeros(K)
        histogram_list.append(histogram)
    # count occurence of each label within a image and increase the related histogram
    for i in range(len(image_list)):
        for label in image_label_list[i]:
           histogram_list[i][label]+=1 
        # normalization of each histogram 
        histogram_list[i] = histogram_list[i]/kp_count_list[i]
    
    return np.float32(histogram_list)

def get_disim_matrix(histogram_list):
    """
    Calculate the dissimilarity matrix based on histograms.
    :param histogram_list: List of histograms.
    :return: Dissimilarity matrix.
    """
    # construct a blank dissimilarity matrix
    image_num = len(histogram_list)
    disim_matrix = np.zeros((image_num,image_num))
    for i in range(image_num):
        # caculate the distance for two different images once to save computation
        # as distance of (a,b) and (b,a) is same and (a,a) is equal to zero
        for j in range(i+1,image_num):
            #use chi_squared to caculate the distance between the normalized histograms of visual words of the images
            h1 = histogram_list[i]
            h2 = histogram_list[j]
            chi_squared_distance = 0.5 * np.sum(( h1- h2)**2 / (h1 + h2+ np.finfo(float).eps))      
            disim_matrix[i,j] = chi_squared_distance
       
    return disim_matrix

def print_matrix(image_list, kp_count_list, k_value_pool, disim_matrix_pool):
    """
    Print the dissimilarity matrices for the given images.
    :param image_list: List of paths to input images.
    :param kp_count_list: List of counts of keypoints.
    :param k_value_pool: List of k-values (as percentages).
    :param disim_matrix_pool: List of computed dissimilarity matrices.
    """
    total_keypoints = sum(kp_count_list)
    # display the number of keypoints in each input image
    print("\n\nNumber of keypoints for each image and the total number of keypoints of all images:\n")
    for image, kp_count in zip(image_list, kp_count_list):
        print(f"# of keypoints in {image} is {kp_count}")

    # display dissimilarity matrices
    print("\n\nDissimilarity matrices for K=5%, 10% and 20% of the total number of keypoints from all images:\n")

    for k_value, disim_matrix in zip(k_value_pool, disim_matrix_pool):
        print(f"K = {int(k_value*100)}% * (total number of keypoints = {total_keypoints}) = {int(total_keypoints * k_value)}")
        print("Dissimilarity Matrix")
        print("\n")

        # Printing header
        print(" "*10, end="")
        for image_path in image_list:
            print(f"{image_path.split('.')[0]:<10}", end="")
        print("\n")

        # Printing matrix values
        for i, image_path in enumerate(image_list):
            print(f"{image_path.split('.')[0]:<10}", end="")
            for j in range(len(image_list)):
                # only display once of the result between 2 images 
                # result of (image 1, image 2) is same with (image2, image1) 
                if j < i:
                    print(f"{' ':<10}", end="")
                else:
                    print(f"{disim_matrix[i][j]:<10.2f}", end="")
            print("\n")
        print("\n")
def show_image(image):
    """
    Function to display image
    :param image:
    :return:
    """
    # name the display window 
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # show the image in the window
    cv2.imshow('image', image)
    # infinity loop with 0 miliseconds wait after loop, press any key can stop the display
    cv2.waitKey(0)
    # destroy or close all windows at any time after exiting the script
    cv2.destroyAllWindows()

def task1(image_path):
    """
    Load an image, extract keypoints using SIFT and display the keypoints.
    :param image_path: Path to the input image.
    """
    image=cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path and file format.")
        return
    image_VGA = resize_image(image,480,600)
    keypoint = sift_y_component(image_VGA,"kp_only")
    image_draw_kp=cv2.drawKeypoints(image_VGA,keypoint,image_VGA.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_kp_cross=draw_cross(image_draw_kp,keypoint,(0,0,255),0.125)
    # combine original resized image and the image with highlighted keypoints horizontally
    image_displayed = np.hstack((image_VGA,image_kp_cross))

    print(f"# of keypoints in {image_path} is {len(keypoint)}")

    show_image(image_displayed)

def task2(image_list):
    """
    Process the input image list to compute the dissimilarity matrix based on visual words.
    :param image_list: List of paths to input images.
    """
    kp_count_list,des_full_list = get_imageinfo_list(image_list)
    
    # set up a K % pool for further adding new value
    # store dissimilarity matrix within different K % in a list for further output
    k_value_pool = [0.05, 0.1, 0.2]
    disim_matrix_pool=[]
    for k_value in k_value_pool:
        K = int(sum(kp_count_list) * k_value)
        image_label_list= get_kclusters_labels(kp_count_list,des_full_list,K)
        histogram_list=get_histogram_list(image_list,K,image_label_list,kp_count_list)
        disim_matrix = get_disim_matrix(histogram_list)
        disim_matrix_pool.append(disim_matrix)

    # output result
    print_matrix(image_list,kp_count_list,k_value_pool,disim_matrix_pool)

def parse_and_run():
    """
    Parse command line arguments to determine which task to run.
    Task 1: Process a single image and display keypoints.
    Task 2: Process a list of images, extract features and calculate a dissimilarity matrix.
    """
    # wrong input handle
    if len(sys.argv)< 2:
        print("Wrong input!")
        print("For task one: Please provide 1 image path.")
        print("For task two: Please provide 2 or more images paths.")
        return
    # input one image path, execute task1()
    elif len(sys.argv)==2:
        task1(sys.argv[1])
    # input two or more images paths, execute task2()
    else:
        task2(sys.argv[1:])
        
if __name__== '__main__':
    parse_and_run()