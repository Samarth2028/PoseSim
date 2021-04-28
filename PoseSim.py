import matplotlib.pyplot as plt
import numpy as np
import math

#Dictionalry of Body parts with their corresponding id's according to mpi openpose model
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
            "Background": 15 }
#Dictionalry of Part Affinity pairs, i.e pairs of valid connections that lead to a human skeleton
POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
            ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

#Initialized Arrays for storing cosine distance and weighted distance for image pairs without application of resizing and cropping 
Cosine1 = []
Weight1 = []

#Initialized Arrays for storing cosine distance and weighted distance for image pairs with application of resizing and cropping
Cosine2 = []
Weight2 = []



"""
Function that takes in 2 arguments:
    pose1:
        Description: takes in the input pose 
        type expected: list of tuples
    pose2:
        Description: takes in the test pose 
        type expected: list of tuples
Returns:
    cosdist:
        Description: the cosine distance between two poses
        type: float

"""
def cosine_distance(pose1, pose2):
	cossim = pose1.dot(np.transpose(pose2)) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))

	cosdist = (1 - cossim)
	return cosdist





"""
Function that takes in 3 arguments:
    pose1:
        Description: takes in the input pose 
        type expected: list of tuples
    pose2:
        Description: takes in the test pose 
        type expected: list of tuples
    conf1:
        Description: takes in the probability/confidence scores associated with each keypoint
        type expected: list of tuples
Returns:
    weighted_dist:
        Description: the weighted distance between two pose
        type: float

"""
def weight_distance(pose1, pose2, conf1):
	sum1 = 1 / np.sum(conf1)

	sum2 = 0
	for i in range(len(pose1)):
		conf_ind = math.floor(i / 2)
		sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])

	weighted_dist = sum1 * sum2

	return weighted_dist




"""
Function that takes in 1 arguments:
    folder:
        Description: takes in the folder containing images to be traversed (used in batch mode for traversing over images)
        type expected: string
Returns:
    exerfilename:
        Description: list of file names in the traversed folder with appropriate paths attached. 
        type: list

"""

def load_images_from_folder(folder):
    exerfilename = []
    print("Retrieving exercise images")
    for root, dirs, files in os.walk(folder):
        for filename in files:
            exerfilename.append(filename)
    return exerfilename



"""
Function that takes in 2 arguments:
    image_path1:
        Description: path of the input_image 
        type expected: string
    image_path2:
        Description: path of test_image
        type expected: string
Returns:
    resized_img1:
        Description: resized input_image
        type: 2d array
    resized_img2:
        Description: resized test_image
        type: 2d array


"""

def load_images(image_path1, image_path2):
    print("Reading images")
    image1 = cv2.imread(image_path1);
    image2 = cv2.imread(image_path2);
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])
    resized_img1 = cv2.resize(image1, (width,height))
    resized_img2 = cv2.resize(image2, (width,height))
    print("Returning images")
    return resized_img1,resized_img2;




"""
Function that takes in 1 arguments:
    image:
        Description: the image for which keypoints are to be detected
        type expected: 2d array representation of image to be passed to the OpenPose model
Returns:
    pose:
        Description: list of 16 Keypoints (x,y)
        type:list
    image:
        Description: the image for which keypoints are estimated
        type: 2d array
    Confidence:
        Description: the confidence scores associated with wach keypoint in the image
        type: list
"""
def estimatePose(image):
    imgWidth = image.shape[1]
    imgHeight = image.shape[0]

    inWidth = 300
    inHeight = 300
    print("Reading posenetwork")
    poseNetwork = cv2.dnn.readNetFromCaffe("./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", "./pose/mpi/pose_iter_160000.caffemodel")
    image_blob = cv2.dnn.blobFromImage(image, (1.0 / 255), (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    poseNetwork.setInput(image_blob)
    output_inference = poseNetwork.forward()
    #print (output_inference.shape)
    print("Making pose coord list")
    pose = [];
    Confidence = []
    for i in range(len(BODY_PARTS)):
        heatMap = output_inference[0, i, :, :]
        minConf, maxConf, minPoint, maxPoint = cv2.minMaxLoc(heatMap)
        
        x = (imgWidth * maxPoint[0]) / output_inference.shape[3]
        y = (imgHeight * maxPoint[1]) / output_inference.shape[2]

        pose.append((int(x), int(y)) if maxConf > 0.1 else None)
        Confidence.append(maxConf)
    #print (pd.DataFrame(pose, columns=['x','y']))
    
    print("Sending pose info...")
    return pose, image, Confidence



"""
Function that takes in 3 arguments:
    pose:
        Description: list of 16 Keypoints (x,y)
        type:list
    image:
        Description: image on which skeleton is to be drawn
        type expected: 2d array of image
    name:
        Description: name of the file with which the output image with skeleton is to be saved
        type expected: string
Returns:


"""
def drawPoseOnImage(pose, image, name):
    print("Drawing pose")
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if pose[idFrom] and pose[idTo]:
            cv2.line(image, pose[idFrom], pose[idTo], (255, 74, 0), 3)
            cv2.ellipse(image, pose[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(image, pose[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(image, str(idFrom), pose[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            cv2.putText(image, str(idTo), pose[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
    fig2 = plt.figure(figsize = (10,10))
    cv2.imwrite(name,cv2.cvtColor(image[:,:,::-1], cv2.COLOR_RGB2BGR))
    axesImg = plt.imshow(image[:,:,::-1])
    plt.ion()
    plt.show(block = False)
    plt.pause(3)




"""
Function that takes in 3 arguments:
    pose1:
        Description: takes in the input pose 
        type expected: list of tuples
    pose2:
        Description: takes in the test pose 
        type expected: list of tuples
   input_img_conf:
        Description: the confidence scores associated with input image pose to determine the weightage to be given to each keypoint for score calculation
        type expected:
Returns:
    cosdist:
        Description: the cosine distance between two poses
        type: float
    weighted_dist:
        Description: the weighted distance between two pose
        type: float

"""
def similarity_score(pose1, pose2, input_img_conf):
    p1 = []
    p2 = []
    pose_1 = np.array(pose1, dtype=float)
    pose_2 = np.array(pose2, dtype=float)

    # Normalize coordinates
    pose_1[:,0] = pose_1[:,0] / max(pose_1[:,0])
    pose_1[:,1] = pose_1[:,1] / max(pose_1[:,1])
    pose_2[:,0] = pose_2[:,0] / max(pose_2[:,0])
    pose_2[:,1] = pose_2[:,1] / max(pose_2[:,1])


    for joint in range(pose_1.shape[0]):
        x1 = pose_1[joint][0]
        y1 = pose_1[joint][1]
        x2 = pose_2[joint][0]
        y2 = pose_2[joint][1]

        p1.append(x1)
        p1.append(y1)
        p2.append(x2)
        p2.append(y2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    cosine_dist = cosine_distance(p1, p2)
    weighted_dist = weight_distance(p1, p2, input_img_conf)
    #print("Cosine Distance:", cosine_dist)
    #print("Weighted Distance:", weighted_dist)
    return cosine_dist,weighted_dist



"""
Function that takes in 4 arguments:
    pose1:
        Description: takes in the input pose 
        type expected: list of tuples
    pose2:
        Description: takes in the test pose 
        type expected: list of tuples
    size:
        Description: size of the canvas to be taken (given as input_image width)
        type expected: int
    name:
        Description: name with which the the image containing both poses together are to be saved
        type expected: string
Returns:

"""
def drawSkeletonsTogether(pose1, pose2, size,name):
    canvas = np.ones(size)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if pose1[idFrom] and pose1[idTo]:
            cv2.line(canvas, pose1[idFrom], pose1[idTo], (0,100,255), 3)
            cv2.ellipse(canvas, pose1[idFrom], (4, 4), 0, 0, 360, (0, 100, 255), cv2.FILLED)
            cv2.ellipse(canvas, pose1[idTo], (4, 4), 0, 0, 360, (0, 100, 255), cv2.FILLED)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if pose2[idFrom] and pose2[idTo]:
            cv2.line(canvas, pose2[idFrom], pose2[idTo], (255, 0, 0), 3)
            cv2.ellipse(canvas, pose2[idFrom], (4, 4), 0, 0, 360, (255, 0, 0), cv2.FILLED)
            cv2.ellipse(canvas, pose2[idTo], (4, 4), 0, 0, 360, (255, 0, 0), cv2.FILLED)

    figure = plt.figure(figsize = (10,10))
    axesImg = plt.imshow(canvas[:,:,::-1])
    cv2.imwrite(name,canvas[:,:,::-1])
    plt.grid(True)
    plt.ion()
    plt.show(block = False)
    plt.pause(3)


"""
Function that takes in 2 arguments:
Scales the pose by taking max(test_pose)/max(input_pose) for each axis
Crops pose by taking : x-min(x) and y-min(y) for all coordinates
    pose1:
        Description: takes in the input pose 
        type expected: list of tuples
    pose2:
        Description: takes in the test pose 
        type expected: list of tuples
Returns:
    crop_resized_pose1:
        Description: resized and cropped pose1
        type: list of tuples
   crop_resized_pose2:
        Description: resized and cropped pose2
        type: list of tuples

"""
def resize_and_crop(pose1,pose2):
    crop_resized_pose1 = np.array(pose1)
    crop_resized_pose2 = np.array(pose2)

    crop_resized_pose1[:,0] = crop_resized_pose1[:,0] - min(crop_resized_pose1[:,0])
    crop_resized_pose1[:,1] = crop_resized_pose1[:,1] - min(crop_resized_pose1[:,1])

    crop_resized_pose2[:,0] = crop_resized_pose2[:,0] - min(crop_resized_pose2[:,0])
    crop_resized_pose2[:,1] = crop_resized_pose2[:,1] - min(crop_resized_pose2[:,1])

    resize_x = max(crop_resized_pose2[:,0])/max(crop_resized_pose1[:,0])

    resize_y = max(crop_resized_pose2[:,1])/max(crop_resized_pose1[:,1])

    crop_resized_pose1[:, 0] = crop_resized_pose1[:, 0] * resize_x
    crop_resized_pose1[:, 1] = crop_resized_pose1[:, 1] * resize_y

    return crop_resized_pose1, crop_resized_pose2




"""
Function that takes in 4 arguments:
    img:
        Description: image name 
        type expected: string
    Cos:
        Description: Cosine distance compared to the input_image in uimages for the associated image
        type expected: float
    Wei:
        Description: Weighted distance compared to the input_image in uimages for the associated image
        type expected: float
    filename:
        Description: filename with which the results are to be saved in the csv file
        type expected:
Returns:
    No return value


"""
def writeresults(img, Cos, Wei, filename):
    with open(filename, 'w', newline='') as file:
        fieldnames = ['Position', 'Cosine Similarity', 'Weighted Distance']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for i, score1, score2 in zip(img, Cos, Wei):
            writer.writerow({'Position': i, 'Cosine Similarity': score1, 'Weighted Distance': score2})


"""
Function that takes in 0 arguments:
        Description: Function that runs the similarity calculation for all of the images in ./images folder as test_images compared to the imput_image in ./uimages folder 

Returns:
        No return value
"""

def runBatchMode():
    images = load_images_from_folder("./images/")
    uimages = load_images_from_folder("./uimages/")
    for i in images:
        input_image,test_image = load_images("./images/"+i, "./uimages/"+uimages[0])
        input_pose, input_img, input_conf = estimatePose(input_image)
        test_pose, test_img, test_conf = estimatePose(test_image)
        print("*****************************Similarity score without Cropping and Scaling Transformation****************************************************************")
        Cos1, wei1 = similarity_score(input_pose,test_pose,input_conf)
        Cosine1.append(Cos1)
        Weight1.append(wei1)
        print("*************************Similarity score with Cropping and ScalingTransformation***********************************")
        cr_pose1,cr_pose2 = resize_and_crop(input_pose,test_pose)
        cr_tuple_pose1  = tuple(map(tuple,cr_pose1))
        cr_tuple_pose2  = tuple(map(tuple,cr_pose2))
        arrInPose  = np.array(input_pose)
        arrTestPose = np.array(test_pose)
        Ymax = max(max(arrInPose[:,0]),max(arrTestPose[:,0]))
        Xmax = max(max(arrInPose[:,0]),max(arrTestPose[:,0]))
        dim = (Xmax,Ymax,3)
        Cos2, wei2 = similarity_score(cr_pose1,cr_pose2,input_conf)
        Cosine2.append(Cos2)
        Weight2.append(wei2)
    writeresults(images, Cosine1, Weight1, 'result1.csv')
    writeresults(images, Cosine2, Weight2, 'result2.csv')



"""
Function that takes in 2 arguments:
 Function to compare a single input image to a test image

    input_img_path:
        Description: path to the input_image when running in individual mode
        type expected: string
    test_img_path:
        Description: path to the test_image when running in indicidual mode
        type expected: string
Output:
    Gives input_image and test_image superimposed with detected skelotons, also poses together before scaling and cropping and after scaling and cropping, each output image is displayed for 3 seconds initaillay on screen
     Saves 4 images as output:
  * input_image_superimposed.jpg
  * test_image_superimposed.jpg
  * Camparision_Before_Cropping.jpg
  * Comparision_after_cropping.jpg
"""
def runSingleCompareMode(input_img_path,test_img_path):
    input_image,test_image = load_images(image_path1 = input_img_path,image_path2 = test_img_path);
    input_pose, input_img, input_conf = estimatePose(input_image)
    test_pose, test_img, test_conf = estimatePose(test_image)
    drawPoseOnImage(input_pose,input_img,'Input_image.jpg')
    drawPoseOnImage(test_pose,test_img,'Test_Image.jpg')
    print("*****************************Similarity score without Cropping and Scaling Transformation****************************************************************")
    Cos1, wei1 = similarity_score(input_pose,test_pose,input_conf)
    print("Cosine Similarity:",Cos1);
    print("Weighted Similarity:",wei1);
    drawSkeletonsTogether(input_pose,test_pose,input_image.shape,'Camparision_Before_Cropping.jpg')
    print("*************************Similarity score with Cropping and ScalingTransformation***********************************")
    cr_pose1,cr_pose2 = resize_and_crop(input_pose,test_pose)
    cr_tuple_pose1  = tuple(map(tuple,cr_pose1))
    cr_tuple_pose2  = tuple(map(tuple,cr_pose2))
    arrInPose  = np.array(input_pose)
    arrTestPose = np.array(test_pose)
    Ymax = max(max(arrInPose[:,0]),max(arrTestPose[:,0]))
    Xmax = max(max(arrInPose[:,0]),max(arrTestPose[:,0]))
    dim = (Xmax,Ymax,3)
    drawSkeletonsTogether(cr_tuple_pose1,cr_tuple_pose2,dim,'Comparision_after_cropping.jpg')
    Cos2, wei2 = similarity_score(cr_pose1,cr_pose2,input_conf)
    print("Cosine Similarity:",Cos1);
    print("Weighted Similarity:",wei1);




"""
Function that takes in 1 required argument and 2 optional Arguments:
    mode:
        Description: Mode to run the program in i.e valid choices = ['batch', 'individual']
        type expected: string
    input:
        Description: path to the input image in case of individual mode
        type expected: string
    test:
        Description: path to the test image in case of individual mode 
        type expected:string
Returns:
    No return value

"""
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",choices=['batch','individual'],required=True,help="Mode to be used for running the program")
    parser.add_argument("-i","--input",default = "./images/med1.jpg",help="Input image")
    parser.add_argument("-t","--test",default="./images/med2.png",help="Test image")
    args = parser.parse_args()
    mode = args.mode;
    if mode == 'individual':
        input_img_path = args.input
        test_img_path = args.test
        runSingleCompareMode(input_img_path,test_img_path)
    elif mode == 'batch':
        runBatchMode()
    else:
        print("Please provide the required arguments and run again")
    

if __name__ == "__main__":
    main()

