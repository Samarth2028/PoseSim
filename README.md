# PoseSim
805 Project on Pose Similarity Estimation
Project Directory Structure:

\---PoseSimilarity
    |   Camparision_Before_Cropping.jpg : 
		image with pose skelotons visualized together after scaling and cropping
    |   Comparision_after_cropping.jpg : 
		image with pose skelotons visualized together before scaling and cropping
    |   input_image_superimposed.jpg : 
		input_image with skeloton superimposed to visually assess the accuracy of the keypoint detection
    |   PoseSim.py :
		Pose Estimation script description for running given in Execution section of this file
    |   Readme.txt
		This file
    |	requirements.txt
		file containing all the dependencies in conda package parser compatible format
    |   result1.csv:
		result for cosine and weighted distance between images saved by 'batch' Mode execution before scaling and resizing
		Helps in assessing the effect of different scales and dimensions of poses in images
    |   result2.csv
		result for cosine and weighted distance between images saved by 'batch' Mode execution before scaling and resizing
		These are the score that accurately give similarity between two poses
    |   test_image_superimposed.jpg: 
		test_image with skeloton superimposed to visually assess the accuracy of the keypoint detection
    |   
    +---images : Directory containing all the test poses for running in batch mode
    |       
    +---pose : Directory containing the Pre-trained OpenPose caffe model with associated prototxt file
    |   \---mpi
    |           pose_deploy_linevec.prototxt
    |           pose_deploy_linevec_faster_4_stages.prototxt
    |           pose_iter_160000.caffemodel
    |           
    \---uimages: Directory containing the single input pose for running in batch mode
            kidst1.jpg


Software Dependencies:
1.) Python version 3.6.13
2.) Pip version 21.0.1

Recommended python package:
* Anaconda python package (As this bundle makes it easier to manage package installations of various versions of python and is less likely to cause path conflicts)

Python Package Dependencies:
1.)matplotlib==3.2.2
2.)numpy==1.18.2
3.)protobuf==3.11.3
4.)opencv-python==4.2.0.34
5.)pandas==1.1.5

Note: The below execution steps are written assuming the Anaconda release of python is being used

Execution Steps:
*Follow the following steps after unzipping the zipped project folder and navigating inside the folder containing PoseSim.py file
1.) Open Anaconda Prompt
2.) Create a virtual environment with all the requirements using the following command:
	* "conda create --name PoseSim --file requirements.txt"
	Note: the --file argument provides requirements.txt file as the list of dependencies to be installed, pip does not recognize "=" with channel information thus use of conda is
	recommended
3.)After the dependencies are setup run the PoseSim.py script using the following commands:
	* PoseSim.py takes in the following arguments
	mode:
        	Description: Mode to run the program in i.e valid choices = ['batch', 'individual']
        	type expected: string
	input:
        	Description: path to the input image in case of individual mode
        	type expected: string
	test:
        	Description: path to the test image in case of individual mode 
        	type expected:string
	To run in 'batch' Mode use the following commands after navigating inside the directory containing PoseSim.py file:
		* "python PoseSim.py -m batch"
		* the project already has some test images in 'images' and input image in 'uimages' folders.
	To run in 'individual' Mode use the following commands after navigating inside the directory containing PoseSim.py file:
		* "python PoseSim.py -m individual -i ./uimages/'input_image_name.jpg' -o ./images/test_image_name.jpg"
		* E.g python PoseSim.py -m individual -i ./images/med1.jpg -t ./images/med2.png

4.) After following the steps above you should be able to get the output in forms .csv files if running in batch mode and images with Similarity Scores in console if running 
individual mode 
