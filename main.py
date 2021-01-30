import cv2
import os
import numpy as np
#import skvideo.io #to use skvideo library, imput "pip install sk_video" into a terminal or command line
import glob



#Determine the title, fps and size of the final video
video_title = 'finalvideo.avi'
frames_per_second = 30
size = (800,600)


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


#List all the files in the directory. All of the jpg files should be in this directory only. Otherwise, the input
#will not work. Make sure the correct directory is listed 
dirListing = os.listdir('/home/chris/git/Face detector/Images/')

#Files array is used to store the input of the images
Files = []

#store all files with jpg extensions into array Files
for item in dirListing:
    if ".jpg" in item:
        Files.append('/home/chris/git/Face detector/Images/'+item)


#Sort Files array
Files.sort()



#Intialize Video exporter
out = cv2.VideoWriter(video_title, cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, size)


#Checks Each image name in the array Files
for i in range(len(Files)):
    
    # Read the input image
    img = cv2.imread(Files[i])
    height, width, layers = img.shape
    size = (width, height)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Export Image to video for stitching
    out.write(img)
    
    #print the directory where the file is located
    print(Files[i])

#Finalize the video and export
out.release()

#Face detection is done!
print("Done!")
















