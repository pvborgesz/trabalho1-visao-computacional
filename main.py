import numpy as np
import dlib
import matplotlib.pyplot as plt
import cv2 as cv 
import os
import bleedfacedetector as fd
import mediapipe as mp
import time
from pkg_resources import resource_filename

haarlocation = resource_filename(__name__, './haarcascade_frontalface_default.xml')
imgLocation = resource_filename(__name__, './3eyes.jpeg')
imgLocation = cv.imread('./pv.jpeg')
hog_detctor = dlib.get_frontal_face_detector()
face_cascade = cv.CascadeClassifier(haarlocation)

def haar_detect(img,scaleFactor = 1.3,minNeighbors = 5,height=0):
    scale=1
    # if the height is 0 then original height will be used
    if height:
       scale = height / img.shape[0]
       img = cv.resize(img, None, fx=scale, fy=scale)

    rscale = 1/scale
    all_faces = face_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
    # resizing all of the coordinates back to original size
    return [[int(var * rscale) for var in face] for face in all_faces]


def ssd_detect(image,conf,returnconf):
    (h, w) = image.shape[:2]
    resizedimage = cv.resize(image, (300, 300))
    blob = cv.dnn.blobFromImage(resizedimage, 1.0,(300, 300), (104.0, 177.0, 123.0))
    all_faces=[]
    # pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        #extract the confidence associated with the prediction
        confidence = detections[0, 0, i, 2]

        #if confidence is less than predefined threshold conf then ignore those predictions
        if confidence < conf:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #if return conf is false then just return the boxes otherwise also return confience
        if not returnconf:
           all_faces.append([startX,startY,endX-startX, endY-startY])
        else:
           all_faces.append([startX,startY,endX-startX, endY-startY,confidence])


def hog_detect(img,upsample=0,height=0):
    if  img.ndim == 3:
        img = cv.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale=1
    # if the height is 0 then original height will be used
    if height:
       scale = height / img.shape[0]
       img = cv.resize(img, None, fx=scale, fy=scale)

    rscale = 1/scale
    faces = hog_detctor(img, upsample)
    all_faces=[]
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        all_faces.append([int(x*rscale),int(y*rscale),int(w*rscale),int(h*rscale)])

    return all_faces

start_time = time.time()
webcam = cv.VideoCapture(0)

solution = mp.solutions.face_detection
face_detect = solution.FaceDetection()
draw = mp.solutions.drawing_utils

# Set model path
model = './Model/emotion-ferplus-8.onnx'
# Now read the model
net = cv.dnn.readNetFromONNX(model)

# Define the emotions
emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

x = np.random.normal(size=1000)

done = True
cont = 0 


while (done):
  verificador, frame = webcam.read()
  if not verificador:
    print(verificador, "err")
    break
  #reconhecer rostos
  faces_list = face_detect.process(frame)
  if (faces_list.detections):
    for face in faces_list.detections:
      draw.draw_detection(frame,face)
  ddepth = cv.CV_16S
  kernel_size = 3
  # window_name = "Laplace Demo"
  frame = cv.GaussianBlur(frame, (5, 5), 0) # da pra aplicar gaussian blur 
  # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  ret, otsu = cv.threshold(frame,0,255,cv.THRESH_BINARY)
  # print(ret,otsu)
  # frame = cv.Laplacian(frame, ddepth, ksize=kernel_size)
  
  img_copy = frame.copy()
  # Use SSD detector with 20% confidence threshold.
  # faces = fd.haar_detect(img_copy,1.3,5,0)
  # faces = fd.haar_detect(img_copy,1.3,1,0)
  # faces = fd.ssd_detect(img_copy, conf=0.2, returnconf=True)
  # faces = face_cascade.detectMultiScale(img_copy) #cascade
  # faces = fd.cnn_detect(img_copy)

  # Apply template Matching
  res = cv.matchTemplate(img_copy,imgLocation,cv.TM_SQDIFF)
  print(imgLocation.shape)
  w, h = imgLocation.shape[:-1]
  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
  top_left = min_loc
  bottom_right = (top_left[0] + w, top_left[1] + h)
  faces = cv.rectangle(img_copy,top_left, bottom_right, 255, 2)
  print(faces)
  x = faces[0] # use with ssd 
#   x,y,w,h,c = faces[0] # use with ssd 
    # Define padding for face roi
  padding = 3
    # Extract the Face from image with padding.
#   face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding] 
  # face = img_copy[y:y+h, x:x+w] 

# Just increasing the padding for demo purpose
  # padding = 20

# Get the Padded face
#   padded_face_demo = img_copy[y-padding:y+h+padding,x-padding:x+w+padding] 
  padded_face_demo = img_copy 

  # plt.figure(figsize=[10, 10])
  gray = cv.cvtColor(img_copy,cv.COLOR_BGR2GRAY)
# Resize into 64x64
  resized_face = cv.resize(gray, (64, 64))
# Reshape the image into required format for the model 
  processed_face = resized_face.reshape(1,1,64,64)
  
  net.setInput(processed_face)
  Output = net.forward()
# The output are the scores for each emotion class
  print('Shape of Output: {} n'.format(Output.shape))
  # print(Output)
#   plt.subplot(121);plt.imshow(padded_face_demo[...,::-1]);plt.title("Padded face");plt.axis('off')
#   plt.subplot(122);plt.imshow(face[...,::-1]);plt.title("Non Padded face");plt.axis('off');
  prob = np.squeeze(Output)
  # print(prob)
  # Compute softmax values for each sets of scores  
  expanded = np.exp(prob - np.max(Output))
  probablities =  expanded / expanded.sum()
  # Get the index of the max probability, use that index to get the predicted emotion in the 
  # emotions list you created above.
  # predicted_emotion = emotions[prob.argmax()]
  fps= (1.0 / (time.time() - start_time))
  
# Get the final probablities 
  # find frequency of pixels in range 0-255
  cv.putText(frame, 'FPS: {:.2f} {}'.format(fps, 'using haar'), (10, 20), cv.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)
  cv.imshow("Faces in webcam", frame)
  # if cont == 2: break
  if cv.waitKey(5) == 27:
    break
webcam.release()      
