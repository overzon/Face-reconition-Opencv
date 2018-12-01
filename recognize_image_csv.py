from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import pandas as pd
global df, datalist
df = pd.DataFrame([],columns=["label","AntMan","BlackPanter","BlackWidow","Captain","DrS","Hulk","IronMan","Loki","Spiderman","StarLord","SW","Thor","Vision","sum"])
datalist = [0,0,0,0,0,0,0,0,0,0,0,0,0]
avg = 0
antman = 0
blackpanter = 0
blackwidow = 0
captain = 0
drs = 0
hulk = 0
ironman = 0
loki = 0
spiderman = 0
starlord = 0
sw = 0
thor = 0
vision = 0

def add_data(data,datafram,label,number_data):
	print("save")
	sum = 0
	for i in range(0,13):
		sum += data[i]
	dfr = pd.DataFrame([{'label':str(label),'AntMan':data[0],'BlackPanter':data[1],'BlackWidow':data[2],'Captain':data[3],'DrS':data[4],'Hulk':data[5],'IronMan':data[6],'Loki':data[7],'Spiderman':data[8],'StarLord':data[9],'SW':data[10],'Thor':data[11],'Vision':data[12],'sum':(data[number_data]/sum)*100}])
	datafram = dfr.append(datafram,ignore_index=True)
	print(datafram)
	print(data[number_data])
	return datafram

def add_acc(data,data1):
	print(data1)
	if data1 == "AntMan":
		data[0] += 1
	if data1 == "BlackPanter":
		data[1] += 1
	if data1 == "BlackWidow":
		data[2] += 1
	if data1 == "Captain":
		data[3] += 1
	if data1 == "DrS":
		data[4] += 1
	if data1 == "Hulk":
		data[5] += 1
	if data1 == "IronMan":
		data[6] += 1
	if data1 == "Loki":
		data[7] += 1
	if data1 == "Spiderman":
		data[8] += 1
	if data1 == "StarLord":
		data[9] += 1
	if data1 == "SW":
		data[10] += 1
	if data1 == "Thor":
		data[11] += 1
	if data1 == "Vision":
		data[12] += 1
	return data

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face recognizer...")
imagePaths = list(paths.list_images(args["dataset"]))
label_name = ""
for (i, imagePath) in enumerate(imagePaths):
    # load our serialized face detector from disk
    # print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
    	"res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    print("i : "+str(i))
    print("Path :"+str(imagePath))

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
    	cv2.resize(image, (300, 300)), 1.0, (300, 300),
    	(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]

    	# filter out weak detections
    	if confidence > args["confidence"]:
    		# compute the (x, y)-coordinates of the bounding box for the
    		# face
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    		# extract the face ROI
    		face = image[startY:endY, startX:endX]
    		(fH, fW) = face.shape[:2]
    		# ensure the face width and height are sufficiently large
    		if fW < 20 or fH < 20:
    			continue
    		# construct a blob for the face ROI, then pass the blob
    		# through our face embedding model to obtain the 128-d
    		# quantification of the face
    		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
    			(0, 0, 0), swapRB=True, crop=False)
    		embedder.setInput(faceBlob)
    		vec = embedder.forward()
    		# perform classification to recognize the face
    		preds = recognizer.predict_proba(vec)[0]
    		j = np.argmax(preds)
    		proba = preds[j]
    		name = le.classes_[j]
    		# draw the bounding box of the face along with the associated
    		# probability
    		text = "{}: {:.2f}%".format(name, proba * 100)
            #     vision += 1
    		datalist = add_acc(datalist,name)
    		#df = add_data(datalist,df,imagePath)
    		y = startY - 10 if startY - 10 > 10 else startY + 10
    		cv2.rectangle(image, (startX, startY), (endX, endY),
    			(0, 0, 255), 2)
    		cv2.putText(image, text, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    imagePath = imagePath.split(os.path.sep)[-2]
    if imagePath == "BlackWidow" and antman == 0:
        df = add_data(datalist,df,"AntMan",0)
        antman += 1
        datalist = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if imagePath == "Captain" and blackwidow == 0:
        df = add_data(datalist,df,"BlackWidow",2)
        blackwidow += 1
        datalist = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if imagePath == "Hulk" and captain == 0:
        df = add_data(datalist,df,"Captain",3)
        captain += 1
        datalist = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if imagePath == "SW" and hulk == 0:
        df = add_data(datalist,df,"Hulk",5)
        hulk += 1
        datalist = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if imagePath == "Thor" and sw == 0:
        df = add_data(datalist,df,"SW",10)
        sw += 1
        datalist = [0,0,0,0,0,0,0,0,0,0,0,0,0]
print(datalist)
df = add_data(datalist,df,"Thor",11)
df.to_csv('avg_predict.csv')
