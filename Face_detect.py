import cv2
import os

# start capturing video
cap  = cv2.VideoCapture(0)
i = 0

while (cap.isOpened()):
    #store it in frames
    ret, frame = cap.read()

    
    # If video ends break loop
    if ret == False:
        break

    # Save Frame by Frame into folder using imwrite method
    #create Data folder if not present, our frames will be saved inside this folder.
    path = "Data"
    a = 'Frame' + str(i) + '.jpg'
    # cv2.imwrite(path, a, frame)
    cv2.imwrite(os.path.join(path, a), frame)
    i += 1
    #we will take first 20 frames from video
    if i==20:
        cap.release()

cv2.destroyAllWindows()

#Detecting Face
#select any frame from data folder to detect face
img_path = 'Data\Frame5.jpg'
image= cv2.imread(img_path)
#conver image to gray scale
img2 = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#call haarcascade_frontalface classifier to get the face co-odinates
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
    )
print(faces)
print("Found {0} Faces".format(len(faces)))
faces1=[]

#extract face
for (x, y, w, h) in faces:
        #multiple faces can be stored inside faces1 if detected.
        faces1.append(img2[y-75: y+h+55, x-75: x + w+55])
        

for i in faces1:
    #save our detetcted face in jpg format
    status = cv2.imwrite('faces_detected.jpg',i)
    print(status)
    break      #only first face will be saved incase of multiple face detected
