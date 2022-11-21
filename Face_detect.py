import cv2

img_path = 'IMG_20221119_150935249_BURST001.jpg'
image= cv2.imread(img_path)
img2 = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
for (x, y, w, h) in faces:
        faces1.append(img2[y-75: y+h+55, x-75: x + w+55])


for i in faces1:
    status = cv2.imwrite('faces_detected.jpg',i)
    print(status)
    break