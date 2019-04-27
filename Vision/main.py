import io
import os
import time
from google.cloud import vision
from google.cloud.vision import types
import cv2


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(os.getcwd(),"credentials.json")
client=vision.ImageAnnotatorClient()

def get_labels(image):
    with io.open(image,'rb') as f:
        content=f.read()
    img=types.Image(content=content)
    response=client.label_detection(image=img)
    return response


start=time.time()
cap=cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    cv2.imshow('frame', rgb) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if (time.time()-start) > 5:
        fname=str(time.time())+".jpg"
        out = cv2.imwrite(fname, frame)
        print(get_labels(os.path.join(os.getcwd(),fname)))
        start=time.time()
        
cap.release()
cv2.destroyAllWindows()
for file in os.listdir(os.getcwd()):
    if file.endswith(".jpg"):
        os.remove(file)