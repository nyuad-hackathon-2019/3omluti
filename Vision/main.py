import io
import os
import time
from google.cloud import vision
from google.cloud.vision import types
import cv2
import tkinter
import pyqrcode

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(os.getcwd(),"credentials.json")
client=vision.ImageAnnotatorClient()

def get_labels(image):
    with io.open(image,'rb') as f:
        content=f.read()
    img=types.Image(content=content)
    response=client.label_detection(image=img)
    return response

def write_to_file(response,time):
    with open("log.txt","a") as f:
        f.write(time)
        f.write('\n-----------\n')
        f.write(str(response)+'\n')

def match_label(response):
    descriptions=["Bottle","Plastic bottle","Bottled water","Water bottle","Mineral water","Plastic"]
    papers=["Paper","Origami","Paper product","Text"]
    for label in response.label_annotations:
        if label.description in descriptions:
            return ("Plastic",label.description)
        elif label.description in papers:
            return ("Paper",label.description)
    return False
    
def display_qr(product_type,product_name):
    scoring_dict={'Plastic':1,"Paper":2}
    qr=pyqrcode.create(scoring_dict[product_type])
    qr_xbm=qr.xbm(scale=30)
    top=tkinter.Tk()
    top.attributes('-fullscreen',True)
    bmp=tkinter.BitmapImage(data=qr_xbm)
    bmp.config(background="white")
    label = tkinter.Label(image=bmp)
    label.pack()
    text=tkinter.Label(text=product_type,font=("Helvetica", 30))
    text.pack()
    top.after(10000,lambda:top.destroy())
    top.mainloop()
    
def capture_webcam(capture_time):
    start=time.time()
    cap=cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        cv2.imshow('frame', rgb) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time()-start > capture_time:
            fname=str(time.time())+".jpg"
            out = cv2.imwrite(fname, frame)
            response=get_labels(os.path.join(os.getcwd(),fname))
            write_to_file(response,fname)
            if match_label(response):
                label_type,label=match_label(response)
                display_qr(label_type,label)
                break
            start=time.time()

    cap.release()
    cv2.destroyAllWindows()
    
    '''
    for file in os.listdir(os.getcwd()):
        if file.endswith(".jpg"):
            os.remove(file)
    '''     
    
if __name__ == "__main__":
    capture_webcam(0.1)