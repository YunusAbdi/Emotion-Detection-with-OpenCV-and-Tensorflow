from keras.models import load_model
import cv2
import numpy as np
import model

model = model.model()

model.load_weights("saves/model-BL-epoch-130- Acc-0.991685.hdf5")
face_clsfr=cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
eyes_clsfr = cv2.CascadeClassifier('Haarcascade/haarcascade_eye.xml')

source=cv2.VideoCapture(1)


# Prepare labels and colors
labels_dict={0:'Angry',1:'Disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
color_dict={0:(1, 5, 33),1:(1, 59, 53),2:(1, 59, 29),3:(76, 89, 0),4:(89, 19, 0),5:(65, 0, 89),6:(89, 0, 43)}



while(True):
    ret,img=source.read()
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.3,5)  
    eyes = eyes_clsfr.detectMultiScale(img,1.3,5)
    for x,y,w,h in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(48,48))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,48,48,3))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(
          img, labels_dict[label], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    xs = []
    ys = []
    for x,y,w,h in eyes:
        if(len(xs)>1): 
            slope = ys[1] - ys[0] / xs[1] - xs[0]
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()