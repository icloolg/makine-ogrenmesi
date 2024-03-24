import cv2
import numpy as np
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions  #1000 farklı nesneyi tanır imagenet veri setinde eğitilmiştir
#preprocess_input:görüntüyü vgg16ya göre ölçekler  decode_predictions:vgg16nın ürettiği tahminleri  insanlar tarafından anlaşılır etiketlere dönüştürür
from keras.preprocessing import image

model=VGG16(weights="imagenet")
cap = cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break

    x=cv2.resize(frame,(224,224))
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)

    predictions=model.predict(x)
    label=decode_predictions(predictions,top=1)[0][0]
    label_name,label_cinfidence=label[1],label[2]

    cv2.putText(frame,f'{label_name}({label_cinfidence*100:.2f}%)',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("vgg",frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
