import cv2
import numpy as np
import time
import os

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

make_dir('data')

classes = ['others', 'paper', 'rock', 'scissors']

for cls in classes:
    make_dir('data/' + cls)

def save(d_name, frame):
    frame = cv2.resize(frame, dsize=(150, 150))
    cv2.imwrite('data/' + str(d_name) + '/' + str(time.time()) + '.jpg', frame)


from trained_model import trained_model
model = trained_model()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while(True):
    ret, frame = cap.read()
    frame = frame[:,80:560]
    frame = cv2.resize(frame, dsize=(150, 150))
    frame = frame[:,::-1] #mirror
    img_save = np.copy(frame)
    img = frame[:,:,::-1] #rgb
    pred = model.predict(img)
    
    frame = cv2.resize(frame, dsize=(500, 500))
    cv2.rectangle(frame, (300, 0), (500, 20), (0, 0, 0), -1)
    cv2.putText(frame, pred, (305, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xff))
    
    cv2.imshow('window', frame)
    key_id = cv2.waitKey(1)
    key_id = key_id & 0xFF
    if key_id == ord('q'):
        break
    elif key_id == ord('r'):
        save('rock', img_save)
    elif key_id == ord('s'):
        save('scissors', img_save)
    elif key_id == ord('p'):
        save('paper', img_save)
    elif key_id == ord('o'):
        save('others', img_save)

cap.release()
cv2.destroyAllWindows()





