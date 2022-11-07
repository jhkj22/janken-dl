import cv2
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

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = frame[:,80:560]
    frame = frame[:,::-1]
    
    cv2.imshow('window', frame)
    key_id = cv2.waitKey(1)
    key_id = key_id & 0xFF
    if key_id == ord('q'):
        break
    elif key_id == ord('r'):
        save('rock', frame)
    elif key_id == ord('s'):
        save('scissors', frame)
    elif key_id == ord('p'):
        save('paper', frame)
    elif key_id == ord('o'):
        save('others', frame)

cap.release()
cv2.destroyAllWindows()





