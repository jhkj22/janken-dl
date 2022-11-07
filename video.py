import cv2
from trained_model import trained_model

model = trained_model()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

classes = ['rock', 'scissors', 'paper', 'others']

while(True):
    #for i in range(4):
    #    cap.grab()
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:,80:560]
    frame = cv2.resize(frame, dsize=(150, 150))
    frame = frame[:,::-1]
    img = frame[:,:,::-1]
    i, v = model.predict(img)
    pred = '{:}: {:}'.format(classes[i], int(v * 100))
    #pred = ''
    
    frame = cv2.resize(frame, dsize=(500, 500))
    cv2.rectangle(frame, (300, 0), (500, 20), (0, 0, 0), -1)
    cv2.putText(frame, pred, (305, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xff))
    cv2.imshow('window', frame)
    key_id = cv2.waitKey(1)
    key_id = key_id & 0xFF
    if key_id == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





