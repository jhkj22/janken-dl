import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time

hand4 = ['rock', 'scissors', 'paper', 'others']
illust4 = []

rock = cv2.imread('../res/hand_illust/rock2.png')
rock = cv2.resize(rock, (250, 250))
illust4.append(rock)
scissors = cv2.imread('../res/hand_illust/scissors2.png')
scissors = cv2.resize(scissors, (250, 250))
illust4.append(scissors)
paper = cv2.imread('../res/hand_illust/paper2.png')
paper = cv2.resize(paper, (250, 250))
illust4.append(paper)
others = cv2.imread('../res/hand_illust/others2.png')
others = cv2.resize(others, (250, 250))
illust4.append(others)

illust_game = []

win = cv2.imread('../res/hand_illust/win.png')
win = cv2.resize(win, (220, 80))
illust_game.append(win)

lose = cv2.imread('../res/hand_illust/lose.png')
lose = cv2.resize(lose, (220, 80))
illust_game.append(lose)

draw = cv2.imread('../res/hand_illust/draw.png')
draw = cv2.resize(draw, (220, 80))
illust_game.append(draw)

img_bar_b = cv2.imread('../res/hand_illust/bar.png')
img_bar_b = cv2.resize(img_bar_b, (230, 30))
img_bar_f = cv2.imread('../res/hand_illust/bar_fill.png')
img_bar_f = cv2.resize(img_bar_f, (230, 30))


from threading import Thread

class hand_predictor(Thread):
    def __init__(self, callback):
        Thread.__init__(self)
        self.callback = callback
    
    def run(self):
        time.sleep(1)
        t = random.randrange(4)
        self.callback(t)

def judge_janken(h_comp, h_human):
    if h_comp == h_human:
        return State.DRAW
    if ((h_comp == 0 and h_human == 1) or
        (h_comp == 1 and h_human == 2) or
        (h_comp == 2 and h_human == 0)):
        return State.COMP_WIN
    return State.COMP_LOSE

class State:
    OTHERS = 0
    START = 1
    JUDGE = 2
    COMP_WIN = 3
    COMP_LOSE = 4
    DRAW = 5

    WAIT_TIME = 2
    
    def __init__(self):
        self.state = State.OTHERS
        self.hand_human = 3
        self.hand_comp = 3
        self.rock_count = 0
        self.start_time = -1
        self.is_run = False

    def update(self, data):
        h_human = data
        if self.state == State.OTHERS:
            if h_human == 0:
                self.rock_count += 1
            if self.rock_count == 2:
                self.start_time = time.time()
                self.rock_count = 0
                self.state = State.START
            self.hand_human = h_human
        elif self.state == State.JUDGE:
            if h_human < 3:
                h_comp = random.randrange(3)
                self.hand_comp = h_comp
                self.state = judge_janken(h_comp, h_human)
            self.hand_human = h_human
        self.is_run = False

state = State()

while(True):
    if not state.is_run:
        hand_predictor(state.update).start()
        state.is_run = True
    
    frame = np.zeros((500, 750, 3), dtype=np.uint8)

    t_human = state.hand_human
    t_comp = state.hand_comp
    capture_img = np.ones((500, 500, 3), dtype=np.uint8) * t_human * 40
    frame[0: 500, 250: 750] = capture_img
    illust_comp = cv2.resize(illust4[t_comp], (250, 250))
    frame[0: 250, 0: 250] = illust_comp
    illust_me = cv2.resize(illust4[t_human], (250, 250))
    frame[250: 500, 0: 250] = illust_me

    if state.state == State.START:
        elapsed_time = time.time() - state.start_time
        ratio = elapsed_time / state.WAIT_TIME
        if ratio > 1:
            state.state = State.JUDGE
            state.start_time = -1
        else:
            rect_width = int(250 * ratio)
            cv2.rectangle(frame,
                          (0, 245), (rect_width, 255), (0, 0, 0xff), -1)
            #img_bar = np.copy(img_bar_b)
            #img_bar[:,:rect_width] = img_bar_f[:,:rect_width]
            #frame[450: 480, 10: 240] = img_bar
    elif state.state == State.COMP_WIN:
        if state.start_time < 0:
            state.start_time = time.time()
        elapsed_time = time.time() - state.start_time
        ratio = elapsed_time / state.WAIT_TIME
        if ratio > 1:
            state.state = State.OTHERS
            state.start_time = -1
            state.hand_comp = 3
        else:
            frame[210: 290, 15: 235] = illust_game[1]
    elif state.state == State.COMP_LOSE:
        if state.start_time < 0:
            state.start_time = time.time()
        elapsed_time = time.time() - state.start_time
        ratio = elapsed_time / state.WAIT_TIME
        if ratio > 1:
            state.state = State.OTHERS
            state.start_time = -1
            state.hand_comp = 3
        else:
            frame[210: 290, 15: 235] = illust_game[0]
    elif state.state == State.DRAW:
        if state.start_time < 0:
            state.start_time = time.time()
        elapsed_time = time.time() - state.start_time
        ratio = elapsed_time / state.WAIT_TIME
        if ratio > 1:
            state.state = State.JUDGE
            state.start_time = -1
        else:
            frame[210: 290, 15: 235] = illust_game[2]
            rect_width = int(250 * ratio)
            cv2.rectangle(frame,
                          (0, 250), (rect_width, 260), (0, 0, 0xff), -1)
    
    cv2.imshow('window', frame)
    key_id = cv2.waitKey(1)
    key_id = key_id & 0xFF
    if key_id == ord('q'):
        break

cv2.destroyAllWindows()





