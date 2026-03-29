import cv2
import numpy as np
import mediapipe as mp
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.65, min_track_confidence=0.65)
cap = cv2.VideoCapture(0)
chidori_cap = cv2.VideoCapture("chidori.mp4")
rasen_cap = cv2.VideoCapture("rasenshuriken.mp4")
pwr = [0, 0]
was_open = [False, False]

def is_hand_open(landmarks):
    count=0
    wrist=landmarks[0]
    tips=[8,12,16,20]
    pips=[6,10,14,18]
    for tip, pip in zip(tips, pips):
        tip_pt=landmarks[tip]
        pip_pt=landmarks[pip]
        tip_dist=np.linalg.norm(
            np.array([tip_pt.x, tip_pt.y])-np.array([wrist.x, wrist.y])
        )
        pip_dist = np.linalg.norm(
            np.array([pip_pt.x, pip_pt.y])-np.array([wrist.x, wrist.y])
        )
        if tip_dist>pip_dist:
            count+=1
    return count>=3
def overlay_effect(frame,effect_frame,x,y,size=200):
    if effect_frame is None:
        return frame
    effect_frame=cv2.resize(effect_frame,(size, size))
    h, w, _=frame.shape

    x1=max(0,x-size//2)
    y1=max(0,y-size//2)
    x2=min(w,x+size//2)
    y2=min(h,y+size//2)

    effect_crop=effect_frame[0:y2 - y1, 0:x2 - x1]
    if effect_crop.shape[0] == 0 or effect_crop.shape[1] == 0:
        return frame
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(effect_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(effect_crop, effect_crop, mask=mask)
    final = cv2.add(bg, fg)
    frame[y1:y2, x1:x2] = final
    return frame
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    ret1, chidori_frame = chidori_cap.read()
    ret2, rasen_frame = rasen_cap.read()
    if not ret1:
        chidori_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, chidori_frame = chidori_cap.read()
    if not ret2:
        rasen_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, rasen_frame = rasen_cap.read()
    found_left = False
    found_right = False
    if result.multi_hand_landmarks and result.multi_handedness:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[i].classification[0].label
            is_right = label == "Right"
            i = 1 if is_right else 0
            landmarks = hand_landmarks.landmark
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            open_hand = is_hand_open(landmarks)
            if open_hand:
                pwr[i] += 0.05
            else:
                pwr[i] -= 0.15

            pwr[i] = np.clip(pwr[i], 0, 1)

            wrist=landmarks[0]
            x=int(wrist.x * w)
            y=int(wrist.y * h)
            if pwr[i] > 0.01:
                size = int(250 * pwr[i])
                if is_right:
                    found_right = True
                    frame = overlay_effect(frame, chidori_frame, x, y, size)
                else:
                    found_left = True
                    frame = overlay_effect(frame, rasen_frame, x, y, size)
    if not found_left:
        pwr[0] = max(0, pwr[0] - 0.15)
    if not found_right:
        pwr[1] = max(0, pwr[1] - 0.15)
    cv2.imshow("Chakra Vision", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
chidori_cap.release()
rasen_cap.release()
cv2.destroyAllWindows()