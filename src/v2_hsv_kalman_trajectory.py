import cv2
import numpy as np
from collections import deque


#  TRACKER V2 — HSV + Trajectoire (Kalman Filter)
#  Cliquez sur l'objet à tracker
#  'r' → réinitialiser 'q' → quitter

SOURCE =0
tolerance_h =15
tolerance_s = 45
tolerance_v =45

TRAIL_LENGTH = 40   # nombre de positions conservées pour dessiner la trainée

# ── Variables globales 
target_hsv  = None
tracking = False
frame_raw   = None

# File des dernières positions détectées : [(cx, cy), ...]
trail =deque(maxlen=TRAIL_LENGTH)

# Filtre de Kalman (initialisé au premier clic)
kalman = None


#  Kalman 
def create_kalman():
    """
    Crée un filtre de Kalman 2D qui modélise :
      - l'état  : [x, y, vx, vy]  (position + vitesse)
      - la mesure : [x, y]         (ce qu'on observe vraiment)

    Le Kalman a deux rôles :
      1. PRÉDIRE où sera l'objet à la frame suivante (même si on ne le voit pas)
      2. CORRIGER cette prédiction quand on a une vraie détection
    """
    kf = cv2.KalmanFilter(4, 2)   # 4 variables d'état, 2 variables mesurées

    # Matrice de transition : comment l'état évolue entre deux frames
    # x_new  = x  + vx        (position + vitesse * 1 frame)
    # y_new  = y  + vy
    # vx_new = vx             (on suppose vitesse constante)
    # vy_new = vy
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)

    # Matrice d'observation : on ne mesure que x et y (pas vx, vy)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]], dtype=np.float32)

    # Bruit du processus : à quel point on fait confiance au modèle physique
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # Bruit de mesure : à quel point on fait confiance au détecteur HSV
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    # Covariance initiale de l'erreur
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


def kalman_update(cx, cy):
    """Donne une nouvelle mesure au Kalman -> retourne la position corrigée."""
    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    return int(prediction[0].item()), int(prediction[1].item())


def kalman_predict_only():
    """Quand l'objet est perdu -> prédit la position sans mesure réelle."""
    prediction = kalman.predict()
    return int(prediction[0].item()), int(prediction[1].item())


# Souris 

def on_mouse_click(event, x, y, flags, param):
    global target_hsv, tracking, kalman, trail

    if event ==cv2.EVENT_LBUTTONDOWN:
        w = frame_raw.shape[1]
        h_img = frame_raw.shape[0]
        x_real = w - x

        region = 20
        y1 =max(0, y - region)
        y2 = min(h_img, y + region)
        x1 =max(0, x_real - region)
        x2 = min(w, x_real + region)

        region_bgr = frame_raw[y1:y2, x1:x2]
        region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        target_hsv = np.mean(region_hsv, axis=(0, 1))

        # Initialise le Kalman avec la position du clic
        kalman = create_kalman()
        kalman.statePre = np.array([[np.float32(x)], [np.float32(y)],[np.float32(0)], [np.float32(0)]])
        kalman.statePost = kalman.statePre.copy()

        trail.clear()
        tracking = True

        h, s, v = int(target_hsv[0].item()), int(target_hsv[1].item()), int(target_hsv[2].item())
        print(f"[CLIC] H={h}  S={s}  V={v}")


#  Masque HSV 

def build_mask(hsv_frame, target):
    h,s,v = int(target[0].item()), int(target[1].item()), int(target[2].item())

    lower = np.array([max(0, h - tolerance_h),max(0, s - tolerance_s),max(0, v - tolerance_v)])
    upper =np.array([min(180, h + tolerance_h),min(255, s + tolerance_s),min(255, v + tolerance_v)])

    mask =cv2.inRange(hsv_frame, lower, upper)

    if h < tolerance_h:
        l2 = np.array([180 - (tolerance_h - h), lower[1], lower[2]])
        u2 = np.array([180, upper[1], upper[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, l2, u2))
    elif h > 180 - tolerance_h:
        l2 = np.array([0, lower[1], lower[2]])
        u2 = np.array([tolerance_h - (180 - h), upper[1], upper[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, l2, u2))

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


#  Détection 

def find_object(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours,key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 500:
        return None

    x, y, w, h = cv2.boundingRect(best)
    cx =x + w//2
    cy =y + h//2
    return cx, cy, w, h, int(area)


#  Dessin 

def draw_trail(frame):
    """
    Dessine la trainée de positions passées.
    Les points récents sont plus lumineux et plus épais que les anciens.
    """
    points = list(trail)
    for i in range(1, len(points)):
        # Opacité proportionnelle à l'ancienneté : plus récent = plus visible
        alpha = i / len(points)
        color = (0, int(255 * alpha), int(120 * alpha))
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, points[i - 1], points[i], color, thickness)


def draw_overlay(frame, cx, cy, w, h, area, predicted=False):
    color =(0, 180, 255) if predicted else (0, 255, 120)
    label ="PREDICTION" if predicted else "TRACKING"

    cv2.rectangle(frame, (cx - w//2, cy - h//2),
                  (cx + w//2, cy + h//2), color, 2)
    cv2.circle(frame, (cx, cy), 6, color, -1)

    arm = 20
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, 1)

    cv2.putText(frame, f"({cx}, {cy})",(cx + 12, cy - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, f"{label}  px={area}",(10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


#  Boucle principale 

def main():
    global frame_raw, target_hsv, tracking, trail

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la source : {SOURCE}")
        return

    print(" TRACKER V2 DÉMARRÉ ")
    print("  Cliquez sur l'objet à suivre")
    print("  'r' → réinitialiser  |  'q' → quitter")

    win = "Tracker V2 — HSV + Trajectoire"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse_click)

    # Dernière taille connue de la bounding box (pour la prédiction)
    last_w, last_h, last_area = 60, 60, 3600

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mirror = cv2.flip(frame, 1)
        frame_raw  = frame
        display = frame_mirror.copy()

        if tracking and target_hsv is not None:
            hsv  = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2HSV)
            mask  = build_mask(hsv, target_hsv)
            result = find_object(mask)

            if result:
                #  Détection réelle : on corrige le Kalman 
                cx_raw, cy_raw, w, h, area = result
                cx, cy = kalman_update(cx_raw, cy_raw)
                last_w, last_h, last_area = w, h, area

                trail.append((cx, cy))
                draw_trail(display)
                draw_overlay(display, cx, cy, w, h, area, predicted=False)

            else:
                #  Objet perdu : le Kalman prédit la position 
                cx,cy = kalman_predict_only()
                trail.append((cx, cy))
                draw_trail(display)
                draw_overlay(display, cx, cy,
                             last_w, last_h, last_area, predicted=True)

        else:
            cv2.putText(display, "Cliquez sur l'objet a tracker",(10, 24), cv2.FONT_HERSHEY_SIMPLEX,0.55, (200, 200, 200), 1)

        cv2.imshow(win, display)

        key =cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            target_hsv = None
            tracking = False
            trail.clear()
            print("[RESET]")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
