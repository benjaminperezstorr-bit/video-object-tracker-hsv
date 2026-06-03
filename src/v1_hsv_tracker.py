
"""
HSV Video Object Tracker

This script implements a simple real-time object tracker using HSV color segmentation.
The user clicks on an object in the webcam stream, and the program tracks regions
with a similar HSV color.

Author: Perez-Storr Benjamin
"""

import cv2
import numpy as np


tolerance_h = 15
tolerance_s = 45
tolerance_v = 45

SOURCE = 0

target_hsv = None
tracking = False
frame_raw = None




def on_mouse_click(event, x, y, flags, param):
    global target_hsv, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        w = frame_raw.shape[1]
        x_real = w-x
        pixel_gbr = frame_raw[y, x_real]
        pixel_hsv = cv2.cvtColor(
            np.uint8([[pixel_gbr]]), cv2.COLOR_BGR2HSV
        )[0][0]
        target_hsv = pixel_hsv
        tracking = True
        h,s,v = int(target_hsv[0]),int(target_hsv[1]), int(target_hsv[2])
        print(f"H = {h}, S = {s}, V = {v}")

def build_mask(hsv_frame,target):
    h,s,v = int(target[0]),int(target[1]), int(target[2])
    upper = np.array([min(180, h+tolerance_h), min(255, s+tolerance_s), min(255, v+tolerance_v)])
    lower = np.array([max(0, h-tolerance_h), max(0, s-tolerance_s), max(0, v-tolerance_v)])

    mask = cv2.inRange(hsv_frame, lower, upper)
    # ── Cas particulier : rouge (teinte qui boucle autour de 0/180)
    if h < tolerance_h:
        lower2 = np.array([180 - (tolerance_h - h), lower[1], lower[2]])
        upper2 = np.array([180, upper[1], upper[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower2, upper2))
    elif h > 180 - tolerance_h:
        lower2 = np.array([0, lower[1], lower[2]])
        upper2 = np.array([tolerance_h - (180 - h), upper[1], upper[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower2, upper2))

    # Nettoyage morphologique : supprime le bruit et bouche les trous
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)


    return mask


def find_object(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # On garde le plus grand contour (c'est probablement l'objet)
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)

    if area < 500:          # trop petit = bruit → on ignore
        return None


    x, y, w, h = cv2.boundingRect(best)
    cx = x + w // 2
    cy = y + h // 2

    return cx, cy, w, h, int(area)


def draw_overlay(frame, result, target_hsv):
    cx, cy, w, h, area = result

    # Bounding box
    cv2.rectangle(frame, (cx - w//2, cy - h//2),
                  (cx + w//2, cy + h//2), (0, 255, 120), 2)

    # Cercle central
    cv2.circle(frame, (cx, cy), 6, (0, 255, 120), -1)

    # Réticule (crosshair)
    arm = 20
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (0, 255, 120), 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (0, 255, 120), 1)

    # Infos texte
    cv2.putText(frame, f"({cx}, {cy})",
                (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)

    # Barre d'état en haut à gauche
    hue = int(target_hsv[0])
    cv2.putText(frame, f"TRACKING  H={hue}  px={area}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)


def main():
    global frame_raw, target_hsv, tracking

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la source vidéo : {SOURCE}")
        return

    print("=== TRACKER DÉMARRÉ ===")
    print("  Cliquez sur l'objet à suivre")
    print("  'r' → réinitialiser  |  'q' → quitter")

    win = "Tracker — cliquez pour sélectionner"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[FIN] Flux vidéo terminé.")
            break

        # Miroir horizontal (plus naturel avec une webcam)
        frame_mirror = cv2.flip(frame, 1)
        frame_raw    = frame           # non-miroir pour lire le bon pixel

        display = frame_mirror.copy()

        if tracking and target_hsv is not None:
            hsv   = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2HSV)
            mask  = build_mask(hsv, target_hsv)
            result = find_object(mask)

            if result:
                draw_overlay(display, result, target_hsv)
            else:
                cv2.putText(display, "OBJET PERDU",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 60, 255), 2)
        else:
            cv2.putText(display, "Cliquez sur l'objet a tracker",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1)

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            target_hsv = None
            tracking   = False
            print("[RESET] Tracker réinitialisé.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()