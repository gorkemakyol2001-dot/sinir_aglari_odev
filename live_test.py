import cv2
import numpy as np
import tensorflow as tf
import os
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh


MODEL_PATH = os.path.join(os.path.dirname(__file__), "lies_detector_final.keras")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    mesh_detector = mp_face_mesh.FaceMesh(refine_landmarks=True)
except Exception as e:
    print(f"HATA: {e}"); exit()

CORE_LANDMARKS = [33, 133, 362, 263, 61, 291, 0, 17, 234, 454]

class ForensicPolygraph:
    def __init__(self):
        self.calibrated = False
        self.calib_frames = []
        self.mu, self.sigma = 0.5, 0.01
        self.recording = False
        self.session_buffer = []
        self.result_text = None

    def process(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            ih, iw, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh_detector.process(rgb)

       
            cv2.rectangle(frame, (0, 0), (iw, 45), (15, 15, 15), -1)
            cv2.putText(frame, "ADLI BIYOMETRIK ANALIZOR v14.0", (20, 30), 2, 0.6, (255, 255, 255), 1)
            
          
            cv2.rectangle(frame, (0, ih-40), (iw, ih), (10, 10, 10), -1)
            instruction = "SPACE: ANALIZI BASLAT/BITIR  |  Q: SISTEMDEN CIK"
            cv2.putText(frame, instruction, (iw//2 - 180, ih-15), 1, 1, (200, 200, 200), 1)

            if res.multi_face_landmarks:
                for flm in res.multi_face_landmarks:
                    pts = [(int(p.x * iw), int(p.y * ih)) for p in flm.landmark]
                    for i in CORE_LANDMARKS: cv2.circle(frame, pts[i], 2, (0, 255, 255), -1)
                    
                    x_s = [pts[i][0] for i in CORE_LANDMARKS]; y_s = [pts[i][1] for i in CORE_LANDMARKS]
                    roi = frame[max(0, min(y_s)-35):min(ih, max(y_s)+35), max(0, min(x_s)-35):min(iw, max(x_s)+35)]

                    if roi.size > 0:
                        roi_in = cv2.resize(roi, (128, 128)) / 255.0
                        score = model.predict(np.expand_dims(roi_in, 0), verbose=0)[0][0]

                        if not self.calibrated:
                          
                            cv2.putText(frame, "BIOMETRIK PROFIL OLUSTURULUYOR...", (20, 80), 1, 1, (0, 255, 255), 1)
                            self.calib_frames.append(score)
                            cv2.rectangle(frame, (20, 95), (20 + int(len(self.calib_frames)*3.5), 105), (0, 255, 255), -1)
                            
                            if len(self.calib_frames) >= 60:
                                self.mu, self.sigma = np.mean(self.calib_frames), np.std(self.calib_frames) + 0.002
                                self.calibrated = True
                        
                        elif self.recording:
                           
                            cv2.putText(frame, "● KAYIT VE ANALIZ AKTIF", (20, 80), 1, 1, (0, 0, 255), 2)
                            self.session_buffer.append(score)
                            
                            z = abs(score - self.mu) / self.sigma
                            bar_w = int(min(100, (z / 1.5) * 100))
                            cv2.putText(frame, "MIMIK GERGINLIGI", (20, ih-65), 1, 0.8, (255, 255, 255), 1)
                            cv2.rectangle(frame, (20, ih-60), (20 + bar_w*3, ih-45), (0, 0, 255), -1)

       
            if self.result_text and not self.recording:
               
                overlay = frame.copy()
                cv2.rectangle(overlay, (iw-380, 70), (iw-20, 180), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
                color = (0, 255, 0) if "DOGRU" in self.result_text else (0, 0, 255)
                cv2.putText(frame, "ANALIZ RAPORU", (iw-360, 105), 1, 1.2, (200, 200, 200), 2)
                cv2.putText(frame, self.result_text, (iw-360, 155), 1, 1.5, color, 3)

            cv2.imshow('Neural Forensic v14.0', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and self.calibrated:
                if not self.recording:
                    self.recording, self.session_buffer, self.result_text = True, [], None
                else:
                    self.recording = False
                    if len(self.session_buffer) > 5:
                        final_z = (np.mean(self.session_buffer) - self.mu) / self.sigma
                        res = "YALAN" if final_z > 1.25 else "DOGRU"
                        self.result_text = f"KARAR: {res} (%{min(99.9, (abs(final_z)/3)*100):.1f})"
            elif key == ord('q'): break
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": ForensicPolygraph().process()