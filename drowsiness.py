import cv2
import numpy as np
import face_recognition
import playsound
from scipy.spatial import distance as dist
from threading import Thread
import time
import json
from datetime import datetime

# Basic settings - I made these adjustable
MIN_EAR = 0.28
EYE_AR_CONSEC_FRAMES = 12
YAWN_THRESHOLD = 0.6

# Counters and flags
eye_counter = 0
yawn_counter = 0
alarm_playing = False
session_start_time = time.time()

# I added these to track more stuff
drowsy_episodes = 0
total_yawns = 0
ear_values = []  # to store ear values for analysis
yawn_times = []  # when yawns happened
alert_level = 0  # 0=normal, 1=mild, 2=moderate, 3=severe

def calculate_ear(eye):
    """Same EAR calculation but with better variable names"""
    # vertical distances
    vertical_1 = dist.euclidean(eye[1], eye[5])
    vertical_2 = dist.euclidean(eye[2], eye[4])
    # horizontal distance
    horizontal = dist.euclidean(eye[0], eye[3])
    # calculate ratio
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def calculate_mar(mouth):
    """I added this - calculates mouth aspect ratio for yawn detection"""
    # mouth height measurements
    height_1 = dist.euclidean(mouth[2], mouth[10])
    height_2 = dist.euclidean(mouth[4], mouth[8])
    # mouth width
    width = dist.euclidean(mouth[0], mouth[6])
    # calculate ratio
    mar = (height_1 + height_2) / (2.0 * width)
    return mar

def check_head_tilt(face_landmarks):
    """Simple head tilt detection - I added this feature"""
    if 'nose_tip' in face_landmarks and 'chin' in face_landmarks:
        nose_points = face_landmarks['nose_tip']
        chin_points = face_landmarks['chin']
        
        # get center points
        nose_center = np.mean(nose_points, axis=0)
        chin_center = np.mean(chin_points, axis=0)
        
        # calculate angle
        angle = np.arctan2(chin_center[1] - nose_center[1], 
                          chin_center[0] - nose_center[0])
        return np.degrees(angle)
    return 0

def determine_alert_level(ear, mar, head_angle):
    """I made this to combine all detection methods"""
    score = 0
    
    # check eye aspect ratio
    if ear < 0.20:
        score += 3
    elif ear < 0.25:
        score += 2
    elif ear < 0.30:
        score += 1
    
    # check if yawning
    if mar > YAWN_THRESHOLD:
        score += 2
    
    # check head position
    if abs(head_angle) > 15:
        score += 1
    
    # return alert level (max 3)
    return min(score, 3)

def play_alarm_sound():
    """Play alarm - same as before but with error handling"""
    try:
        playsound.playsound('alarm.wav')
    except:
        print("Couldn't play alarm sound - check if alarm.wav exists")

def draw_facial_landmarks(frame, face_landmarks):
    """Draw different colored outlines for face features"""
    # draw eyes in yellow
    if 'left_eye' in face_landmarks:
        left_eye_points = np.array(face_landmarks['left_eye'])
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 255), 2)
    
    if 'right_eye' in face_landmarks:
        right_eye_points = np.array(face_landmarks['right_eye'])
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 255), 2)
    
    # draw mouth in green
    if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
        mouth_points = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
        mouth_array = np.array(mouth_points)
        cv2.polylines(frame, [mouth_array], True, (0, 255, 0), 2)

def draw_stats_on_screen(frame, ear, mar, head_angle, current_alert_level):
    """Display all the information on screen"""
    height, width = frame.shape[:2]
    
    # main metrics
    cv2.putText(frame, f"Eye Ratio: {ear:.3f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Mouth Ratio: {mar:.3f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Head Angle: {head_angle:.1f}°", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # alert status with colors
    alert_messages = ["NORMAL", "MILD DROWSY", "MODERATE DROWSY", "SEVERE DROWSY"]
    alert_colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
    
    cv2.putText(frame, f"Status: {alert_messages[current_alert_level]}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_colors[current_alert_level], 2)
    
    # session stats on the right
    session_time = time.time() - session_start_time
    cv2.putText(frame, f"Time: {session_time/60:.1f} min", (width-200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Drowsy Count: {drowsy_episodes}", (width-200, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Yawns: {total_yawns}", (width-200, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def save_session_data():
    """Save session report to a file"""
    session_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration_minutes': (time.time() - session_start_time) / 60,
        'drowsy_episodes': drowsy_episodes,
        'total_yawns': total_yawns,
        'average_ear': np.mean(ear_values) if ear_values else 0,
        'min_ear': min(ear_values) if ear_values else 0,
        'yawn_frequency': len(yawn_times) / ((time.time() - session_start_time) / 60) if yawn_times else 0
    }
    
    filename = f"drowsiness_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(session_data, f, indent=2)
    print(f"Session data saved to {filename}")

def main():
    """Main function - I organized everything here"""
    global eye_counter, yawn_counter, alarm_playing, drowsy_episodes, total_yawns, alert_level
    
    # setup camera
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Enhanced Drowsiness Detection System")
    print("Features: Eye tracking, Yawn detection, Head pose, Statistics")
    print("Press 'q' to quit, 's' to save session data")
    print("Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # detect faces
        face_landmarks_list = face_recognition.face_landmarks(frame)
        
        if face_landmarks_list:
            for face_landmarks in face_landmarks_list:
                # get eye coordinates
                left_eye = face_landmarks['left_eye']
                right_eye = face_landmarks['right_eye']
                
                # calculate eye aspect ratio
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_values.append(avg_ear)  # store for analysis
                
                # calculate mouth aspect ratio for yawn detection
                mouth_ratio = 0
                if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
                    mouth_points = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
                    mouth_ratio = calculate_mar(mouth_points)
                    
                    # detect yawning
                    if mouth_ratio > YAWN_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter >= 3:  # confirmed yawn
                            total_yawns += 1
                            yawn_times.append(time.time() - session_start_time)
                            yawn_counter = 0
                            cv2.putText(frame, "YAWN DETECTED!", (200, 200),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # check head position
                head_angle = check_head_tilt(face_landmarks)
                
                # determine overall alert level
                alert_level = determine_alert_level(avg_ear, mouth_ratio, head_angle)
                
                # check for drowsiness (eye closure)
                if avg_ear < MIN_EAR:
                    eye_counter += 1
                    
                    if eye_counter >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_playing:
                            alarm_playing = True
                            drowsy_episodes += 1
                            
                            # play alarm in background
                            alarm_thread = Thread(target=play_alarm_sound)
                            alarm_thread.daemon = True
                            alarm_thread.start()
                        
                        # show big warning
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 300),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        
                        # red border
                        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)
                else:
                    eye_counter = 0
                    alarm_playing = False
                
                # draw face features
                draw_facial_landmarks(frame, face_landmarks)
                
                # display all stats
                draw_stats_on_screen(frame, avg_ear, mouth_ratio, head_angle, alert_level)
        
        else:
            cv2.putText(frame, "No face detected - Please position yourself in front of camera", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # show the frame
        cv2.imshow("Advanced Drowsiness Detection", frame)
        
        # handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            save_session_data()
            print("Session data saved!")
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # final summary
    print("\n=== Final Session Summary ===")
    print(f"Session Duration: {(time.time() - session_start_time)/60:.1f} minutes")
    print(f"Total Drowsy Episodes: {drowsy_episodes}")
    print(f"Total Yawns Detected: {total_yawns}")
    print(f"Average Eye Aspect Ratio: {np.mean(ear_values):.3f}" if ear_values else "No data")
    
    # auto-save session data
    save_session_data()

if __name__ == "__main__":
    main()