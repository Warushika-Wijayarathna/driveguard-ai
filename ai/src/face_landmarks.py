import time

import cv2
import mediapipe as mp


# MediaPipe modules used for detecting a face and its landmarks.
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def main() -> None:
    # MSMF is the Windows camera backend that worked on your machine.
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not camera.isOpened():
        raise RuntimeError("Could not open the default webcam.")

    previous_time = time.perf_counter()

    # FaceDetection gives us a face detection confidence score.
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,             # 0 is designed for faces near the camera
        min_detection_confidence=0.5,
    )

    # FaceMesh finds detailed landmark points on the face.
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,       # Process a continuous video stream
        max_num_faces=1,               # We only monitor one driver
        refine_landmarks=True,         # Adds more accurate eye landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Styles used when drawing eye and mouth landmarks.
    landmark_style = mp_drawing.DrawingSpec(
        color=(0, 255, 0),
        thickness=1,
        circle_radius=1,
    )

    connection_style = mp_drawing.DrawingSpec(
        color=(0, 200, 255),
        thickness=1,
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read a frame from the webcam.")
                break

            # OpenCV provides BGR images, while MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect the face and calculate its confidence.
            detection_result = face_detector.process(rgb_frame)

            # Detect detailed facial landmarks.
            mesh_result = face_mesh.process(rgb_frame)

            face_detected = bool(detection_result.detections)
            landmarks_detected = bool(mesh_result.multi_face_landmarks)

            if face_detected and landmarks_detected:
                # We configured the program to detect only one face.
                detection = detection_result.detections[0]
                face_landmarks = mesh_result.multi_face_landmarks[0]

                # MediaPipe returns the detector's confidence as a value
                # between 0 and 1.
                confidence = detection.score[0]

                cv2.putText(
                    frame,
                    f"Face confidence: {confidence:.2f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Draw the right eye outline.
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_style,
                )

                # Draw the left eye outline.
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_style,
                )

                # Draw the lips/mouth outline.
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_style,
                )

            else:
                # Show a visible warning whenever detection or tracking fails.
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # Calculate the current FPS.
            current_time = time.perf_counter()
            elapsed_time = current_time - previous_time
            previous_time = current_time

            fps = 1 / elapsed_time if elapsed_time > 0 else 0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("DriveGuard AI - Face Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

    finally:
        # Close MediaPipe resources.
        face_detector.close()
        face_mesh.close()

        # Release the camera and close the OpenCV window.
        camera.release()
        cv2.destroyAllWindows()

        print("Camera and MediaPipe resources released safely.")


if __name__ == "__main__":
    main()