# OpenCV provides tools for working with cameras, images, and video.
import cv2

# The time module lets us measure how long each frame takes.
import time


def main() -> None:
    # Open the default webcam.
    # Camera index 0 usually represents the built-in or primary webcam.
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

    # Check whether OpenCV successfully opened the webcam.
    if not camera.isOpened():
        raise RuntimeError("Could not open the default webcam.")

    # Record the starting time for our first FPS calculation.
    previous_time = time.perf_counter()

    try:
        # Keep reading frames until the user presses Q
        # or OpenCV fails to read from the webcam.
        while True:
            # camera.read() returns:
            # success: True if a frame was captured
            # frame:   The captured image
            success, frame = camera.read()

            # Stop if the webcam did not provide a valid frame.
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            # Record the current time.
            current_time = time.perf_counter()

            # Calculate how many seconds passed between two frames.
            elapsed_time = current_time - previous_time

            # Save this time for the next loop iteration.
            previous_time = current_time

            # FPS means frames per second.
            # For example, if one frame takes 0.05 seconds:
            # FPS = 1 / 0.05 = 20
            #
            # We check elapsed_time to avoid division by zero.
            fps = 1 / elapsed_time if elapsed_time > 0 else 0

            # Draw the calculated FPS on the webcam frame.
            cv2.putText(
                frame,                          # Image to draw on
                f"FPS: {fps:.1f}",              # Text with one decimal place
                (20, 40),                       # Text position: x=20, y=40
                cv2.FONT_HERSHEY_SIMPLEX,        # Font
                1,                              # Font size
                (0, 255, 0),                    # Green in BGR format
                2,                              # Text thickness
            )

            # Display the current frame in a window.
            cv2.imshow("DriveGuard AI - Camera Test", frame)

            # Wait one millisecond for keyboard input.
            # Exit when the user presses lowercase or uppercase Q.
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

    finally:
        # This section runs whether the loop ends normally
        # or an unexpected error occurs.

        # Give control of the webcam back to Windows.
        camera.release()

        # Close every window created by OpenCV.
        cv2.destroyAllWindows()

        print("Camera released safely.")


# Run main() only when this file is executed directly.
#
# It will not run automatically if another Python file
# imports camera_test.py in the future.
if __name__ == "__main__":
    main()