import cv2

# Open the camera (0 is usually the default, try 1 or 2 if you have multiple cameras)
cap = cv2.VideoCapture(2)

# Set the resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Adjust for downward orientation: rotate 180 degrees if the feed is upside down
    # Options: cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Display the live feed
    cv2.imshow("Downward Arducam Feed", rotated_frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()