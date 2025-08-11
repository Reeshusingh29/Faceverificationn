import cv2

# Webcam ko open karo (usually 0 hota hai default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera se image capture nahi ho paaya")
        break

    # Frame ko grayscale me convert karo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Grayscale frame ko window me dikhao
    cv2.imshow('Grayscale Camera', gray)

    # 'q' press karne par loop break ho jaaye
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Grayscale image ko save karna hai toh yahan kar sakte ho
        cv2.imwrite('captured_gray_image.jpg', gray)
        print("Image saved!")
        break

# Camera release karo aur windows band karo
cap.release()
cv2.destroyAllWindows()
