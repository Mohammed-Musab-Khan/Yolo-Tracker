import cv2
from ultralytics import YOLO

# You can combine activity 1 and 2 into 1 single program.

# Function to predict and print object information without drawing rectangles
def object_predict1():
    # Load YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Predict objects in the image
    results = model.predict("cat_dog.jpg")
    result = results[0]

    # Print class names and number of detected boxes
    print(result.names)
    print(len(result.boxes))

    # Iterate over detected boxes and print object information
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        # Print object details
        print("Object type:", class_id)
        print("Coordinates:", cords)
        print("Probability:", conf)
        print("---")


# Function to predict and draw rectangles on detected objects
def object_predict2():
    # Load YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Read input image
    image = cv2.imread("cat_dog.jpg")

    # Predict objects in the image
    results = model.predict("cat_dog.jpg")
    result = results[0]

    # Iterate over detected boxes and draw rectangles with labels
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        # Check if probability exceeds 0.5
        if conf > 0.5:
            # Draw bounding box and label on the image
            x1, y1, x2, y2 = cords
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_id} ({conf})"
            image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vehicle_tracking():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    video_path = "highway.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# vehicle_tracking
def vehicle_tracking_assignment():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file 
    video_path = "highway.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Iterate over detected objects
            for box in results[0].boxes:
                conf = box.conf[0].item()
                if conf >= 0.5:  # Check probability threshold
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    x1, y1, x2, y2 = cords
                    class_id = results[0].names[box.cls[0].item()]
                    label = f"{class_id} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


object_predict1()
# object_predict2()
# vehicle_tracking()
# vehicle_tracking_assignment()