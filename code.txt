import streamlit as st
import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet(r"C:\Users\avani\OneDrive\Desktop\dataset\yolov3.weights", r"C:\Users\avani\OneDrive\Desktop\dataset\yolov3.cfg")
    classes = []
    with open(r"C:\Users\avani\OneDrive\Desktop\dataset\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getUnconnectedOutLayersNames()
    return net, classes, layer_names

def analyze_traffic_light_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([15, 120, 70])
    upper_yellow = np.array([35, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

    red_percentage = cv2.countNonZero(mask_red) / (roi.shape[0] * roi.shape[1])
    yellow_percentage = cv2.countNonZero(mask_yellow) / (roi.shape[0] * roi.shape[1])
    green_percentage = cv2.countNonZero(mask_green) / (roi.shape[0] * roi.shape[1])

    if max(red_percentage, yellow_percentage, green_percentage) == red_percentage:
        color = "Red"
    elif max(red_percentage, yellow_percentage, green_percentage) == yellow_percentage:
        color = "Yellow"
    else:
        color = "Green"

    return color

def detect_traffic_lights(image, net, classes, layer_names):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    traffic_light_boxes = []
    traffic_light_colors = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 9:  # Assuming class 9 corresponds to traffic light in your class names file
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                roi = image[y:y+h, x:x+w]

                color = analyze_traffic_light_color(roi)

                traffic_light_boxes.append((x, y, x+w, y+h))
                traffic_light_colors.append(color)

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return traffic_light_boxes, image, traffic_light_colors

def main():
    st.title("Traffic Light Detection App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        net, classes, layer_names = load_yolo()

        if net is not None:
            traffic_light_boxes, annotated_image, traffic_light_colors = detect_traffic_lights(image, net, classes, layer_names)

            st.image(annotated_image, channels="BGR", caption="Detected Traffic Lights")

            if traffic_light_colors:
                average_color = max(set(traffic_light_colors), key=traffic_light_colors.count)
                suggest_drive(average_color)
            else:
                st.write("No traffic lights detected. Continue driving.")

def suggest_drive(color):
    suggestions = {
        "Red": "Stop! Do not proceed.",
        "Green": "Continue driving.",
        "Yellow": "Go slow, prepare to stop."
    }

    st.write(f"Suggestion: {suggestions.get(color, 'Continue driving.')}")

if __name__ == "__main__":
    main()
