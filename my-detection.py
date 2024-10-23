import jetson.inference
import jetson.utils

# Initialize the object detection network with SSD MobileNet v2
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load an image (change this path to your actual image path)
image_path = "/home/nvidia/jetson-inference/data/images/dog_0.jpg"
img = jetson.utils.loadImage(image_path)

# Perform object detection
detections = net.Detect(img)

# Print the detected objects' class ID, confidence, and bounding box coordinates
for detection in detections:
    print(f"ClassID: {detection.ClassID}")
    print(f"Confidence: {detection.Confidence:.5f}")
    print(f"Left: {detection.Left:.2f}, Top: {detection.Top:.2f}")
    print(f"Right: {detection.Right:.2f}, Bottom: {detection.Bottom:.2f}")
    print(f"Width: {detection.Width:.2f}, Height: {detection.Height:.2f}")
    print(f"Area: {detection.Area:.2f}")
    print(f"Center: ({detection.Center[0]:.2f}, {detection.Center[1]:.2f})")

# Save the annotated image with detected bounding boxes
output_path = "/home/nvidia/jetson-inference/data/images/dog_0_detected.jpg"
jetson.utils.saveImage(output_path, img)
print(f"Annotated image saved to {output_path}")
