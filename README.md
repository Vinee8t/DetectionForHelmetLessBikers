****Helmetless Biker Detection System****


*Overview:-*

The Helmetless Biker Detection System is a computer vision-based solution designed to automatically detect bikers who are riding without helmets. The system processes video feeds in real-time using the YOLOv8 object detection model and a CSRT tracker for reliable object tracking. It captures high-quality images of helmetless riders, providing a scalable solution for traffic enforcement and road safety.

*Features:-*

-Real-Time Detection: Utilizes the YOLOv8 model for accurate and real-time detection of bikers and helmets.

-Image Capture: Captures high-resolution images of violations for further analysis.

-Adaptability: Works in diverse environments such as urban roads, highways, and rural areas.



*Installation:-*

Prerequisites-

Ensure you have the following installed on your system:

-Python 3.8+
-OpenCV
-Pandas
-NumPy
-Ultralytics YOLO

Model Setup-

-Download the pre-trained YOLOv8 model weights (best1.pt) and place it in the root directory.

-Prepare a coco1.txt file containing the class names for detection (e.g., bike, nohelmet).

Dataset-

Source: A custom dataset containing images of bikers with and without helmets.



*Characteristics:-*

-Size: Contains diverse images captured in different environments.

-Diversity: Includes urban, highway, and rural scenes.

-Balance: Maintains a balanced distribution of helmet and no-helmet cases.



*How It Works:-*

Video Processing:

-The system processes video frames using OpenCV.

-Each frame is resized for consistency.

Detection:-

-The YOLOv8 model identifies objects in the frame and checks for the classes bike and nohelmet.

Image Capture:-

-If both bike and nohelmet are detected, the system captures an enlarged image of the violation.

![image](https://github.com/user-attachments/assets/a7e13a70-248e-446d-8818-84c4f3623204)


Results:-

![image](https://github.com/user-attachments/assets/45fbbfc0-3a9c-4f96-90cd-9d462d1137ba)

![image](https://github.com/user-attachments/assets/a059a779-9e2c-4a4b-ab1b-2c0f623fe215)


-Accuracy: High detection accuracy with minimal false positives and false negatives.

-Reliability: Robust tracking ensures consistent performance under varying conditions.

-Scalability: Can be deployed for live traffic monitoring.


*Challenges and Solutions:-*

Dataset Preparation:

-Challenge: Acquiring a diverse and balanced dataset.

-Solution: Curated images from different environments and ensured proper labeling.

Real-Time Processing:

-Challenge: Balancing accuracy with speed.

-Solution: Processed every third frame and optimized the model for faster inference.
