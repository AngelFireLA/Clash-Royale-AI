import json

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results

model = YOLO(r'C:\Dev\Python\clash-royale-ai\epoch50.pt')
#model = YOLO(r'C:\Dev\Python\PylaWallDetector\current.pt')

image_path = r'C:\Dev\Python\clash-royale-ai\images\recording\2025-04-15 21-19-59\7.png'
image = cv2.imread(image_path)
results = model(image, conf=0.4)

result: Results = results[0]
print(json.loads(result.to_json()))
annotated_frame = result.plot()
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
