# Foot and Mouth Virus Detection in Cattle using YOLOv7 🐄🦠  

This project implements an **AI-based detection system** for identifying **Foot-and-Mouth Disease Virus (FMDV) in cattle** using the **YOLOv7** object detection model. The system is trained on a custom dataset using **Roboflow**, and experiment tracking is done with **Weights & Biases (W&B)**.

---

## 🚀 Features
- **Real-time detection** of infected cattle using YOLOv7.
- **Dataset preprocessing and augmentation** using Roboflow.
- **Experiment tracking** with Weights & Biases.
- **Deployable on Google Colab or Local Machine**.

---

## 📂 Project Structure
```
FMD_Virus_Detection_YOLOv7/
│── yolov7/                    # Cloned YOLOv7 repository
│── runs/detect/exp/           # Results folder
│── best.pt                    # Trained YOLOv7 model weights
│── detect.py                   # Detection script
│── requirements.txt            # Dependencies for the project
│── README.md                   # Project documentation
│── data.yaml                   # Dataset configuration file
│── sample_images/              # Example images for testing
```

---

## 🛠️ Setup Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/FMD_Virus_Detection_YOLOv7.git
cd FMD_Virus_Detection_YOLOv7
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Download YOLOv7 and Pretrained Weights**
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### **4️⃣ Download the Dataset**
This dataset was managed via **Roboflow**. To download it manually:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov7")
```

---

## 🖼️ Running Object Detection

### **Detect Objects in an Image**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source /path/to/image.jpg
```

### **Detect Objects in a Folder**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source /path/to/folder/
```

### **Real-time Detection Using Webcam**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source 0
```

---

## 📊 Performance Evaluation
- **mAP (Mean Average Precision):** 86%
- **Precision:** 93%
- **Recall:** 87%

---

## 🛠️ Tools & Technologies Used
- **Python**
- **YOLOv7**
- **Roboflow** (Dataset management & augmentation)
- **Weights & Biases** (Experiment tracking)
- **OpenCV** (Image processing)
- **Google Colab** (Training & inference)


---

## 🌟 Acknowledgments
- **YOLOv7** by WongKinYiu
- **Roboflow** for dataset processing
- **Weights & Biases** for model tracking


