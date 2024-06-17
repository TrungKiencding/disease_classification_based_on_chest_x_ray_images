import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pydicom
import numpy as np

pathology_list = ['Кардиомегалия', 'Эмфизема', 'Экссудат', 'Грыжа', 'Узелок', 'Пневмоторакс', 'Ателектаз',
                  'Утолщение плевры', 'Опухоль', 'Отёк', 'Консолидация', 'Инфильтрация', 'Фиброз', 'Пневмония']

# Đặt thiết bị (CPU hoặc GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Khởi tạo mô hình với kiến trúc đã lưu
num_classes = 14  # Số lớp phân loại của bạn
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load state_dict đã lưu
model.load_state_dict(torch.load('resnet18-model.pt',  map_location=device))
# Đặt mô hình vào chế độ đánh giá
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess dicom file


def load_dicom(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    img = Image.fromarray(img).convert('RGB')

    return img


# Streamlit interface
st.title("X-ray Image Classification")

uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm"])

if uploaded_file is not None:
    image = load_dicom(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    # Dự đoán phân loại ảnh
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)  # Áp dụng hàm sigmoid
        predicted = (probabilities > 0.62).int()

    if torch.sum(predicted) == 0:  # Kiểm tra nếu tất cả các giá trị dự đoán đều là 0
        decoded = ['No Findings']
    else:
        decoded = [pathology_list[i]
                   for i in range(len(predicted[0])) if predicted[0][i] == 1]
    st.subheader(
        "Выявить риск следующих типов заболеваний легких на основе рентгена:")

    for index, value in enumerate(decoded):
        st.write(f"{value} ")
