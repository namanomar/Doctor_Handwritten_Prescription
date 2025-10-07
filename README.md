# 🩺 Doctor’s Handwriting Prediction using Deep Learning and Computer Vision

## 📘 Overview
This project aims to **decode and predict handwritten text from doctors' prescriptions** using modern **deep learning techniques**.  
It addresses one of the toughest OCR challenges — **irregular, messy, and stylized handwriting** — by combining robust preprocessing, data augmentation, and neural sequence modeling.

---

## 🧠 Objective
Build an end-to-end **deep learning model** that transcribes doctors’ handwritten prescriptions into **legible digital text** for use in healthcare systems, pharmacies, or EHRs.

---

## 🧩 Features
- End-to-end handwriting recognition pipeline  
- Works with scanned or photographed prescription images  
- Preprocessing for denoising, normalization, and segmentation  
- CNN / CRNN / Transformer-based models with **CTC decoding**  
- Synthetic handwriting data generation  
- Evaluation via **Character Error Rate (CER)** and **Word Error Rate (WER)**  

---

## 🏗️ Methodology Flow

The complete pipeline (till *Evaluation*) is shown below:

```mermaid
flowchart LR
  %% --- STYLE DEFINITIONS ---
  classDef data fill:#d6eaf8,stroke:#1b4f72,stroke-width:2px,color:#000,font-weight:bold
  classDef process fill:#fdebd0,stroke:#b9770e,stroke-width:2px,color:#000,font-weight:bold
  classDef model fill:#d5f5e3,stroke:#1d8348,stroke-width:2px,color:#000,font-weight:bold
  classDef eval fill:#f5b7b1,stroke:#922b21,stroke-width:2px,color:#000,font-weight:bold
  classDef final fill:#dcdcdc,stroke:#424949,stroke-width:2px,color:#000,font-weight:bold

  A([Start]):::data --> B[Data Collection]:::data --> C[Annotation and Labeling]:::data
  C --> D[Preprocessing]:::process --> E[Augmentation and Split]:::process
  E --> F[Model Design]:::model --> G[Training Setup]:::model
  G --> H[Validation and Tuning]:::model --> I[Evaluation]:::eval --> J([End]):::final

  subgraph Data_Stage[🩵 Data Preparation]
    direction TB
    B1[Scan or Photograph Prescriptions]
    B2[Collect Typed Transcriptions]
    B3[Use Public Handwriting Datasets]
    B --> B1 & B2 & B3
    C2[Mark Word or Line Boxes]
    C3[Perform Quality Control]
    C --> C2 & C3
  end

  subgraph Preprocess_Stage[🟠 Preprocessing and Augmentation]
    direction TB
    D1[Deskew, Crop, Resize]
    D2[Convert to Grayscale or Binarize]
    D3[Normalize Intensity and Contrast]
    D --> D1 & D2 & D3
    E1[Apply Rotation, Noise, Elastic Distortion]
    E2[Generate Synthetic Samples]
    E3[Split into Train, Validation, Test]
    E --> E1 & E2 & E3
  end

  subgraph Model_Stage[🟢 Model Training and Validation]
    direction TB
    F1[Select CNN, CRNN, or Transformer Architecture]
    F2[Add CTC or Seq2Seq Decoder]
    F --> F1 & F2
    G1[Define Loss Function and Optimizer]
    G2[Use GPU and Mixed Precision]
    G --> G1 & G2
    H1[Track CER, WER, and Accuracy]
    H2[Use Early Stopping and Checkpointing]
    H --> H1 & H2
  end

  subgraph Eval_Stage[🔴 Evaluation]
    direction TB
    I1[Analyze Word-Level and Character Errors]
    I2[Visualize Confusion Matrix]
    I3[Collect Human Feedback]
    I --> I1 & I2 & I3
  end
