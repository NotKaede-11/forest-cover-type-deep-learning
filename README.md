# Forest Cover Type Classification — Deep Learning Finals Project

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-green.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-88.75%25-brightgreen.svg)]()
[![Macro--F1](https://img.shields.io/badge/Test%20Macro--F1-0.8315-blueviolet.svg)]()

> **Finals Project for Deep Learning | Group 8**  
> An industrial-scale multiclass neural network study classifying 581,012 forest land plots in Roosevelt National Forest, Northern Colorado, using cartographic map features.

---

## 🌲 Executive Summary

In environmental management and forestry, mapping tree distribution across hundreds of thousands of acres of mountainous wilderness is critical for wildfire planning, habitat conservation, and timber management. However, surveying over 800,000 acres by foot would take decades and cost millions.

This project trains and tunes a **feedforward Dense Neural Network (MLP)** in TensorFlow/Keras on the **UCI Forest Cover Type dataset (581,012 records, 54 features)** to classify plots into **7 major forest cover types** based exclusively on non-invasive map clues (elevation, slope, aspect, distance to hydrology/roads/fire points, and soil/wilderness indicators).

### Key Study Findings
* **Winning Model (R1 — Wider Network)**: 54 inputs $\rightarrow$ Two hidden layers of 100 neurons each (ReLU) $\rightarrow$ 7 Softmax outputs (16,307 trainable parameters).
* **Final Held-Out Test Accuracy**: **88.75%** on 58,102 unseen records.
* **Final Test Macro-F1**: **0.8315** (averaging all 7 classes equally to combat severe class imbalance).
* **Zero Overfitting**: Training accuracy (**89.29%**) vs Validation accuracy (**88.66%**) represents an ultra-narrow gap of just **0.64 percentage points**.
* **High Catch Rates**: Lodgepole Pine (**92.77% recall**), Krummholz (**91.57%**), and Cottonwood/Willow (**91.24%**).
* **Primary Challenge**: Douglas-fir (**64.42% recall**) due to ecological feature overlap with Ponderosa Pine in lower canyon transition zones.

---

## 👥 Group 8 Team Members & Roles

| Member Name | Project Role | Primary Ownership |
| :--- | :--- | :--- |
| **Beomarc Andrew D. Cartoneros** | **Data Lead** | Data loading, exploratory data analysis, anti-leakage scaling, and stratified splitting. |
| **Matthew R. Lopez** | **Model Lead** | Baseline neural network architecture, Keras compilation, and model definitions. |
| **Axiel Achia M. Escabal** | **Experiment Lead** | Planning and logging the 9 controlled tuning runs, loss curves, and early stopping. |
| **Alon U. Villanueva** | **Report Lead** | Academic case study writing, slide deck synthesis, error analysis, and client recommendations. |

---

## 📁 Repository Structure & Deliverables

```text
├── pdf/
│   ├── Group8_Academic_Report.pdf       # 5-page formal academic case study (APA 7th edition, figures & tables)
│   └── Group8_Report.pdf                # Academic report distribution copy
│
├── pptx/
│   └── Group8_Case_Study.pptx           # 12-slide editable presentation deck with full speaker notes & photos
│
├── docx/
│   ├── Group8_Plain_English_Executive_Guide.docx       # Non-technical project guide for managers & judges (no buzzwords)
│   ├── Group8_PowerPoint_Presentation_Script_and_Guide.docx # Slide-by-slide word-for-word spoken presentation script
│   ├── Group8_Rubric_Section_6_Presentation_and_QA_Guide.docx # 6 rubric pillars (10 min) + live code Q&A defense (5 min)
│   ├── Group8_Academic_Report_Plain_English_Guide.docx # Companion guide decoding every table & chart in the academic report
│   ├── Group8_Jupyter_Notebook_Code_Walkthrough.docx   # Cell-by-cell code walkthrough with 12 defense Q&As
│   ├── Group8_Academic_Report.docx                     # Editable Word copy of the 5-page academic case study
│   ├── Group8_PPT_Content_Pack.docx                    # Detailed slide content planning pack
│   └── Group8_Presentation_Review_and_QA.docx          # Rehearsal review guide with 41 answered technical questions
│
├── verification/
│   └── Group8_Reproduction.ipynb        # Clean, self-contained reproduction Jupyter notebook
│
├── Group8_Project_Package.zip           # Portable archive with complete datasets, code, models, and evidence
└── README.md                            # Repository documentation
```

---

## 🔬 Experimental Methodology & Results

We tested nine controlled configurations, modifying exactly **one factor at a time** against our baseline model (R0):

| Run | Configuration Tested | Epochs Run / Restored | Val. Loss | Val. Accuracy | Val. Macro-F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **R0** | Baseline (2 $\times$ 50 ReLU, lr=0.001, batch 100) | 16 / 14 | 0.3417 | 85.75% | 0.7827 | Benchmark |
| **R1** | **Wider Layers (2 $\times$ 100 ReLU)** | **13 / 11** | **0.2793** | **88.66%** | **0.8272** | **★ WINNER** |
| **R2** | Deeper Network (3 $\times$ 50 ReLU) | 20 / 18 | 0.2889 | 88.38% | 0.8163 | Evaluated |
| **R3** | Tanh Activation Function | 34 / 32 | 0.2682 | 89.31% | 0.8254 | High acc, lower F1 |
| **R4** | Batch Size 500 | 22 / 20 | 0.3611 | 85.26% | 0.7628 | Suboptimal minimum |
| **R5** | Learning Rate 0.0003 | 42 / 40 | 0.3433 | 85.87% | 0.7753 | Slow convergence |
| **R6** | Patience 5 (Extended Early Stopping) | 50 / 46 | 0.2901 | 88.21% | 0.8270 | Hit epoch cap |
| **R7** | Balanced Class Weights | 15 / 13 | 0.4965 | 79.59% | 0.7353 | Boosted minority recall |
| **R8** | Dropout (0.2 Regularization) | 24 / 22 | 0.4185 | 82.51% | 0.7114 | Unneeded regularization |

### Deterministic Model Selection
* **Pre-declared decision rule**: Highest Validation Macro-F1 $\rightarrow$ Validation Accuracy $\rightarrow$ Lower Validation Loss $\rightarrow$ Fewer Parameters.
* **Why R1 won over R3**: Model R3 achieved higher raw accuracy (89.31%), but did so by over-predicting the majority pine classes at the expense of rare species. Model R1 achieved a higher Macro-F1 (0.8272), ensuring balanced performance across all 7 forest cover types.

---

## 📊 Final Held-Out Performance (Test Set: 58,102 Rows)

| True Cover Type | Test Support | Precision | Recall (Class Acc.) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Spruce / Fir** | 21,184 | 90.78% | 85.99% | 0.8832 |
| **Lodgepole Pine** | 28,331 | 88.44% | 92.77% | 0.9056 |
| **Ponderosa Pine** | 3,576 | 85.89% | 88.84% | 0.8734 |
| **Cottonwood / Willow** | 274 | 73.10% | 91.24% | 0.8117 |
| **Aspen** | 949 | 73.27% | 67.86% | 0.7046 |
| **Douglas-fir** | 1,737 | 84.58% | 64.42% | 0.7314 |
| **Krummholz** | 2,051 | 90.59% | 91.57% | 0.9108 |
| **Overall / Macro Average** | **58,102** | **83.81%** | **83.24%** | **0.8315** |

---

## 🎯 Practical Business & Environmental Recommendations

1. **Automated Screening & Triage**: Deploy Model R1 to automatically classify the ~85% of forest acreage dominated by standard Lodgepole Pine and Spruce/Fir stands, saving significant field-surveying budget and ranger time.
2. **Human-in-the-Loop Verification**: Require manual ranger inspections or aerial drone surveys whenever Douglas-fir is predicted, or whenever the maximum softmax output probability drops below 60%.
3. **Geographic Boundaries**: This model was trained strictly on Colorado Rocky Mountain temperate conifers and alpine landtypes. It **cannot** be transferred to tropical rainforests (such as in the Philippines) or lowland ecosystems without retraining on local cartographic and botanical data.

---

## 📚 References & Data Citations

* **Blackard, J. A.** (1998). *Covertype* [Data set]. UCI Machine Learning Repository. [https://doi.org/10.24432/C50K5N](https://doi.org/10.24432/C50K5N)
* **Blackard, J. A.** (1999). *The Forest CoverType dataset* [Data set documentation]. Department of Forest Sciences, Colorado State University.
* **Keras & TensorFlow**: Chollet, F. et al. (2015). *Keras: Deep learning for humans*. [https://keras.io/](https://keras.io/)
* **Scikit-learn**: Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python*. JMLR, 12, 2825-2830.
