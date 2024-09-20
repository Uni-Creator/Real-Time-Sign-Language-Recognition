---

# **Sign Language Recognition with LSTM**

## **Description**

This project implements a Long Short-Term Memory (LSTM) neural network to classify sequences of sign language gestures. The system is trained on a dataset containing sequences of hand gesture frames for different actions in American Sign Language (ASL), such as 'nothing', 'hello', 'thanks', and 'I love you'. The model is designed to recognize these gestures from input sequences and classify them accurately into their respective categories.

### **Project Overview**

The project pipeline is as follows:

1. **Data Preparation**: The input data consists of a sequence of frames, with each frame represented as a numpy array of key points (features). These sequences are pre-processed and stored in a folder structure where each action category contains multiple gesture sequences.
   
2. **Model Architecture**: The model is built using PyTorch and consists of three LSTM layers followed by fully connected (dense) layers. The LSTM layers are designed to capture temporal dependencies in the gesture sequences. The final classification is performed using a softmax activation function to output probabilities for each gesture class.
   
3. **Training and Evaluation**: The model is trained using the Adam optimizer and cross-entropy loss. Training is performed with early stopping to prevent overfitting, and the best model is saved. The performance of the model is evaluated using a confusion matrix and class-wise probabilities on the test set.

4. **GPU Support**: The model is optimized to run on a GPU (if available) for faster computation. This is handled using PyTorch's `torch.device`.

5. **Visualization**: A confusion matrix is generated at the end of the training process to visualize the performance of the model, showing the predicted versus actual gesture categories.

---

## **Technologies Used**
- **Python**: For model implementation and data handling.
- **PyTorch**: Used for building and training the LSTM model.
- **Numpy**: For data manipulation and sequence loading.
- **Matplotlib & Seaborn**: For plotting and visualizing the confusion matrix.
- **Sklearn**: For train/test data splitting and evaluation metrics.

---

# **Project Files Structure**
```plaintext
.
├── Data/                           # Directory containing gesture sequences for each action
│   ├── hello/                      # Folder for "hello" action sequences
│   ├── iloveyou/                   # Folder for "I love you" action sequences
│   ├── nothing/                    # Folder for "nothing" action sequences
│   └── thanks/                     # Folder for "thanks" action sequences
├── best_lstm_model.h5              # Saved model with the best validation accuracy
├── main.py                         # Main script for  evaluating the model
├── trainer.py                      # Script for training the model
├── dataCollection.py               # Script for creating data
├── README.md                       # Project readme file
└── requirements.txt                # Python dependencies required for the project
```

---

# **Setup Instructions**

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Uni-Creator/Real-Time-Sign-Language-Recognition.git
   cd Real-Time-Sign-Language-Recognition
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**:
   - Download or collect sequences of sign language gestures.
   - Place them in the `Data/` directory with each action sequence in its respective folder.

### **Running the Model**

To train the model:

```bash
python main.py
```

### **GPU Acceleration**

If a CUDA-compatible GPU is available, the model will automatically use it for training and inference. Ensure you have installed the appropriate versions of PyTorch and CUDA to support GPU execution.

### **Evaluating the Model**

After training, the model will evaluate the performance on the test set. The confusion matrix and probabilities for each class will be printed and visualized.

---

# **Project Workflow**

1. **Data Loading**: Load gesture sequences for each action (e.g., hello, thanks, etc.).
2. **Model Training**: Train the LSTM model to classify these sequences into the appropriate actions.
3. **Evaluation**: Test the trained model and generate a confusion matrix to assess its performance.

---

# **Contributing**

Contributions are welcome! If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

---

# **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

# **Contact**

For any questions or inquiries about this project, please feel free to reach out at abhayr245654@gmail.com.

---
