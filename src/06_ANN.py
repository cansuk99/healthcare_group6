import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score




# ============================================================================
# 1. LOAD FINAL FEATURE SET
# ============================================================================


df = pd.read_pickle('data/selected-features/feature_all_not_onehot.pkl')

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

print(f"🔹 Numeric columns ({len(numeric_cols)}):")
print(numeric_cols)
print("\n🔸 Categorical columns ({len(categorical_cols)}):")
print(categorical_cols)


# ============================================================================
# 2. DROP COLS
# ============================================================================

cols_to_drop = [
    'encounter_id', 'patient_nbr',
    'admission_type_desc', 'discharge_disposition_desc', 'admission_source_desc',
    'readmitted',
]

df = df.drop(columns=cols_to_drop, errors='ignore')

# Squared terms
# Neural nets with hidden layers and nonlinear activations (like ReLU, tanh) can already learn nonlinear curves by themselves.
# You don’t need to manually inject squared terms — the network can approximate that relationship.
# They’re perfectly correlated with their base columns, so they add multicollinearity.
#  That doesn’t “break” a neural net, but it can slow learning a bit because the optimizer sees redundant information.

# 'diag_1', 'diag_2', 'diag_3' -- add too much noice we need to drop them from the other models as well


noise = ['diag_1', 'diag_2', 'diag_3',
    'time_in_hospital_squared', 'num_medications_squared', 'num_lab_procedures_squared'
]

df = df.drop(columns=noise, errors='ignore')

df.shape[1]

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print(numeric_cols)

categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
print(categorical_cols)


# ============================================================================
# 2. Model
# ============================================================================

#Separate target variable
y = df['readmitted_binary'].astype(int)
X = df.drop(columns=['readmitted_binary'])

#column types
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

numeric_cols
categorical_cols


# One-Hot Encode categorical columns

if categorical_cols:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = encoder.fit_transform(X[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    X_cat = pd.DataFrame(X_cat, columns=cat_feature_names, index=X.index)
else:
    X_cat = pd.DataFrame(index=X.index)

# Scale numeric columns

scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X[numeric_cols]),
                     columns=numeric_cols, index=X.index)


#Combine encoded categorical and scaled numeric

X_final = pd.concat([X_num, X_cat], axis=1)
print(f"shape: {X_final.shape}")

#Convert to NumPy arrays for ANN

X_np = X_final.to_numpy().astype(np.float32)
y_np = y.to_numpy().reshape(-1, 1).astype(np.float32)

#Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")



# ==========================================================
#  (Weighted Binary Cross-Entropy) - (NumPy-Only)
# ==========================================================

# ----------  Thread Optimization (must come first) ----------
import os, multiprocessing

n_threads = max(1, multiprocessing.cpu_count() - 1)
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[var] = str(n_threads)

print(f"Using {n_threads} threads for NumPy operations...")


# # ---------- 1️ Hyperparameters ----------

# input_dim  = X_train.shape[1]   # Number of input features — automatically set from your dataset
# hidden_dim = 32                 # Number of neurons in the hidden layer
#                                 # Tuning tip:
#                                 #   - Start small (16–64 neurons)
#                                 #   - Increase if underfitting (loss remains high, low recall)
#                                 #   - Decrease if overfitting (training loss drops but test loss worsens)

# output_dim = 1                  # One output neuron for binary classification (0 = not readmitted, 1 = readmitted)

# lr         = 0.001              # Learning rate: controls how big each weight update is
#                                 # Tuning tip:
#                                 #   - Too high → training oscillates or diverges
#                                 #   - Too low → very slow convergence
#                                 #   - Try 0.1, 0.01, 0.001 and monitor the loss curve

# epochs     = 1000               # Number of full passes over the training data
#                                 # Tuning tip:
#                                 #   - If loss stops improving early, reduce epochs or use early stopping
#                                 #   - If loss still decreasing at the end, increase epochs slightly



# # ---------- 2️ Class Weights ----------
# # Count how many samples belong to each class in the training set
# pos_count = (y_train == 1).sum()   # Number of readmitted patients (minority class)
# neg_count = (y_train == 0).sum()   # Number of not-readmitted patients (majority class)
# total     = len(y_train)           # Total number of samples in the training set

# pos_count
# neg_count

# # Compute class weights inversely proportional to class frequency
# #  This makes the rare class (readmitted) more "expensive" to misclassify.
# weight_pos = total / pos_count     # Weight for positive class (readmitted)
# weight_neg = total / neg_count     # Weight for negative class (not readmitted)

# # normalize the scale
# scale = (weight_pos + weight_neg) / 2
# weight_pos /= scale
# weight_neg /= scale



# print(f"weights → neg:{float(weight_neg):.2f}, pos:{float(weight_pos):.2f}")

# # ---------- 3️ Initialize Parameters ----------
# # Create a random number generator with a fixed seed for reproducibility
# rng = np.random.default_rng(42)

# # Initialize weights for the first (input → hidden) layer
# # Shape: (number of input features, number of hidden neurons)
# # Each weight is drawn from a normal distribution with mean=0 and std=0.1

# # W1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
# W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim))

# # Initialize biases for the hidden layer — start at zero (neutral baseline)
# b1 = np.zeros((1, hidden_dim))

# # Initialize weights for the second (hidden → output) layer
# # Shape: (number of hidden neurons, number of output neurons)
# # Again, small random values help break symmetry so neurons learn differently

# #W2 = rng.normal(0, 0.1, (hidden_dim, output_dim))
# W2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, output_dim))
# # Initialize bias for the output neuron — also zeros

# b2 = np.zeros((1, output_dim))

# # ---------- 4️ Activation Functions ----------

# # ReLU (Rectified Linear Unit)
# def relu(x):
#     # Returns x for positive inputs, and 0 for negative inputs
#     # Formula: ReLU(x) = max(0, x)
#     # Used in the hidden layer to introduce non-linearity
#     return np.maximum(0, x)

# # Derivative of ReLU
# def relu_deriv(x):
#     # Gradient is 1 for x > 0 and 0 for x <= 0
#     # Used during backpropagation to compute gradients efficiently
#     return (x > 0).astype(float)

# # Sigmoid
# def sigmoid(x):
#     # Squashes input to the range (0, 1)
#     # Formula: sigmoid(x) = 1 / (1 + e^-x)
#     # Used in the output layer for binary classification — interpretable as probability
#     x = np.clip(x, -50, 50)   # prevent overflow
#     return 1 / (1 + np.exp(-x))


# # ---------- 5️ Weighted Binary Cross-Entropy ----------
# def weighted_bce(y_true, y_pred):
#     # Ensure y_true is a column vector (shape: [n_samples, 1])
#     y_true = y_true.reshape(-1, 1)

#     # Compute the weighted binary cross-entropy loss
#     # Formula (element-wise):
#     # L = - [ w_pos * y_true * log(y_pred) + w_neg * (1 - y_true) * log(1 - y_pred) ]
#     #
#     # where:
#     #   - y_true ∈ {0, 1} → actual label (0 = not readmitted, 1 = readmitted)
#     #   - y_pred ∈ (0, 1) → predicted probability from sigmoid
#     #   - w_pos, w_neg → weights that rebalance the importance of each class
#     #
#     # The small constant 1e-9 prevents log(0), which would cause NaN errors.
#     return -(weight_pos * y_true * np.log(y_pred + 1e-9) +
#               weight_neg * (1 - y_true) * np.log(1 - y_pred + 1e-9)).mean()

# #balanced dataset, just use:
# #   loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


# # ---------- 6️ Training Loop ----------
# import time

# start = time.perf_counter()


# losses = []             # Track all loss values for plotting later
# best_loss = np.inf      # Initialize the "best" loss to infinity (so first one is always better)
# no_improve = 0          # Counter for how many epochs had no improvement
# patience = 300          # Stop after 300 epochs without improvement


# for epoch in range(epochs):

#     # ----------- FORWARD PROPAGATION -----------
#     # Step 1: Linear combination at hidden layer
#     #   z1 = X_train * W1 + b1
#     z1 = X_train @ W1 + b1

#     # Step 2: Apply activation (ReLU)
#     #   a1 = ReLU(z1)
#     #   Introduces non-linearity so the model can learn complex patterns
#     a1 = relu(z1)

#     # Step 3: Output layer (linear → sigmoid)
#     #   z2 = a1 * W2 + b2
#     #   y_pred = sigmoid(z2)
#     #   Converts raw outputs into probabilities (0–1)
#     z2 = a1 @ W2 + b2
#     y_pred = sigmoid(z2)

#     # ----------- COMPUTE LOSS -----------
#     # Compare predictions (y_pred) with actual labels (y_train)
#     # Using weighted BCE to correct for class imbalance
#     loss = weighted_bce(y_train, y_pred)

#     # ----------- BACKPROPAGATION -----------
#     # Goal: compute how much each weight contributed to the loss
#     # and adjust it in the opposite direction of the gradient.

#     # Step 1: Derivative of loss w.r.t. output (chain rule)
#     # (difference between predicted and true probability, scaled by weights)
#     dL_dy = (
#         (weight_pos * y_train / (y_pred + 1e-9))
#         - (weight_neg * (1 - y_train) / (1 - y_pred + 1e-9))
#     ) * (y_pred * (1 - y_pred))  # derivative of sigmoid

#     # Step 2: Gradients for output layer (W2, b2)
#     dL_dW2 = a1.T @ dL_dy / len(y_train)              # gradient wrt W2
#     dL_db2 = dL_dy.mean(axis=0, keepdims=True)        # gradient wrt b2

#     # Step 3: Propagate error back to hidden layer
#     dL_da1 = dL_dy @ W2.T                             # gradient wrt activation
#     dL_dz1 = dL_da1 * relu_deriv(z1)                  # gradient wrt pre-activation (ReLU')

#     # Step 4: Gradients for hidden layer (W1, b1)
#     dL_dW1 = X_train.T @ dL_dz1 / len(y_train)
#     dL_db1 = dL_dz1.mean(axis=0, keepdims=True)

#     # ----------- WEIGHT UPDATES -----------
#     # Gradient descent step: move weights in the direction that reduces loss
#     W1 -= lr * dL_dW1
#     b1 -= lr * dL_db1
#     W2 -= lr * dL_dW2
#     b2 -= lr * dL_db2

#     # ---------- Diagnostics ----------
#     if epoch % 200 == 0:
#         grad_norm_W1 = np.linalg.norm(dL_dW1)
#         grad_norm_W2 = np.linalg.norm(dL_dW2)
#         weight_norm_W1 = np.linalg.norm(W1)
#         weight_norm_W2 = np.linalg.norm(W2)
#         print(f"Epoch {epoch:4d} | Loss={loss:.4f} | "
#               f"grad_W1={grad_norm_W1:.4f} | grad_W2={grad_norm_W2:.4f} | "
#               f"|W1|={weight_norm_W1:.4f} | |W2|={weight_norm_W2:.4f}")

#     # ---------- Early Stopping ----------
#     if loss < best_loss - 1e-5:
#         best_loss = loss
#         no_improve = 0
#     else:
#         no_improve += 1
#     if no_improve >= patience:
#         print(f"Early stopping at epoch {epoch} — no improvement for {patience} steps.")
#         break

# print(f"It took {time.perf_counter() - start:.2f} seconds")

# # ---------- 7️ Evaluation ----------
# def predict(X):
#     return (sigmoid(relu(X @ W1 + b1) @ W2 + b2) > 0.5).astype(int)

# y_hat = predict(X_test)
# accuracy = (y_hat == y_test).mean()
# precision = ( (y_hat[y_test==1] == 1).sum() / max((y_hat==1).sum(),1) )
# recall = ( (y_hat[y_test==1] == 1).sum() / max((y_test==1).sum(),1) )
# f1 = 2 * precision * recall / max((precision + recall),1e-9)

# print(f"\n Accuracy: {accuracy:.3f}")
# print(f"Precision: {precision:.3f}")
# print(f" Recall: {recall:.3f}")
# print(f"F1: {f1:.3f}")

# ------------------Results ----------------
# Accuracy: 0.869
# Precision: 0.113
# Recall: 0.022
# F1: 0.037


# =====================================================
#  OVERSAMPLING + DIAGNOSTICS
# =====================================================


# # ---------- 1 OVERSAMPLING ----------
# pos_idx = np.where(y_train == 1)[0]
# neg_idx = np.where(y_train == 0)[0]
# oversample_ratio = int(len(neg_idx) / len(pos_idx)) // 2  # moderate oversampling

# oversampled_pos = np.tile(pos_idx, oversample_ratio)
# new_idx = np.concatenate([neg_idx, oversampled_pos])
# np.random.shuffle(new_idx)

# X_train_over = X_train[new_idx]
# y_train_over = y_train[new_idx]

# print(f"Original samples: {len(y_train)}")
# print(f"Oversampled samples: {len(y_train_over)}")
# print(f"Class 1 proportion: {y_train_over.mean():.2%}")

# # ---------- 2️ Initialize Parameters ----------
# rng = np.random.default_rng(42)
# W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim))
# b1 = np.zeros((1, hidden_dim))
# W2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, output_dim))
# b2 = np.zeros((1, output_dim))

# # ---------- 3️ Activation Functions ----------
# def relu(x): return np.maximum(0, x)
# def relu_deriv(x): return (x > 0).astype(float)
# def sigmoid(x):
#     x = np.clip(x, -50, 50)  # prevent overflow
#     return 1 / (1 + np.exp(-x))

# # ---------- 4️ Unweighted BCE ----------
# def bce(y_true, y_pred):
#     y_true = y_true.reshape(-1, 1)
#     return -(y_true * np.log(y_pred + 1e-9) +
#              (1 - y_true) * np.log(1 - y_pred + 1e-9)).mean()

# # ---------- 5️ Hyperparameters ----------
# lr = 0.001
# epochs = 2000
# patience = 300
# losses = []
# best_loss = np.inf
# no_improve = 0

# # ---------- 6️ Training Loop with Diagnostics ----------
# for epoch in range(epochs):
#     # Forward
#     z1 = X_train_over @ W1 + b1
#     a1 = relu(z1)
#     z2 = a1 @ W2 + b2
#     y_pred = sigmoid(z2)
#     loss = bce(y_train_over, y_pred)
#     losses.append(loss)

#     # Backprop
#     dL_dy = (y_pred - y_train_over)
#     dL_dW2 = a1.T @ dL_dy / len(y_train_over)
#     dL_db2 = dL_dy.mean(axis=0, keepdims=True)
#     dL_da1 = dL_dy @ W2.T
#     dL_dz1 = dL_da1 * relu_deriv(z1)
#     dL_dW1 = X_train_over.T @ dL_dz1 / len(y_train_over)
#     dL_db1 = dL_dz1.mean(axis=0, keepdims=True)

#     # Update
#     W1 -= lr * dL_dW1
#     b1 -= lr * dL_db1
#     W2 -= lr * dL_dW2
#     b2 -= lr * dL_db2

#     # Diagnostics every 200 epochs
#     if epoch % 200 == 0:
#         grad_norm_W1 = np.linalg.norm(dL_dW1)
#         grad_norm_W2 = np.linalg.norm(dL_dW2)
#         weight_norm_W1 = np.linalg.norm(W1)
#         weight_norm_W2 = np.linalg.norm(W2)
#         print(f"Epoch {epoch:4d} | Loss={loss:.4f} | "
#               f"grad_W1={grad_norm_W1:.4f} | grad_W2={grad_norm_W2:.4f} | "
#               f"|W1|={weight_norm_W1:.4f} | |W2|={weight_norm_W2:.4f}")

#     # Early stopping
#     if loss < best_loss - 1e-5:
#         best_loss = loss
#         no_improve = 0
#     else:
#         no_improve += 1
#     if no_improve >= patience:
#         print(f" Early stopping at epoch {epoch} — no improvement for {patience} steps.")
#         break

# # ---------- 7️ Plot Loss ----------
# plt.figure(figsize=(7,4))
# plt.plot(losses, label="Training loss")
# plt.title("Training Loss Curve (Oversampling)")
# plt.xlabel("Epoch")
# plt.ylabel("BCE Loss")
# plt.grid(True)
# plt.legend()
# plt.show()

# # ---------- 8️ Evaluation ----------
# def predict(X, threshold=0.5):
#     return (sigmoid(relu(X @ W1 + b1) @ W2 + b2) > threshold).astype(int)

# y_hat = predict(X_test)

# accuracy = (y_hat == y_test).mean()
# precision = ((y_hat[y_test==1] == 1).sum() / max((y_hat==1).sum(),1))
# recall = ((y_hat[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
# f1 = 2 * precision * recall / max((precision + recall),1e-9)

# print(f" Accuracy : {accuracy:.3f}")
# print(f" Precision: {precision:.3f}")
# print(f" Recall   : {recall:.3f}")
# print(f" F1-score : {f1:.3f}")

# # ================================================
# #PRECISION–RECALL CURVE & F1 vs THRESHOLD
# # =====================================================

# from sklearn.metrics import precision_recall_curve

# # 1️⃣ Get continuous probabilities (not thresholded)
# y_prob = sigmoid(relu(relu(X_test @ W1 + b1) @ W2 + b2) @ W3 + b3)

# # 2️⃣ Compute precision, recall, thresholds
# precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# # 3️⃣ Compute F1 for each threshold
# f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

# # 4️⃣ Find best threshold (max F1)
# best_idx = np.argmax(f1_scores)
# best_thresh = thresholds[best_idx]
# best_f1 = f1_scores[best_idx]

# print(f"\n✨ Optimal threshold = {best_thresh:.3f}")
# print(f"✨ Best F1 = {best_f1:.3f} | Precision = {precisions[best_idx]:.3f} | Recall = {recalls[best_idx]:.3f}")

# # 5️⃣ Plot Precision–Recall curve
# plt.figure(figsize=(7,5))
# plt.plot(recalls, precisions, color="royalblue", lw=2)
# plt.title("Precision–Recall Curve (Undersampled ANN)")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.grid(True)
# plt.show()

# # 6️⃣ Plot F1 vs Threshold
# plt.figure(figsize=(7,5))
# plt.plot(thresholds, f1_scores[:-1], color="darkorange", lw=2)
# plt.axvline(best_thresh, color="red", ls="--", label=f"Best threshold = {best_thresh:.3f}")
# plt.title("F1-Score vs Threshold (Undersampled ANN)")
# plt.xlabel("Decision Threshold")
# plt.ylabel("F1 Score")
# plt.legend()
# plt.grid(True)
# plt.show()

# # ------------------- Evaluation with a treshhold of 0.33----------------------------

# opt_thresh = 0.33  # or use `best_thresh` variable if still in memory
# y_hat_opt = (y_prob > opt_thresh).astype(int)

# accuracy_opt = (y_hat_opt == y_test).mean()
# precision_opt = ((y_hat_opt[y_test==1] == 1).sum() / max((y_hat_opt==1).sum(),1))
# recall_opt = ((y_hat_opt[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
# f1_opt = 2 * precision_opt * recall_opt / max((precision_opt + recall_opt),1e-9)

# print(f"\n Re-evaluation at optimal threshold = {opt_thresh:.3f}")
# print(f"Accuracy : {accuracy_opt:.3f}")
# print(f" Precision: {precision_opt:.3f}")
# print(f" Recall   : {recall_opt:.3f}")
# print(f" F1-score : {f1_opt:.3f}")



# # ---------- 9️⃣ Confusion Matrix ----------
# cm = confusion_matrix(y_test, y_hat)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"])
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix (Oversampled ANN)")
# plt.show()





# =============================================================
# # (OVERSAMPLING + CLIPPING + LR DECAY + 2 HIDDEN LAYERS)
# # =============================================================

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # ---------- 1️⃣ OVERSAMPLING ----------
# pos_idx = np.where(y_train == 1)[0]
# neg_idx = np.where(y_train == 0)[0]
# oversample_ratio = int(len(neg_idx) / len(pos_idx)) // 2
# oversampled_pos = np.tile(pos_idx, oversample_ratio)
# new_idx = np.concatenate([neg_idx, oversampled_pos])
# np.random.shuffle(new_idx)

# X_train_over = X_train[new_idx]
# y_train_over = y_train[new_idx]

# print(f"Oversampled dataset: {len(y_train_over)} samples | Class 1 % = {y_train_over.mean():.2%}")

# # ---------- 2️⃣ Initialization (He for ReLU) ----------
# rng = np.random.default_rng(42)
# hidden1, hidden2, output_dim = 64, 32, 1
# input_dim = X_train_over.shape[1]

# W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden1))
# b1 = np.zeros((1, hidden1))
# W2 = rng.normal(0, np.sqrt(2 / hidden1), (hidden1, hidden2))
# b2 = np.zeros((1, hidden2))
# W3 = rng.normal(0, np.sqrt(2 / hidden2), (hidden2, output_dim))
# b3 = np.zeros((1, output_dim))

# # ---------- 3️⃣ Activations ----------
# def relu(x): return np.maximum(0, x)
# def relu_deriv(x): return (x > 0).astype(float)
# def sigmoid(x):
#     x = np.clip(x, -50, 50)
#     return 1 / (1 + np.exp(-x))

# # ---------- 4️⃣ Binary Cross-Entropy ----------
# def bce(y_true, y_pred):
#     y_true = y_true.reshape(-1, 1)
#     return -(y_true * np.log(y_pred + 1e-9) +
#              (1 - y_true) * np.log(1 - y_pred + 1e-9)).mean()

# # ---------- 5️⃣ Hyperparameters ----------
# # lr = 0.001
# # decay_rate = 0.995      # multiply lr by this each epoch
# # clip_value = 1.0        # max L2-norm for gradient clipping
# # epochs = 2000
# # patience = 300
# # losses = []
# # best_loss, no_improve = np.inf, 0

# lr = 0.001
# decay_rate = 0.999      # slower decay — keeps learning active longer
# clip_value = 2.0        # allows stronger gradient updates
# epochs = 2500           # more time to converge
# patience = 500          # waits longer before stopping early

# # ---------- 6️⃣ Training Loop ----------
# for epoch in range(epochs):
#     # ---- Forward ----
#     z1 = X_train_over @ W1 + b1
#     a1 = relu(z1)
#     z2 = a1 @ W2 + b2
#     a2 = relu(z2)
#     z3 = a2 @ W3 + b3
#     y_pred = sigmoid(z3)

#     # ---- Loss ----
#     loss = bce(y_train_over, y_pred)
#     losses.append(loss)

#     # ---- Backprop ----
#     dL_dy = (y_pred - y_train_over)
#     dL_dW3 = a2.T @ dL_dy / len(y_train_over)
#     dL_db3 = dL_dy.mean(axis=0, keepdims=True)
#     dL_da2 = dL_dy @ W3.T
#     dL_dz2 = dL_da2 * relu_deriv(z2)
#     dL_dW2 = a1.T @ dL_dz2 / len(y_train_over)
#     dL_db2 = dL_dz2.mean(axis=0, keepdims=True)
#     dL_da1 = dL_dz2 @ W2.T
#     dL_dz1 = dL_da1 * relu_deriv(z1)
#     dL_dW1 = X_train_over.T @ dL_dz1 / len(y_train_over)
#     dL_db1 = dL_dz1.mean(axis=0, keepdims=True)

#     # ---- Gradient Clipping ----
#     for grad in [dL_dW1, dL_dW2, dL_dW3]:
#         norm = np.linalg.norm(grad)
#         if norm > clip_value:
#             grad *= clip_value / norm

#     # ---- Update Weights ----
#     W1 -= lr * dL_dW1; b1 -= lr * dL_db1
#     W2 -= lr * dL_dW2; b2 -= lr * dL_db2
#     W3 -= lr * dL_dW3; b3 -= lr * dL_db3

#     # ---- Learning-Rate Decay ----
#     lr *= decay_rate

#     # ---- Diagnostics every 200 epochs ----
#     if epoch % 200 == 0:
#         print(f"Epoch {epoch:4d} | Loss={loss:.4f} | "
#               f"grad_norms: [{np.linalg.norm(dL_dW1):.3f}, {np.linalg.norm(dL_dW2):.3f}, {np.linalg.norm(dL_dW3):.3f}] | "
#               f"lr={lr:.5f}")

#     # ---- Early Stopping ----
#     if loss < best_loss - 1e-5:
#         best_loss, no_improve = loss, 0
#     else:
#         no_improve += 1
#     if no_improve >= patience:
#         print(f"\n Early stopping at epoch {epoch} (no improvement for {patience} steps)")
#         break

# # ---------- 7️⃣ Plot Loss ----------
# plt.figure(figsize=(7,4))
# plt.plot(losses)
# plt.title("Training Loss Curve (Oversampling + LR Decay + Clipping)")
# plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.grid(True); plt.show()

# # ---------- 8️⃣ Evaluate ----------
# def predict(X, threshold=0.5):
#     return (sigmoid(relu(relu(X @ W1 + b1) @ W2 + b2) @ W3 + b3) > threshold).astype(int)

# y_hat = predict(X_test)
# accuracy = (y_hat == y_test).mean()
# precision = ((y_hat[y_test==1] == 1).sum() / max((y_hat==1).sum(),1))
# recall = ((y_hat[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
# f1 = 2 * precision * recall / max((precision + recall),1e-9)
# print(f"\n Accuracy : {accuracy:.3f}\n Precision: {precision:.3f}\n Recall   : {recall:.3f}\n F1-score : {f1:.3f}")

# cm = confusion_matrix(y_test, y_hat)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted","Readmitted"]).plot(cmap="Blues")
# plt.title("Confusion Matrix (Advanced ANN)")
# plt.show()

# Accuracy : 0.852
# Precision: 0.171
# Recall   : 0.077
#  F1-score : 0.106
#-test2-
# Accuracy : 0.849
#  Precision: 0.165
# Recall   : 0.080
# F1-score : 0.108


# =============================================================
# ADVANCED ANN (UNDERSAMPLING + LR DECAY + CLIPPING + 2 HIDDEN LAYERS)
# =============================================================

loss_history = []

# ---------- 1️⃣ UNDERSAMPLING ----------
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]

# Take ~2× more negatives than positives for moderate balance
rng = np.random.default_rng(42)

# ⬇️ Option A 3 to 1 
# neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx) * 2, replace=False)
#  Accuracy : 0.835
# Precision: 0.194
# Recall   : 0.141
# F1-score : 0.164

#  Re-evaluation at optimal threshold = 0.330
# Accuracy : 0.579
# Precision: 0.150
# Recall   : 0.580
# F1-score : 0.239


# ⬇️ OPTION B (50/50): uncomment this instead if you want perfect balance
neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)



# Re-evaluation at optimal threshold = 0.502
#  Accuracy : 0.546
#  Precision: 0.151
#  Recall   : 0.586
#  F1-score : 0.241

#test with new features added - mo optimal threshold
# Accuracy : 0.605
# Precision: 0.156
# Recall   : 0.557
# F1-score : 0.243

#  Re-evaluation at optimal threshold = 0.551
#  Accuracy : 0.691
#  Precision: 0.170
#  Recall   : 0.442
#  F1-score : 0.246

new_idx = np.concatenate([pos_idx, neg_sample_idx])
np.random.shuffle(new_idx)

X_train_under = X_train[new_idx]
y_train_under = y_train[new_idx]

print(f"Original dataset: {len(y_train)} samples")
print(f"Undersampled dataset: {len(y_train_under)} samples")
print(f"Class 1 proportion: {y_train_under.mean():.2%}")

# ---------- 2️⃣ Initialize Parameters (He Initialization for ReLU) ----------
rng = np.random.default_rng(42)
hidden1, hidden2, output_dim = 64, 32, 1
input_dim = X_train_under.shape[1]

W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden1))
b1 = np.zeros((1, hidden1))
W2 = rng.normal(0, np.sqrt(2 / hidden1), (hidden1, hidden2))
b2 = np.zeros((1, hidden2))
W3 = rng.normal(0, np.sqrt(2 / hidden2), (hidden2, output_dim))
b3 = np.zeros((1, output_dim))

# ---------- 3️⃣ Activation Functions ----------
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

# ---------- 4️⃣ Binary Cross-Entropy Loss ----------
def bce(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    return -(y_true * np.log(y_pred + 1e-9) +
             (1 - y_true) * np.log(1 - y_pred + 1e-9)).mean()

# ---------- 5️⃣ Hyperparameters ----------
lr = 0.001
decay_rate = 0.999      # slower decay — keeps learning rate alive longer
clip_value = 2.0       # looser clipping — allows stronger updates
epochs = 2500           # give more time to converge
patience = 300          # waits longer before early stopping

losses = []
best_loss = np.inf
no_improve = 0

# ---------- 6️⃣ Training Loop with Diagnostics ----------
for epoch in range(epochs):
    # ---- Forward Pass ----
    z1 = X_train_under @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    y_pred = sigmoid(z3)

    # ---- Loss ----
    loss = bce(y_train_under, y_pred)
    losses.append(loss)

    # ---- Backpropagation ----
    dL_dy = (y_pred - y_train_under)
    dL_dW3 = a2.T @ dL_dy / len(y_train_under)
    dL_db3 = dL_dy.mean(axis=0, keepdims=True)
    dL_da2 = dL_dy @ W3.T
    dL_dz2 = dL_da2 * relu_deriv(z2)
    dL_dW2 = a1.T @ dL_dz2 / len(y_train_under)
    dL_db2 = dL_dz2.mean(axis=0, keepdims=True)
    dL_da1 = dL_dz2 @ W2.T
    dL_dz1 = dL_da1 * relu_deriv(z1)
    dL_dW1 = X_train_under.T @ dL_dz1 / len(y_train_under)
    dL_db1 = dL_dz1.mean(axis=0, keepdims=True)

    # ---- Gradient Clipping ----
    for grad in [dL_dW1, dL_dW2, dL_dW3]:
        norm = np.linalg.norm(grad)
        if norm > clip_value:
            grad *= clip_value / norm

    # ---- Update Weights ----
    W1 -= lr * dL_dW1; b1 -= lr * dL_db1
    W2 -= lr * dL_dW2; b2 -= lr * dL_db2
    W3 -= lr * dL_dW3; b3 -= lr * dL_db3

    # ---- Learning Rate Decay ----
    lr *= decay_rate

    # ---- Diagnostics ----
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss={loss:.4f} | "
              f"grad_norms: [{np.linalg.norm(dL_dW1):.3f}, {np.linalg.norm(dL_dW2):.3f}, {np.linalg.norm(dL_dW3):.3f}] | lr={lr:.6f}")

    # ---- Early Stopping ----
    if loss < best_loss - 1e-5:
        best_loss = loss
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= patience:
        print(f" Early stopping at epoch {epoch} — no improvement for {patience} steps.")
        break

# ---------- 7️⃣ Plot Loss ----------
plt.figure(figsize=(7,4))
plt.plot(losses)
plt.title("Training Loss Curve (Undersampling + LR Decay + Clipping)")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True)
plt.show()
plt.savefig(os.path.join(ann_folder, "ann_training_loss.png"), dpi=300, bbox_inches="tight")
plt.close
# ---------- 8️⃣ Evaluation ----------
def predict(X, threshold=0.5):
    return (sigmoid(relu(relu(X @ W1 + b1) @ W2 + b2) @ W3 + b3) > threshold).astype(int)

y_hat = predict(X_test)
accuracy = (y_hat == y_test).mean()
precision = ((y_hat[y_test==1] == 1).sum() / max((y_hat==1).sum(),1))
recall = ((y_hat[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
f1 = 2 * precision * recall / max((precision + recall),1e-9)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

# ================================================
# PRECISION–RECALL CURVE & F1 vs THRESHOLD
# =====================================================

from sklearn.metrics import precision_recall_curve

# 1️⃣ Get continuous probabilities (not thresholded)
y_prob = sigmoid(relu(relu(X_test @ W1 + b1) @ W2 + b2) @ W3 + b3)

# 2️⃣ Compute precision, recall, thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# 3️⃣ Compute F1 for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

# 4️⃣ Find best threshold (max F1)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n✨ Optimal threshold = {best_thresh:.3f}")
print(f"✨ Best F1 = {best_f1:.3f} | Precision = {precisions[best_idx]:.3f} | Recall = {recalls[best_idx]:.3f}")

# 5️⃣ Plot Precision–Recall curve
plt.figure(figsize=(7,5))
plt.plot(recalls, precisions, color="royalblue", lw=2)
plt.title("Precision–Recall Curve (Undersampled ANN)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

# 6️⃣ Plot F1 vs Threshold
plt.figure(figsize=(7,5))
plt.plot(thresholds, f1_scores[:-1], color="darkorange", lw=2)
plt.axvline(best_thresh, color="red", ls="--", label=f"Best threshold = {best_thresh:.3f}")
plt.title("F1-Score vs Threshold (Undersampled ANN)")
plt.xlabel("Decision Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(ann_folder, "F1_vs_Threshold.png"), dpi=300, bbox_inches="tight")
plt.show()

# ------------------- Evaluation with a treshhold of 0.502----------------------------

opt_thresh = 0.551  # or use `best_thresh` variable if still in memory
y_hat_opt = (y_prob > opt_thresh).astype(int)

accuracy_opt = (y_hat_opt == y_test).mean()
precision_opt = ((y_hat_opt[y_test==1] == 1).sum() / max((y_hat_opt==1).sum(),1))
recall_opt = ((y_hat_opt[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
f1_opt = 2 * precision_opt * recall_opt / max((precision_opt + recall_opt),1e-9)

print(f"\n Re-evaluation at optimal threshold = {opt_thresh:.3f}")
print(f" Accuracy : {accuracy_opt:.3f}")
print(f" Precision: {precision_opt:.3f}")
print(f" Recall   : {recall_opt:.3f}")
print(f" F1-score : {f1_opt:.3f}")

# under option are the results of the tests

# ---------- 9️⃣ Confusion Matrix ----------
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay)

cm = confusion_matrix(y_test, y_hat)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"]).plot(cmap="Blues")
plt.title("Confusion Matrix (Undersampled ANN)")
plt.show()

# =========================RESUTLS=========================
# ⬇️ Option A 3 to 1 
# neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx) * 2, replace=False)
#  Accuracy : 0.835
# Precision: 0.194
# Recall   : 0.141
# F1-score : 0.164
#  Re-evaluation at optimal threshold = 0.330
# Accuracy : 0.579
# Precision: 0.150
# Recall   : 0.580
# F1-score : 0.239


# ⬇️ OPTION B (50/50): uncomment this instead if you want perfect balance
#neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

# Re-evaluation at optimal threshold = 0.502
#  Accuracy : 0.546
#  Precision: 0.151
#  Recall   : 0.586
#  F1-score : 0.241

#test with NEW FEATURE ADDED  no optimal threshold
# Accuracy : 0.605
# Precision: 0.156
# Recall   : 0.557
# F1-score : 0.243

#  Re-evaluation at optimal threshold = 0.551 ------------------------------------- and the winner is---------
#  Accuracy : 0.691
#  Precision: 0.170
#  Recall   : 0.442
#  F1-score : 0.246



#The best-performing and most clinically useful model is the 2-layer ANN trained on 50/50 undersampling, with engineered interaction features, and using a tuned classification threshold from the precision–recall curve.
#This model provides a good balance between identifying high-risk readmissions (recall up to ~0.58) and keeping precision at a workable level (~0.15–0.17), achieving an F1 score of ~0.24–0.25.
#The deeper 3-layer model did not improve F1, so we prefer the simpler, more interpretable 2-layer configuration.


#F1 ≈ 0.25 means your model strikes a decent balance — catching about half of true readmissions, while keeping precision around 15–17%.
# That’s strong for a real-world health dataset...?

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, PrecisionRecallDisplay,
    roc_curve, RocCurveDisplay, roc_auc_score
    )

# ----- AUC -----
auc = roc_auc_score(y_test, y_prob)

# ----- Prepare subplots -----
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ANN Readmission Model – Performance Summary", fontsize=16, weight='bold')

# ===== 1️⃣ Precision–Recall curve =====
prec, rec, _ = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = f1_scores.argmax()
best_f1 = f1_scores[best_idx]

axs[0].plot(rec, prec, color='blue', lw=2, label='PR curve')
axs[0].scatter(rec[best_idx], prec[best_idx], color='red', label=f'Best F1 = {best_f1:.3f}')
axs[0].set_xlabel('Recall')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision–Recall Curve')
axs[0].legend()
axs[0].grid(alpha=0.3)

# ===== 2️⃣ Confusion Matrix =====
cm = confusion_matrix(y_test, y_hat_05)
disp = ConfusionMatrixDisplay(cm)
disp.plot(ax=axs[1], cmap='Blues', colorbar=False)
axs[1].set_title('Confusion Matrix')

# ===== 3️⃣ ROC Curve =====
fpr, tpr, _ = roc_curve(y_test, y_prob)
axs[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
axs[2].plot([0, 1], [0, 1], 'k--', lw=1)
axs[2].set_xlabel('False Positive Rate')
axs[2].set_ylabel('True Positive Rate')
axs[2].set_title('ROC Curve')
axs[2].legend()
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.85)


plt.savefig("ann_readmission_summary.png", dpi=300, bbox_inches="tight")

#------------------------------SAVING------------
import os


ann_folder = os.path.join("models", "ANN")
os.makedirs(ann_folder, exist_ok=True)

# ---------weights--------

import pickle

ann_weights = {
    "W1": W1, "b1": b1,
    "W2": W2, "b2": b2,
    # include W3, W4 if deeper
}
with open(os.path.join(ann_folder, "ann_model_weights.pkl"), "wb") as f:
    pickle.dump(ann_weights, f)

#  -------------Training history----
pd.DataFrame({
    "epoch": range(len(loss_history)),
    "loss": loss_history
}).to_csv(os.path.join(ann_folder, "ann_training_history.csv"), index=False)

plt.savefig(os.path.join(ann_folder, "ann_summary_plots.png"), dpi=300, bbox_inches="tight")

# model setup

import json

ann_metadata = {
    "architecture": "2-layer ANN (32 hidden neurons)",
    "sampling": "50/50 undersampling",
    "best_threshold": 0.55,
    "metrics": {
        "accuracy": 0.691,
        "precision": 0.170,
        "recall": 0.442,
        "f1": 0.246
    }
}

with open(os.path.join(ann_folder, "ann_metadata.json"), "w") as f:
    json.dump(ann_metadata, f, indent=4)



# Save the underlying data as well
pr_df = pd.DataFrame({
    "threshold": list(thresholds) + [1.0],
    "precision": precisions,
    "recall": recalls
})
pr_df.to_csv(os.path.join(ann_folder, "ann_precision_recall_data.csv"), index=False)
print("💾 Saved Precision–Recall curve and CSV data.")


import os

# # =============================================================
# # ADVANCED ANN (3 Hidden Layers + Undersampling + LR Decay + Clipping)
# # =============================================================

# # 3 hidden layers:                (128 → 64 → 32) for deeper feature interactions
# #  He initialization + ReLU       Keeps gradients healthy
# #  Gradient clipping + decay	   Prevents explosions & stabilizes training
# #  Early stopping	               Stops automatically when no improvement
# #  ROC + AUC + CM plots	           Full diagnostic suite
# #  Predict()	                   Supports 3 hidden layers for clean inference

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     precision_recall_curve,
#     roc_curve,
#     auc
# )

# # =============================================================
# # 0. DATA PREP: UNDERSAMPLING
# # =============================================================

# # assume you already have: X_train, y_train, X_test, y_test as numpy arrays
# # y_* should be binary {0,1}

# # pick indices by class
# pos_idx = np.where(y_train == 1)[0]
# neg_idx = np.where(y_train == 0)[0]

# rng = np.random.default_rng(42)

# # ⬇️ OPTION A (current): ~2x negatives per positive (33% positive)
# #neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx)*2, replace=False)
# # Result: Accuracy : 0.868
# #         Precision: 0.215
# #         Recall   : 0.061
# #         F1-score : 0.095


# # ⬇️ OPTION B (50/50): uncomment this instead if you want perfect balance
# #neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

# #Results------------
# # [Threshold=0.5]
# # Accuracy : 0.568
# # Precision: 0.139
# # Recall   : 0.540
# # F1-score : 0.221
# # >>>

# # Optimal threshold = 0.544
# # ✨ Best F1 = 0.226 | Precision = 0.162 | Recall = 0.372

# #  Re-evaluation at optimal threshold = 0.544
# # Accuracy : 0.709
# # Precision: 0.162
# # Recall   : 0.372
# # F1-score : 0.226

# # [Threshold=0.5]
# #  Accuracy : 0.593
# # Precision: 0.143
# # Recall   : 0.517
# #  F1-score : 0.224

# # Re-evaluation at optimal threshold = 0.488
# #  Accuracy : 0.559
# # Precision: 0.141
# #  Recall   : 0.563
# #  F1-score : 0.225



# # merge + shuffle
# new_idx = np.concatenate([pos_idx, neg_sample_idx])
# rng.shuffle(new_idx)

# X_train_under = X_train[new_idx]
# y_train_under = y_train[new_idx]

# # ensure y is column vector for math safety
# y_train_under = y_train_under.reshape(-1, 1)

# print(f"Original dataset: {len(y_train)} samples")
# print(f"Undersampled dataset: {len(y_train_under)} samples")
# print(f"Class 1 proportion: {y_train_under.mean():.2%}")
# print(f"X_train_under shape: {X_train_under.shape}")
# print(f"y_train_under shape: {y_train_under.shape}")

# # =============================================================
# # 1. INITIALIZE NETWORK (3 hidden layers)
# # =============================================================

# # architecture sizes
# input_dim  = X_train_under.shape[1]
# hidden1    = 128
# hidden2    = 64
# hidden3    = 32
# output_dim = 1

# rng = np.random.default_rng(42)

# # He initialization for ReLU layers
# W1 = rng.normal(0, np.sqrt(2 / input_dim),  (input_dim,  hidden1))
# b1 = np.zeros((1, hidden1))

# W2 = rng.normal(0, np.sqrt(2 / hidden1),   (hidden1,    hidden2))
# b2 = np.zeros((1, hidden2))

# W3 = rng.normal(0, np.sqrt(2 / hidden2),   (hidden2,    hidden3))
# b3 = np.zeros((1, hidden3))

# W4 = rng.normal(0, np.sqrt(2 / hidden3),   (hidden3,    output_dim))
# b4 = np.zeros((1, output_dim))

# # =============================================================
# # 2. ACTIVATIONS + LOSS
# # =============================================================

# def relu(x):
#     return np.maximum(0, x)

# def relu_deriv(x):
#     return (x > 0).astype(float)

# def sigmoid(x):
#     x = np.clip(x, -50, 50)
#     return 1 / (1 + np.exp(-x))

# def bce(y_true, y_pred):
#     # y_true, y_pred both shape (N,1)
#     return -(y_true * np.log(y_pred + 1e-9)
#              + (1 - y_true) * np.log(1 - y_pred + 1e-9)).mean()

# # =============================================================
# # 3. HYPERPARAMETERS
# # =============================================================

# lr          = 0.001        # learning
# decay_rate  = 0.998        # how fast lr shrinks each epoch
# clip_value  = 2.0          # max grad norm per layer
# epochs      = 4000
# patience    = 800          # early stopping patience

# losses      = []
# best_loss   = np.inf
# no_improve  = 0

# # =============================================================
# # 4. TRAINING LOOP
# # =============================================================

# for epoch in range(epochs):
#     # ---------- Forward ----------
#     z1 = X_train_under @ W1 + b1                 # (N,128)
#     a1 = relu(z1)
#     z2 = a1 @ W2 + b2                            # (N,64)
#     a2 = relu(z2)
#     z3 = a2 @ W3 + b3                            # (N,32)
#     a3 = relu(z3)
#     z4 = a3 @ W4 + b4                            # (N,1)
#     y_pred = sigmoid(z4)                         # (N,1)

#     # ---------- Loss ----------
#     loss = bce(y_train_under, y_pred)
#     losses.append(loss)

#     # ---------- Backprop ----------
#     # dL/dy_pred for BCE with sigmoid is (y_pred - y_true)
#     dL_dy = (y_pred - y_train_under)             # (N,1)

#     # layer 4 grads
#     dL_dW4 = a3.T @ dL_dy / len(y_train_under)   # (32,1)
#     dL_db4 = dL_dy.mean(axis=0, keepdims=True)   # (1,1)

#     dL_da3 = dL_dy @ W4.T                        # (N,32)

#     # layer 3 grads
#     dL_dz3 = dL_da3 * relu_deriv(z3)             # (N,32)
#     dL_dW3 = a2.T @ dL_dz3 / len(y_train_under)  # (64,32)
#     dL_db3 = dL_dz3.mean(axis=0, keepdims=True)  # (1,32)

#     dL_da2 = dL_dz3 @ W3.T                       # (N,64)

#     # layer 2 grads
#     dL_dz2 = dL_da2 * relu_deriv(z2)             # (N,64)
#     dL_dW2 = a1.T @ dL_dz2 / len(y_train_under)  # (128,64)
#     dL_db2 = dL_dz2.mean(axis=0, keepdims=True)  # (1,64)

#     dL_da1 = dL_dz2 @ W2.T                       # (N,128)

#     # layer 1 grads
#     dL_dz1 = dL_da1 * relu_deriv(z1)             # (N,128)
#     dL_dW1 = X_train_under.T @ dL_dz1 / len(y_train_under)  # (in,128)
#     dL_db1 = dL_dz1.mean(axis=0, keepdims=True)             # (1,128)

#     # ---------- Gradient Clipping ----------
#     for grad in [dL_dW1, dL_dW2, dL_dW3, dL_dW4]:
#         norm = np.linalg.norm(grad)
#         if norm > clip_value:
#             grad *= clip_value / norm

#     # ---------- Update Weights ----------
#     W1 -= lr * dL_dW1; b1 -= lr * dL_db1
#     W2 -= lr * dL_dW2; b2 -= lr * dL_db2
#     W3 -= lr * dL_dW3; b3 -= lr * dL_db3
#     W4 -= lr * dL_dW4; b4 -= lr * dL_db4

#     # ---------- Learning Rate Decay ----------
#     lr *= decay_rate

#     # ---------- Diagnostics ----------
#     if epoch % 200 == 0:
#         print(
#             f"Epoch {epoch:4d} | Loss={loss:.4f} | "
#             f"grad_norms: [{np.linalg.norm(dL_dW1):.3f}, "
#             f"{np.linalg.norm(dL_dW2):.3f}, "
#             f"{np.linalg.norm(dL_dW3):.3f}, "
#             f"{np.linalg.norm(dL_dW4):.3f}] | lr={lr:.6f}"
#         )

#     # ---------- Early Stopping ----------
#     if loss < best_loss - 1e-5:
#         best_loss  = loss
#         no_improve = 0
#     else:
#         no_improve += 1

#     if no_improve >= patience:
#         print(f"\n🛑 Early stopping at epoch {epoch} — no improvement for {patience} steps.")
#         break

# # =============================================================
# # 5. TRAINING LOSS CURVE
# # =============================================================

# plt.figure(figsize=(7,4))
# plt.plot(losses)
# plt.title("Training Loss Curve (3-layer ANN, Undersampling)")
# plt.xlabel("Epoch")
# plt.ylabel("BCE Loss")
# plt.grid(True)
# plt.show()

# # =============================================================
# # 6. EVALUATION (DEFAULT THRESHOLD = 0.5)
# # =============================================================

# # forward pass on test set to get probabilities

# y_test = y_test.reshape(-1)

# z1_test = X_test @ W1 + b1
# a1_test = relu(z1_test)
# z2_test = a1_test @ W2 + b2
# a2_test = relu(z2_test)
# z3_test = a2_test @ W3 + b3
# a3_test = relu(z3_test)
# z4_test = a3_test @ W4 + b4
# y_prob = sigmoid(z4_test).reshape(-1)  # shape (N_test,)

# y_hat_05 = (y_prob > 0.5).astype(int)

# accuracy = (y_hat_05 == y_test).mean()
# precision = ((y_hat_05[y_test==1] == 1).sum() / max((y_hat_05==1).sum(),1))
# recall = ((y_hat_05[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
# f1 = 2 * precision * recall / max((precision + recall),1e-9)

# print(f"\n[Threshold=0.5]")
# print(f"Accuracy : {accuracy:.3f}")
# print(f"Precision: {precision:.3f}")
# print(f"Recall   : {recall:.3f}")
# print(f"F1-score : {f1:.3f}")

# # =============================================================
# # 7. THRESHOLD TUNING VIA PRECISION-RECALL
# # =============================================================

# precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
# f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

# best_idx = np.argmax(f1_scores)
# best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
# best_f1 = f1_scores[best_idx]

# print(f"\n✨ Optimal threshold = {best_thresh:.3f}")
# print(f"✨ Best F1 = {best_f1:.3f} | Precision = {precisions[best_idx]:.3f} | Recall = {recalls[best_idx]:.3f}")

# y_hat_opt = (y_prob > best_thresh).astype(int)

# accuracy_opt = (y_hat_opt == y_test).mean()
# precision_opt = ((y_hat_opt[y_test==1] == 1).sum() / max((y_hat_opt==1).sum(),1))
# recall_opt = ((y_hat_opt[y_test==1] == 1).sum() / max((y_test==1).sum(),1))
# f1_opt = 2 * precision_opt * recall_opt / max((precision_opt + recall_opt),1e-9)

# print(f"\ Re-evaluation at optimal threshold = {best_thresh:.3f}")
# print(f" Accuracy : {accuracy_opt:.3f}")
# print(f"Precision: {precision_opt:.3f}")
# print(f" Recall   : {recall_opt:.3f}")
# print(f" F1-score : {f1_opt:.3f}")




# # =============================================================
# # 8. ROC CURVE + AUC
# # =============================================================

# fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"\n🏁 ROC AUC = {roc_auc:.3f}")

# plt.figure(figsize=(7,5))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Recall / Sensitivity)')
# plt.title('ROC Curve (3-layer ANN)')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# # =============================================================
# # 9. CONFUSION MATRIX (AT OPTIMAL THRESHOLD)
# # =============================================================

# cm = confusion_matrix(y_test, y_hat_opt)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                               display_labels=["Not Readmitted", "Readmitted"])
# disp.plot(cmap="Blues")
# plt.title(f"Confusion Matrix (3-layer ANN @ {best_thresh:.2f} threshold)")
# plt.show()



#--------------------------------RESULTS----------------------

# ⬇️ OPTION A (current): ~2x negatives per positive (33% positive)
#neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx)*2, replace=False)
# Result: Accuracy : 0.868
#         Precision: 0.215
#         Recall   : 0.061
#         F1-score : 0.095


# ⬇️ OPTION B (50/50): uncomment this instead if you want perfect balance
#neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

#Results------------
# [Threshold=0.5]
# Accuracy : 0.568
# Precision: 0.139
# Recall   : 0.540
# F1-score : 0.221
# >>>

# Optimal threshold = 0.544
# ✨ Best F1 = 0.226 | Precision = 0.162 | Recall = 0.372

#  Re-evaluation at optimal threshold = 0.544
# Accuracy : 0.709
# Precision: 0.162
# Recall   : 0.372
# F1-score : 0.226

# [Threshold=0.5]
#  Accuracy : 0.593
# Precision: 0.143
# Recall   : 0.517
#  F1-score : 0.224

# Re-evaluation at optimal threshold = 0.488
#  Accuracy : 0.559
# Precision: 0.141
#  Recall   : 0.563
#  F1-score : 0.225