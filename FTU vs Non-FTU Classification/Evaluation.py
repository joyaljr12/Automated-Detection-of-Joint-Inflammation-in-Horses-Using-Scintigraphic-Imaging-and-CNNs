import numpy as np
from Test import test_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define class names for binary classification
class_names = ['Non-FTU', 'FTU']
num_classes = len(class_names)

def evaluate_model():
    
    # Get predictions & true labels from test_model()
    all_labels, all_predictions, _ = test_model()

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels= np.arange(num_classes))

    #compute sensitivity and specificity
    TN, FP, FN, TP = conf_matrix.ravel()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 

    # Generate classification report
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)

    print("Classification Report:\n", class_report)

    # Print sensitivity and specificity
    print(f"✅ Sensitivity (Recall of FTU): {sensitivity:.4f}")
    print(f"✅ Specificity (Recall of Non-FTU): {specificity:.4f}")

    return conf_matrix, class_report, sensitivity, specificity

def plot_confusion_matrix(conf_matrix):

    disp = ConfusionMatrixDisplay(confusion_matrix= conf_matrix, display_labels= class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('confusion matrix for FTU and Non FTU classification')
    plt.gca().set_xlabel('Predicted labels')
    plt.gca().set_ylabel('True labels')
    plt.tight_layout()
    plt.show()
    

# Run evaluation if script is executed directly
if __name__ == '__main__':
    conf_matrix, class_report, sensitivity, specificity  = evaluate_model()
    plot_confusion_matrix(conf_matrix)

