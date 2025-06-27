import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from Test import test_model

# === Evaluate model ===
def evaluate_model():
    all_labels, all_predictions, avg_test_loss, class_names = test_model()

    
    # Overall test accuracy
    correct_total = np.sum(np.array(all_predictions) == np.array(all_labels))
    total_samples = len(all_labels)
    test_accuracy = correct_total / total_samples * 100


    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=np.arange(len(class_names)))

    # Classification report (dict format for structured access)
    class_report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    print("\n📊 Classification Summary with Sensitivity and Specificity:\n")
    for i, cls in enumerate(class_names):
        metrics = class_report[cls]
        support = int(metrics["support"])
        recall = metrics['recall']
        correct = int(metrics["recall"] * support)
        incorrect = support - correct

        # Sensitivity = recall
        sensitivity = recall

        # Specificity = TN / (TN + FP)
        TP = conf_matrix[i, i]
        FN = conf_matrix[i, :].sum() - TP
        FP = conf_matrix[:, i].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f"{cls:10} | Support: {support:3} | Correct: {correct:3} | Incorrect: {incorrect:3} | "
      f"Sensitivity: {sensitivity*100:.2f}% | Specificity: {specificity*100:.2f}%")



    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")
    print(f"\n✅ Average Test Loss: {avg_test_loss:.4f}")

    return conf_matrix, class_names

# === Plot confusion matrix ===
def plot_confusion_matrix(conf_matrix, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix - 10-Class FTU Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# === Run if executed ===
if __name__ == '__main__':
    conf_matrix, class_names = evaluate_model()
    plot_confusion_matrix(conf_matrix, class_names)
