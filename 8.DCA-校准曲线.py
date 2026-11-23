import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
from matplotlib.gridspec import GridSpec
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
data = pd.read_csv('SAP60.csv')

# Features and target
X = data.drop('SAP', axis=1)
y = data['SAP']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]


# DCA curve function - Fixed version
def dca_curve(y_true, y_prob, thresholds):
    """
    Calculate net benefits for Decision Curve Analysis (DCA)

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    thresholds : array-like
        List of threshold values

    Returns:
    --------
    net_benefits : array
        Net benefit for each threshold
    all_positive_net_benefit : array
        Net benefit when all samples are positive
    all_negative_net_benefit : array
        Net benefit when all samples are negative
    """
    net_benefits = []
    n = len(y_true)

    for threshold in thresholds:
        # Predict positive if probability >= threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate true positives and false positives
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Calculate net benefit
        # Special handling when threshold = 1.0 to avoid division by zero
        if threshold == 1.0:
            net_benefit = (tp / n)
        else:
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)

    # All positive net benefit: equals prevalence (constant)
    prevalence = np.mean(y_true)
    all_positive_net_benefit = np.full_like(thresholds, prevalence)

    # All negative net benefit: equals 0 (constant)
    all_negative_net_benefit = np.zeros_like(thresholds)

    return np.array(net_benefits), all_positive_net_benefit, all_negative_net_benefit


# 5-fold cross validation
def perform_cross_validation(X, y, model, n_splits=5):
    """
    Perform 5-fold cross validation and calculate various evaluation metrics

    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Target variable
    model : sklearn estimator
        Machine learning model
    n_splits : int
        Number of folds for cross validation

    Returns:
    --------
    cv_results : dict
        Dictionary containing all evaluation results
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results from each fold
    cv_results = {
        'fprs': [],
        'tprs': [],
        'roc_aucs': [],
        'precisions': [],
        'recalls': [],
        'pr_aucs': [],
        'prob_trues': [],
        'prob_preds': [],
        'brier_scores': [],
        'net_benefits': [],
        'all_positive_nb': [],
        'all_negative_nb': [],
        'thresholds': np.linspace(0.01, 0.99, 100)  # DCA thresholds
    }

    # Store all predictions
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Clone and train model
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)

        # Predict probabilities
        y_prob_fold = fold_model.predict_proba(X_val_fold)[:, 1]

        # Store true labels and predicted probabilities
        all_y_true.extend(y_val_fold)
        all_y_prob.extend(y_prob_fold)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_val_fold, y_prob_fold)
        roc_auc = auc(fpr, tpr)
        cv_results['fprs'].append(fpr)
        cv_results['tprs'].append(tpr)
        cv_results['roc_aucs'].append(roc_auc)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_val_fold, y_prob_fold)
        pr_auc = auc(recall, precision)
        cv_results['precisions'].append(precision)
        cv_results['recalls'].append(recall)
        cv_results['pr_aucs'].append(pr_auc)

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_val_fold, y_prob_fold, n_bins=10, strategy='quantile')
            cv_results['prob_trues'].append(prob_true)
            cv_results['prob_preds'].append(prob_pred)
        except:
            # Use fewer bins if sample size is small
            prob_true, prob_pred = calibration_curve(y_val_fold, y_prob_fold, n_bins=5, strategy='quantile')
            cv_results['prob_trues'].append(prob_true)
            cv_results['prob_preds'].append(prob_pred)

        # Brier score
        brier_score = brier_score_loss(y_val_fold, y_prob_fold)
        cv_results['brier_scores'].append(brier_score)

        # DCA curve - using fixed function
        thresholds = cv_results['thresholds']
        net_benefits, all_positive_nb, all_negative_nb = dca_curve(
            y_val_fold, y_prob_fold, thresholds
        )
        cv_results['net_benefits'].append(net_benefits)
        cv_results['all_positive_nb'].append(all_positive_nb)
        cv_results['all_negative_nb'].append(all_negative_nb)

    # Convert all predictions to arrays
    cv_results['all_y_true'] = np.array(all_y_true)
    cv_results['all_y_prob'] = np.array(all_y_prob)

    # Calculate mean ROC curve
    cv_results['mean_fpr'] = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(cv_results['mean_fpr'])
    for i in range(n_splits):
        fpr_interp = np.interp(cv_results['mean_fpr'], cv_results['fprs'][i], cv_results['tprs'][i])
        mean_tpr += fpr_interp
    mean_tpr /= n_splits
    cv_results['mean_tpr'] = mean_tpr
    cv_results['mean_roc_auc'] = auc(cv_results['mean_fpr'], mean_tpr)

    # Calculate mean PR curve
    cv_results['mean_recall'] = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(cv_results['mean_recall'])
    for i in range(n_splits):
        recall_rev = cv_results['recalls'][i][::-1]
        precision_rev = cv_results['precisions'][i][::-1]
        precision_interp = np.interp(cv_results['mean_recall'], recall_rev, precision_rev)
        mean_precision += precision_interp
    mean_precision /= n_splits
    cv_results['mean_precision'] = mean_precision
    cv_results['mean_pr_auc'] = auc(cv_results['mean_recall'], mean_precision)

    # Calculate mean calibration curve
    if len(cv_results['prob_preds']) > 0:
        cv_results['mean_prob_pred'] = np.linspace(0, 1, 100)
        mean_prob_true = np.zeros_like(cv_results['mean_prob_pred'])
        valid_curves = 0
        for i in range(len(cv_results['prob_preds'])):
            try:
                prob_pred = cv_results['prob_preds'][i]
                prob_true = cv_results['prob_trues'][i]
                if len(prob_pred) > 1:
                    true_interp = np.interp(cv_results['mean_prob_pred'], prob_pred, prob_true)
                    mean_prob_true += true_interp
                    valid_curves += 1
            except:
                continue
        if valid_curves > 0:
            mean_prob_true /= valid_curves
        cv_results['mean_prob_true'] = mean_prob_true

    # Calculate mean DCA curves
    if len(cv_results['net_benefits']) > 0:
        cv_results['mean_net_benefits'] = np.mean(cv_results['net_benefits'], axis=0)
        cv_results['mean_all_positive_nb'] = np.mean(cv_results['all_positive_nb'], axis=0)
        cv_results['mean_all_negative_nb'] = np.mean(cv_results['all_negative_nb'], axis=0)
    else:
        cv_results['mean_net_benefits'] = np.zeros_like(cv_results['thresholds'])
        cv_results['mean_all_positive_nb'] = np.zeros_like(cv_results['thresholds'])
        cv_results['mean_all_negative_nb'] = np.zeros_like(cv_results['thresholds'])

    # Mean Brier score
    cv_results['mean_brier_score'] = np.mean(cv_results['brier_scores'])

    return cv_results


# Perform 5-fold cross validation
print("Performing 5-fold cross validation...")
cv_results = perform_cross_validation(X, y, model, n_splits=5)
print("Cross validation completed!")

# Calculate metrics on test set
y_pred_test = (y_prob >= 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_prob)
test_brier = brier_score_loss(y_test, y_prob)

# Create visualization
plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=plt.gcf())

# 1. ROC Curve
ax1 = plt.subplot(gs[0, 0])

# Plot ROC curves for each fold
for i in range(5):
    ax1.plot(cv_results['fprs'][i], cv_results['tprs'][i], lw=1, alpha=0.3,
             label=f'Fold {i + 1} (AUC = {cv_results["roc_aucs"][i]:.3f})')
# Plot mean ROC curve
ax1.plot(cv_results['mean_fpr'], cv_results['mean_tpr'], color='red', lw=2,
         label=f'Mean ROC (AUC = {cv_results["mean_roc_auc"]:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve (5-fold CV)')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax2 = plt.subplot(gs[0, 1])
# Plot PR curves for each fold
for i in range(5):
    ax2.plot(cv_results['recalls'][i], cv_results['precisions'][i], lw=1, alpha=0.3,
             label=f'Fold {i + 1} (AUC = {cv_results["pr_aucs"][i]:.3f})')
# Plot mean PR curve
ax2.plot(cv_results['mean_recall'], cv_results['mean_precision'], color='red', lw=2,
         label=f'Mean PR (AUC = {cv_results["mean_pr_auc"]:.3f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve (5-fold CV)')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. DCA Curve - Fixed version
ax3 = plt.subplot(gs[0, 2])
# Plot DCA curves for each fold
for i in range(5):
    ax3.plot(cv_results['thresholds'], cv_results['net_benefits'][i], lw=1, alpha=0.3,
             label=f'Fold {i + 1}')
# Plot mean DCA curve
ax3.plot(cv_results['thresholds'], cv_results['mean_net_benefits'],
         label='Model', color='red', lw=2)
ax3.plot(cv_results['thresholds'], cv_results['mean_all_positive_nb'],
         label='All Positive', linestyle='--', color='blue', lw=2)
ax3.plot(cv_results['thresholds'], cv_results['mean_all_negative_nb'],
         label='All Negative', linestyle='--', color='green', lw=2)
ax3.set_xlim([0.0, 1.0])
ax3.set_xlabel('Threshold Probability')
ax3.set_ylabel('Net Benefit')
ax3.set_title('Decision Curve Analysis (DCA) (5-fold CV)')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Calibration Curve
ax4 = plt.subplot(gs[1, 0])
# Plot calibration curves for each fold
for i in range(5):
    if len(cv_results['prob_preds'][i]) > 0:
        ax4.plot(cv_results['prob_preds'][i], cv_results['prob_trues'][i],
                 marker='o', alpha=0.3, label=f'Fold {i + 1}')
# Plot mean calibration curve
if 'mean_prob_true' in cv_results:
    ax4.plot(cv_results['mean_prob_pred'], cv_results['mean_prob_true'],
             marker='o', label='Mean Calibration', color='red', lw=2)
ax4.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='gray')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Actual Probability')
ax4.set_title('Calibration Curve (5-fold CV)')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Probability Distribution Histogram
ax5 = plt.subplot(gs[1, 1])
# Use all cross-validation predictions
y_true_all = cv_results['all_y_true']
y_prob_all = cv_results['all_y_prob']

ax5.hist(y_prob_all[y_true_all == 0], bins=20, alpha=0.6,
         label=f'Negative (n={np.sum(y_true_all == 0)})', color='blue', edgecolor='black')
ax5.hist(y_prob_all[y_true_all == 1], bins=20, alpha=0.6,
         label=f'Positive (n={np.sum(y_true_all == 1)})', color='red', edgecolor='black')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Frequency')
ax5.set_title('Predicted Probability Distribution (5-fold CV)')
ax5.legend(loc='upper center', fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Performance Metrics Table
ax6 = plt.subplot(gs[1, 2])
ax6.axis('tight')
ax6.axis('off')

# Calculate cross-validation metrics
y_pred_cv = (cv_results['all_y_prob'] >= 0.5).astype(int)
cv_accuracy = accuracy_score(cv_results['all_y_true'], y_pred_cv)
cv_f1 = f1_score(cv_results['all_y_true'], y_pred_cv)
cv_precision = precision_score(cv_results['all_y_true'], y_pred_cv)
cv_recall = recall_score(cv_results['all_y_true'], y_pred_cv)
cv_roc_auc = roc_auc_score(cv_results['all_y_true'], cv_results['all_y_prob'])
cv_brier = brier_score_loss(cv_results['all_y_true'], cv_results['all_y_prob'])

metrics_data = [
    ['Metric', 'CV (Mean ± Std)', 'Test Set'],
    ['ROC AUC', f'{cv_roc_auc:.3f} ± {np.std(cv_results["roc_aucs"]):.3f}', f'{test_roc_auc:.3f}'],
    ['PR AUC', f'{cv_results["mean_pr_auc"]:.3f} ± {np.std(cv_results["pr_aucs"]):.3f}', '-'],
    ['Brier Score', f'{cv_results["mean_brier_score"]:.3f} ± {np.std(cv_results["brier_scores"]):.3f}',
     f'{test_brier:.3f}'],
    ['Accuracy', f'{cv_accuracy:.3f}', f'{test_accuracy:.3f}'],
    ['F1 Score', f'{cv_f1:.3f}', f'{test_f1:.3f}'],
    ['Precision', f'{cv_precision:.3f}', f'{test_precision:.3f}'],
    ['Recall', f'{cv_recall:.3f}', f'{test_recall:.3f}']
]

table = ax6.table(cellText=metrics_data,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style the table
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header row
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white', fontweight='bold')
    elif j == 0:  # First column
        cell.set_facecolor('#eaeaf2')
    elif j == 1:  # CV column
        cell.set_facecolor('#d4e6f1')
    elif j == 2:  # Test column
        cell.set_facecolor('#d5f5e3')

plt.suptitle('XGBoost Model Performance Evaluation (5-fold CV + Test Set Validation)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# Print detailed results
print("\n" + "=" * 60)
print("Model Performance Evaluation Results")
print("=" * 60)
print(f"\n【Cross-Validation Results (5-fold)】")
print(f"  ROC AUC: {cv_roc_auc:.4f} ± {np.std(cv_results['roc_aucs']):.4f}")
print(f"  PR AUC:  {cv_results['mean_pr_auc']:.4f} ± {np.std(cv_results['pr_aucs']):.4f}")
print(f"  Brier Score: {cv_results['mean_brier_score']:.4f} ± {np.std(cv_results['brier_scores']):.4f}")
print(f"  Accuracy: {cv_accuracy:.4f}")
print(f"  F1 Score: {cv_f1:.4f}")
print(f"  Precision: {cv_precision:.4f}")
print(f"  Recall: {cv_recall:.4f}")

print(f"\n【Test Set Results】")
print(f"  ROC AUC: {test_roc_auc:.4f}")
print(f"  Brier Score: {test_brier:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print("=" * 60)