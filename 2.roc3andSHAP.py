# ====================================================
# SAP 危险因素预测：多模型比较 + SHAP 可解释性分析
# ====================================================
# 功能：
# 1. 训练多种机器学习模型（LR、SVM、RF、XGBoost、Stacking等）
# 2. 5折交叉验证评估
# 3. 保存所有结果到CSV
# 4. 绘制ROC曲线、PR曲线
# 5. 对最佳模型进行SHAP可解释性分析
# ====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve, accuracy_score,
    recall_score, confusion_matrix, f1_score
)
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# XGBoost
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost未安装，将跳过XGBoost模型")
    XGB_AVAILABLE = False

# SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP未安装，将跳过SHAP分析")
    SHAP_AVAILABLE = False


# ====================================================
# 1. 数据加载
# ====================================================
def load_data(filepath, shuffle=True, random_state=42):
    """加载数据"""
    data = pd.read_csv(filepath)

    if shuffle:
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"数据形状: {data.shape}")
    print(f"列名: {data.columns.tolist()}")


    # 提取特征和标签
    y = data["Death"].values
    X = data.iloc[:, 0:31].values

    # 获取特征名称
    feature_names = data.columns[0:31].tolist()

    print(f"样本数: {len(y)}")
    print(f"Death=0: {sum(y == 0)} 例")
    print(f"Death=1: {sum(y == 1)} 例")
    print(f"特征数: {X.shape[1]}")

    return X, y, feature_names, data


# ====================================================
# 2. 定义模型
# ====================================================
def get_models(X, y):
    """定义所有要比较的模型"""

    # 计算正类比例，用于处理不平衡数据
    pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1.0

    models = {}

    # 1. 线性模型
    #models["Logistic Regression"] = LogisticRegression(
    #    max_iter=1000, class_weight='balanced', random_state=42
    #)

    # 2. 支持向量机
    models["SVM (RBF)"] = SVC(probability=True, class_weight='balanced', random_state=42)
    models["Linear SVM"] = LinearSVC(class_weight='balanced', random_state=42, max_iter=1000)

    # 3. 决策树
    models["Decision Tree"] = DecisionTreeClassifier(class_weight='balanced', random_state=42)

    # 4. 集成学习
    models["Random Forest"] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    models["Extra Trees"] = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # 5. 梯度提升
    models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    models["AdaBoost"] = AdaBoostClassifier(n_estimators=50, random_state=42)


    # 7. 神经网络
    models["MLP (Deep)"] = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam',
        alpha=0.001, batch_size=32, learning_rate='adaptive', max_iter=500,
        early_stopping=True, random_state=42)

    # 8. 贝叶斯
    models["Gaussian Naive Bayes"] = GaussianNB()

    # 9. 判别分析
    #models["Linear Discriminant"] = LinearDiscriminantAnalysis()
    models["Quadratic Discriminant"] = QuadraticDiscriminantAnalysis()

    # 10. K近邻
    #models["KNN (k=5)"] = KNeighborsClassifier(n_neighbors=5)
    #models["KNN (k=10)"] = KNeighborsClassifier(n_neighbors=10)

    # 11. XGBoost
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        )

    print(f"总共 {len(models)} 个模型将被测试")
    print("模型列表:", list(models.keys()))

    return models


# ====================================================
# 3. SHAP 分析函数（自动适配模型类型）
# ====================================================
def get_shap_explainer(model, X_train, model_name):
    """根据模型类型自动选择合适的 SHAP Explainer"""

    model_type = type(model).__name__
    print(f"    检测到模型类型: {model_type}")

    # 树模型
    tree_models = ['XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier',
                   'GradientBoostingClassifier', 'DecisionTreeClassifier',
                   'ExtraTreesClassifier', 'BaggingClassifier']

    if any(t in model_type for t in tree_models):
        try:
            explainer = shap.TreeExplainer(model)
            print("    使用 TreeExplainer")
            return explainer, 'tree'
        except Exception as e:
            print(f"    TreeExplainer 失败: {str(e)[:50]}, 尝试 KernelExplainer...")

    # 线性模型
    linear_models = ['LogisticRegression', 'LinearSVC', 'LinearDiscriminantAnalysis']
    if any(t in model_type for t in linear_models):
        try:
            explainer = shap.LinearExplainer(model, X_train)
            print("    使用 LinearExplainer")
            return explainer, 'linear'
        except Exception as e:
            print(f"    LinearExplainer 失败: {str(e)[:50]}, 尝试 KernelExplainer...")

    # 默认使用 KernelExplainer
    n_background = min(100, X_train.shape[0])
    background_idx = np.random.choice(X_train.shape[0], n_background, replace=False)
    X_background = X_train[background_idx]

    def predict_proba(X):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        except AttributeError:
            try:
                from scipy.special import expit
                return expit(model.decision_function(X))
            except AttributeError:
                return model.predict(X).astype(float)

    try:
        explainer = shap.KernelExplainer(predict_proba, X_background)
        print(f"    使用 KernelExplainer (背景数据: {n_background} 样本)")
        return explainer, 'kernel'
    except Exception as e:
        print(f"    KernelExplainer 失败: {str(e)}")
        return None, None


def perform_shap_analysis(model, X_train, X_test, feature_names, model_name, results_dir, max_samples=300):
    """对模型进行完整的 SHAP 可解释性分析"""

    print(f"\n{'=' * 60}")
    print(f"正在对 {model_name} 进行 SHAP 分析...")
    print('=' * 60)

    if not SHAP_AVAILABLE:
        print("  SHAP 未安装，跳过")
        return None, None, None

    shap_dir = os.path.join(results_dir, f'SHAP_{model_name.replace(" ", "_")}')
    os.makedirs(shap_dir, exist_ok=True)

    # 采样加速
    if X_test.shape[0] > max_samples:
        sample_idx = np.random.choice(X_test.shape[0], max_samples, replace=False)
        X_test_sample = X_test[sample_idx]
        print(f"  采样 {max_samples}/{X_test.shape[0]} 个样本")
    else:
        X_test_sample = X_test

    try:
        # 选择解释器
        explainer, model_type = get_shap_explainer(model, X_train, model_name)
        if explainer is None:
            return None, None, None

        # 计算 SHAP 值
        print(f"  计算 SHAP 值...")

        if model_type == 'tree':
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values_pos = shap_values[1]
            else:
                shap_values_pos = shap_values
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
        elif model_type == 'linear':
            shap_values_pos = explainer.shap_values(X_test_sample)
            expected_value = explainer.expected_value
        else:
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_pos = shap_values
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

        print(f"  SHAP 值计算完成")

        # 保存 SHAP 值
        shap_df = pd.DataFrame(shap_values_pos, columns=feature_names)
        try:
            predictions = model.predict_proba(X_test_sample)[:, 1]
        except:
            predictions = model.predict(X_test_sample)
        shap_df['Prediction'] = predictions
        shap_df.to_csv(os.path.join(shap_dir, 'shap_values.csv'), index=False)

        # 绘制 Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_pos, X_test_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, '1_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制 Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_pos, X_test_sample, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, '2_feature_importance_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 特征重要性表格
        feature_importance = np.abs(shap_values_pos).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_|SHAP|': feature_importance,
            'Std_Dev': np.std(np.abs(shap_values_pos), axis=0)
        }).sort_values('Mean_|SHAP|', ascending=False)
        importance_df.to_csv(os.path.join(shap_dir, '3_feature_importance.csv'), index=False)

        # 绘制 Top 6 依赖图
        top_features = importance_df.head(6)['Feature'].tolist()
        for feat in top_features:
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feat, shap_values_pos, X_test_sample,
                                     feature_names=feature_names, show=False)
                plt.title(f'SHAP Dependence - {feat}', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, f'dependence_{feat}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass

        print(f"  SHAP 分析完成，结果保存在 {shap_dir}")
        print(f"\n  Top 10 重要特征:")
        print(importance_df.head(10).to_string(index=False))

        return shap_values_pos, explainer, importance_df

    except Exception as e:
        print(f"  SHAP 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ====================================================
# 4. 训练和评估函数
# ====================================================
def train_and_evaluate(X, y, models, feature_names, results_dir, n_folds=5):
    """训练模型并进行交叉验证评估"""

    pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1.0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 存储所有结果
    all_fold_results = []
    all_roc_data = {}
    all_pr_data = {}

    # 存储模型对象（用于后续 SHAP 分析）
    trained_models = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"训练模型: {model_name} ({list(models.keys()).index(model_name) + 1}/{len(models)})")
        print('=' * 60)

        # 存储当前模型的折叠结果
        fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
        fold_roc_aucs, fold_pr_aucs = [], []
        fold_tprs, fold_tnrs = [], []

        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        precisions, pr_aucs = [], []
        mean_recall = np.linspace(0, 1, 100)

        successful_folds = 0

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model.fit(X_train, y_train)

                # 预测
                y_pred = model.predict(X_test)

                # 概率预测
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except AttributeError:
                    if hasattr(model, 'decision_function'):
                        from scipy.special import expit
                        y_pred_proba = expit(model.decision_function(X_test))
                    else:
                        y_pred_proba = y_pred.astype(float)

                # 计算指标
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # ROC
                if len(np.unique(y_pred_proba)) > 1:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc_val = auc(fpr, tpr)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    aucs.append(roc_auc_val)
                else:
                    roc_auc_val = 0.5
                    tprs.append(np.zeros_like(mean_fpr))
                    aucs.append(0.5)

                # PR
                if len(np.unique(y_pred_proba)) > 1:
                    p, r, _ = precision_recall_curve(y_test, y_pred_proba)
                    pr_auc_val = auc(r, p)
                    p_interp = np.interp(mean_recall, r[::-1], p[::-1])
                    precisions.append(p_interp)
                    pr_aucs.append(pr_auc_val)
                else:
                    pr_auc_val = np.mean(y_test)
                    precisions.append(np.ones_like(mean_recall) * np.mean(y_test))
                    pr_aucs.append(np.mean(y_test))

                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    sensitivity = specificity = 0

                fold_accuracies.append(acc)
                fold_precisions.append(prec)
                fold_recalls.append(rec)
                fold_f1s.append(f1)
                fold_roc_aucs.append(roc_auc_val)
                fold_pr_aucs.append(pr_auc_val)
                fold_tprs.append(sensitivity)
                fold_tnrs.append(specificity)

                # 保存每一折结果
                all_fold_results.append({
                    'Model': model_name, 'Fold': fold,
                    'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1,
                    'ROC_AUC': roc_auc_val, 'PR_AUC': pr_auc_val,
                    'Sensitivity': sensitivity, 'Specificity': specificity,
                    'Train_Samples': len(train_idx), 'Test_Samples': len(test_idx)
                })

                successful_folds += 1

            except Exception as e:
                print(f"  Fold {fold} 失败: {str(e)[:80]}")
                all_fold_results.append({
                    'Model': model_name, 'Fold': fold, 'Error': str(e)[:200],
                    'Train_Samples': len(train_idx), 'Test_Samples': len(test_idx)
                })
                continue

        if successful_folds == 0:
            print(f"  模型 {model_name} 所有fold失败，跳过")
            continue

        # 计算平均值和标准差
        mean_metrics = {
            'Accuracy': np.mean(fold_accuracies), 'Std_Accuracy': np.std(fold_accuracies),
            'Precision': np.mean(fold_precisions), 'Std_Precision': np.std(fold_precisions),
            'Recall': np.mean(fold_recalls), 'Std_Recall': np.std(fold_recalls),
            'F1': np.mean(fold_f1s), 'Std_F1': np.std(fold_f1s),
            'ROC_AUC': np.mean(fold_roc_aucs), 'Std_ROC_AUC': np.std(fold_roc_aucs),
            'PR_AUC': np.mean(fold_pr_aucs), 'Std_PR_AUC': np.std(fold_pr_aucs),
            'Sensitivity': np.mean(fold_tprs), 'Specificity': np.mean(fold_tnrs),
            'Successful_Folds': successful_folds
        }

        all_fold_results.append({
            'Model': model_name, 'Fold': 'MEAN', **mean_metrics
        })

        # 存储 ROC/PR 数据
        if tprs:
            all_roc_data[model_name] = {
                'mean_fpr': mean_fpr,
                'mean_tpr': np.mean(tprs, axis=0),
                'std_tpr': np.std(tprs, axis=0),
                'mean_auc': mean_metrics['ROC_AUC'],
                'std_auc': mean_metrics['Std_ROC_AUC']
            }
            all_pr_data[model_name] = {
                'mean_recall': mean_recall,
                'mean_precision': np.mean(precisions, axis=0),
                'std_precision': np.std(precisions, axis=0),
                'mean_pr_auc': mean_metrics['PR_AUC'],
                'std_pr_auc': mean_metrics['Std_PR_AUC']
            }

        # 保存训练好的模型（用于SHAP）
        # 用全部数据重新训练一个最终模型
        final_model = model.__class__(**model.get_params())
        final_model.fit(X, y)
        trained_models[model_name] = final_model

        # 打印结果
        print(f"\n  {model_name} 性能汇总:")
        print(f"    ROC AUC: {mean_metrics['ROC_AUC']:.4f} ± {mean_metrics['Std_ROC_AUC']:.4f}")
        print(f"    PR AUC:  {mean_metrics['PR_AUC']:.4f} ± {mean_metrics['Std_PR_AUC']:.4f}")
        print(f"    F1:      {mean_metrics['F1']:.4f}")
        print(f"    Recall:  {mean_metrics['Recall']:.4f}")
        print(f"    Prec:    {mean_metrics['Precision']:.4f}")

    return all_fold_results, all_roc_data, all_pr_data, trained_models


# ====================================================
# 5. 绘图函数
# ====================================================
def plot_roc_curves(all_roc_data, results_dir, top_n=6):
    """绘制 ROC 曲线"""
    if not all_roc_data:
        return

    # 按 AUC 排序取前 N 个
    sorted_models = sorted(all_roc_data.items(), key=lambda x: x[1]['mean_auc'], reverse=True)[:top_n]

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_models)))

    for idx, (model_name, data) in enumerate(sorted_models):
        plt.plot(data['mean_fpr'], data['mean_tpr'], color=colors[idx], lw=2,
                 label=f'{model_name} (AUC = {data["mean_auc"]:.3f} ± {data["std_auc"]:.3f})')
        plt.fill_between(data['mean_fpr'],
                         data['mean_tpr'] - data['std_tpr'],
                         data['mean_tpr'] + data['std_tpr'],
                         color=colors[idx], alpha=0.1)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ROC_Curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存")


def plot_pr_curves(all_pr_data, results_dir, y, top_n=6):
    """绘制 PR 曲线"""
    if not all_pr_data:
        return

    sorted_models = sorted(all_pr_data.items(), key=lambda x: x[1]['mean_pr_auc'], reverse=True)[:top_n]

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_models)))
    pos_rate = np.mean(y)

    for idx, (model_name, data) in enumerate(sorted_models):
        plt.plot(data['mean_recall'], data['mean_precision'], color=colors[idx], lw=2,
                 label=f'{model_name} (AP = {data["mean_pr_auc"]:.3f} ± {data["std_pr_auc"]:.3f})')

    plt.axhline(y=pos_rate, color='gray', linestyle='--', label=f'Random (AP = {pos_rate:.3f})')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'PR_Curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存")


def plot_model_comparison(all_fold_results, results_dir):
    """绘制模型比较图"""
    # 提取汇总结果
    summary_results = [r for r in all_fold_results if r.get('Fold') == 'MEAN']
    if not summary_results:
        return

    df = pd.DataFrame(summary_results)
    df = df.sort_values('PR_AUC', ascending=False).head(8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：PR AUC 排名
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
    bars = ax1.barh(range(len(df)), df['PR_AUC'].values, color=colors)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['Model'].values, fontsize=9)
    ax1.set_xlabel('PR AUC')
    ax1.set_title('Model Ranking by PR AUC')
    ax1.invert_yaxis()

    # 添加数值标签
    for i, (idx, row) in enumerate(df.iterrows()):
        ax1.text(row['PR_AUC'] + 0.01, i, f'{row["PR_AUC"]:.3f}', va='center', fontsize=8)

    # 右图：性能雷达图
    ax2 = axes[1]
    top3 = df.head(3)
    metrics = ['ROC_AUC', 'PR_AUC', 'F1', 'Recall', 'Precision']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in top3.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=row['Model'][:20])
        ax2.fill(angles, values, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Top 3 Models Performance')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'Model_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型比较图已保存")


# ====================================================
# 6. 主函数
# ====================================================
def main():
    """主函数"""

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("SAP 危险因素预测 - 多模型比较 + SHAP 分析")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    X, y, feature_names, data = load_data('DD1.csv')

    # 2. 定义模型
    print("\n[2] 定义模型...")
    models = get_models(X, y)

    # 3. 训练和评估
    print("\n[3] 开始训练和评估...")
    all_fold_results, all_roc_data, all_pr_data, trained_models = train_and_evaluate(
        X, y, models, feature_names, results_dir
    )

    # 4. 保存结果
    print("\n[4] 保存结果...")
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False, encoding='utf-8-sig')

    # 生成汇总表
    summary_df = results_df[results_df['Fold'] == 'MEAN'].copy()
    if not summary_df.empty:
        summary_df = summary_df.sort_values('PR_AUC', ascending=False)
        summary_df.to_csv(os.path.join(results_dir, 'model_summary.csv'), index=False, encoding='utf-8-sig')

        print("\n模型排名 (按 PR AUC):")
        print(summary_df[['Model', 'PR_AUC', 'ROC_AUC', 'F1', 'Recall', 'Precision']].head(10).to_string(index=False))

    # 5. 绘图
    print("\n[5] 绘制图表...")
    plot_roc_curves(all_roc_data, results_dir)
    plot_pr_curves(all_pr_data, results_dir, y)
    plot_model_comparison(all_fold_results, results_dir)

    # 6. SHAP 分析
    print("\n[6] SHAP 可解释性分析...")
    if SHAP_AVAILABLE and trained_models:
        # 选择最佳模型进行 SHAP 分析
        if not summary_df.empty:
            best_model_name = summary_df.iloc[0]['Model']
            best_model = trained_models.get(best_model_name)

            if best_model is not None:
                print(f"\n对最佳模型 '{best_model_name}' 进行 SHAP 分析")

                # 划分数据用于 SHAP
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # 重新训练最佳模型
                best_model.fit(X_train, y_train)

                # SHAP 分析
                perform_shap_analysis(best_model, X_train, X_test, feature_names,
                                      best_model_name, results_dir)
            else:
                print(f"最佳模型 {best_model_name} 未找到，跳过 SHAP 分析")
        else:
            print("没有成功的模型，跳过 SHAP 分析")
    else:
        if not SHAP_AVAILABLE:
            print("SHAP 未安装，跳过。安装命令: pip install shap")

    # 7. 完成
    print("\n" + "=" * 80)
    print(f"分析完成！所有结果保存在: {results_dir}")
    print("=" * 80)

    # 列出保存的文件
    print("\n生成的文件:")
    for f in sorted(os.listdir(results_dir)):
        f_path = os.path.join(results_dir, f)
        f_size = os.path.getsize(f_path) / 1024
        print(f"  - {f} ({f_size:.1f} KB)")


# ====================================================
# 运行主函数
# ====================================================
if __name__ == "__main__":
    main()