# ML dashboard - Streamlit Demo
# launch with `uv run streamlit run unit6b_streamlit_extra.py`

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Title
st.markdown("<h2 style='text-align: center; color: #2e86ab;'>🤖 Interactive ML Model Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Compare different models and visualize their performance in real-time</p>", unsafe_allow_html=True)

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create tabs for configuration
tab1, tab2, tab3 = st.tabs(["Data", "Model", "Visualization"])

# === DATA TAB ===
with tab1:
    st.markdown("#### Dataset Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dataset_selector = st.selectbox(
            "Dataset",
            options=['Iris Dataset', 'Synthetic Data'],
            index=0
        )
    
    with col2:
        test_size_slider = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05
        )
    
    with col3:
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            step=1
        )

# === MODEL TAB ===
with tab2:
    st.markdown("#### Model Configuration")
    
    model_selector = st.selectbox(
        "Model",
        options=['Random Forest', 'Logistic Regression', 'SVM'],
        index=0
    )
    
    st.markdown("---")
    
    # Show parameters based on selected model
    if model_selector == 'Random Forest':
        st.markdown("**Random Forest Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            rf_n_estimators = st.slider(
                "N Estimators",
                min_value=10,
                max_value=300,
                value=100,
                step=10
            )
        with col2:
            rf_max_depth = st.slider(
                "Max Depth",
                min_value=1,
                max_value=15,
                value=5,
                step=1
            )
    
    elif model_selector == 'Logistic Regression':
        st.markdown("**Logistic Regression Parameters**")
        lr_C = st.slider(
            "C Parameter",
            min_value=0.001,
            max_value=1000.0,
            value=1.0,
            step=0.001,
            format="%.3f"
        )
    
    else:  # SVM
        st.markdown("**SVM Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            svm_C = st.slider(
                "C Parameter",
                min_value=0.001,
                max_value=1000.0,
                value=1.0,
                step=0.001,
                format="%.3f"
            )
        with col2:
            svm_kernel = st.selectbox(
                "Kernel",
                options=['rbf', 'linear', 'poly'],
                index=0
            )

# === VISUALIZATION TAB ===
with tab3:
    st.markdown("#### Visualization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_x = st.selectbox(
            "X Feature",
            options=feature_names,
            index=0
        )
    
    with col2:
        feature_y = st.selectbox(
            "Y Feature",
            options=feature_names,
            index=1
        )
    
    with col3:
        show_decision_boundary = st.checkbox(
            "Show Decision Boundary",
            value=True
        )

# === MAIN TRAINING AND VISUALIZATION ===
st.markdown("---")

# Status placeholder
status_placeholder = st.empty()
status_placeholder.info("**Status:** Ready to train model")

try:
    status_placeholder.warning("**Status:** Preparing data...")
    
    # Prepare data
    if dataset_selector == 'Iris Dataset':
        X_data, y_data = X, y
    else:
        X_data, y_data = make_classification(
            n_samples=300, n_features=4, n_classes=3,
            n_informative=4, n_redundant=0,
            random_state=random_state
        )
    
    # Select features for visualization
    X_selected = X_data[:, [feature_names.index(feature_x), 
                            feature_names.index(feature_y)]]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_data,
        test_size=test_size_slider,
        random_state=random_state
    )
    
    # Configure model
    if model_selector == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state
        )
    elif model_selector == 'Logistic Regression':
        model = LogisticRegression(
            C=lr_C,
            random_state=random_state,
            max_iter=1000
        )
    else:  # SVM
        model = SVC(
            C=svm_C,
            kernel=svm_kernel,
            random_state=random_state
        )
    
    status_placeholder.warning("**Status:** Training model...")
    
    # Train model
    model.fit(X_train, y_train)
    
    status_placeholder.warning("**Status:** Model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    status_placeholder.success("**Status:** Training completed successfully!")
    
    # Create result tabs
    results_tab1, results_tab2 = st.tabs(["Visualizations", "Metrics"])
    
    # === VISUALIZATIONS TAB ===
    with results_tab1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Data distribution
        scatter = axes[0].scatter(
            X_selected[:, 0], X_selected[:, 1],
            c=y_data, cmap='viridis', alpha=0.7, s=50
        )
        axes[0].set_xlabel(feature_x)
        axes[0].set_ylabel(feature_y)
        axes[0].set_title('Full Dataset Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Decision boundary
        if show_decision_boundary:
            h = 0.02
            x_min, x_max = X_selected[:, 0].min() - 0.5, X_selected[:, 0].max() + 0.5
            y_min, y_max = X_selected[:, 1].min() - 0.5, X_selected[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        
        # Scatter test points
        axes[1].scatter(
            X_test[:, 0], X_test[:, 1],
            c=y_test, cmap='viridis',
            edgecolors='black', s=50, alpha=0.8
        )
        axes[1].set_xlabel(feature_x)
        axes[1].set_ylabel(feature_y)
        axes[1].set_title(f'Test Set & Decision Boundary\n(Accuracy: {accuracy:.3f})')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
        axes[2].set_title('Confusion Matrix')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # === METRICS TAB ===
    with results_tab2:
        st.markdown(f"**Model:** {model_selector}")
        st.markdown(f"**Dataset:** {dataset_selector}")
        st.markdown(f"**Features:** {feature_x} vs {feature_y}")
        st.markdown(f"**Test Accuracy:** {accuracy:.3f}")
        st.markdown(f"**Training Size:** {len(X_train)} samples")
        st.markdown(f"**Test Size:** {len(X_test)} samples")
        
        st.markdown("### Classification Report")
        if dataset_selector == 'Iris Dataset':
            report = classification_report(y_test, y_pred, target_names=target_names)
        else:
            report = classification_report(y_test, y_pred)
        
        st.text(report)

except Exception as e:
    status_placeholder.error(f"**Status:** Error: {str(e)}")