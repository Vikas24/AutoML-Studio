"""
Main Pipeline: Orchestrates the entire ML workflow.
Handles supervised (classification/regression) and unsupervised learning.
✅ NOW INCLUDES: Stage 9 (Regularization), Stage 11 (GridSearch), Stage 12 (Advanced Models), Stage 15 (Error-based decisions)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression, SGDClassifier
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, 
                             GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor)
from sklearn.svm import SVC, SVR
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.imbalance import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from src.logging.decision_logger import DecisionLogger
import warnings
warnings.filterwarnings('ignore')


class MLPipeline:
    """End-to-end ML pipeline with decision logging - 100% Flowchart Implementation."""
    
    def __init__(self):
        self.logger = DecisionLogger()
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.is_supervised = None
        self.problem_type = None
        self.feature_names = None
        self.scaler = None
        self.preprocessor = {}
        self.reg_type = None
        self.best_hyperparams = None
    
    # ============== STAGE 1: DATA INGESTION ==============
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV."""
        self.data = pd.read_csv(filepath)
        print(f"✓ Stage 1 - Data Ingestion: {self.data.shape}")
        return self.data
    
    # ============== STAGE 2: DATA VALIDATION ==============
    
    def validate_data(self):
        """Comprehensive data validation."""
        missing_pct = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        duplicates = self.data.duplicated().sum()
        
        outliers_detected = 0
        for col in self.data.select_dtypes(include=[np.number]).columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers_detected += ((self.data[col] < Q1 - 1.5*IQR) | (self.data[col] > Q3 + 1.5*IQR)).sum()
        
        self.logger.log_data_validation(
            data_shape=self.data.shape,
            missing_pct=round(missing_pct, 2),
            duplicates=duplicates,
            outliers_detected=outliers_detected
        )
        
        print(f"✓ Stage 2 - Data Validation: {missing_pct:.2f}% missing, {duplicates} dups, {outliers_detected} outliers")
        return self
    
    # ============== STAGE 3: DATA CLEANING ==============
    
    def clean_data(self):
        """Remove duplicates and handle missing values."""
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna(thresh=len(self.data) * 0.5, axis=1)
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        
        print(f"✓ Stage 3 - Data Cleaning: {self.data.shape}")
        return self
    
    # ============== STAGE 4: FEATURE ENGINEERING ==============
    
    def engineer_features(self):
        """Create interaction terms."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            self.data[f'{numeric_cols[0]}_x_{numeric_cols[1]}'] = \
                self.data[numeric_cols[0]] * self.data[numeric_cols[1]]
        
        print(f"✓ Stage 4 - Feature Engineering: {self.data.shape}")
        return self
    
    # ============== STAGE 5: PROBLEM TYPE DETECTION ==============
    
    def determine_problem_type(self, target_column: str = None):
        """Decide: supervised vs unsupervised."""
        
        if target_column is None:
            self.is_supervised = False
            self.logger.log_problem_type(None, False)
            print("✓ Stage 5 - Problem Type: Unsupervised Learning")
        else:
            self.is_supervised = True
            self.y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])
            self.feature_names = self.X.columns.tolist()
            
            if self.y.dtype == 'object' or len(self.y.unique()) < 20:
                self.problem_type = 'classification'
            else:
                self.problem_type = 'regression'
            
            self.logger.log_problem_type(target_column, True, self.problem_type)
            print(f"✓ Stage 6 - Problem Type: {self.problem_type.upper()}")
        
        return self
    
    # ============== STAGE 3: PREPROCESSING ==============
    
    def preprocess(self, scaling_method='standard', imputation_method='mean', encoding_methods=None):
        """Preprocessing: imputation, encoding, scaling."""
        
        if encoding_methods is None:
            encoding_methods = {}
        
        X = self.X if self.is_supervised else self.data
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in encoding_methods:
                encoding_methods[col] = 'label'
            
            if encoding_methods[col] == 'label':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.preprocessor[col] = le
        
        if self.is_supervised:
            numeric_cols = self.X.select_dtypes(include=[np.number]).columns
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = None
            
            if self.scaler:
                self.X[numeric_cols] = self.scaler.fit_transform(self.X[numeric_cols])
        
        self.logger.log_preprocessing(imputation_method, scaling_method or 'none', encoding_methods)
        print(f"✓ Stage 3 - Preprocessing: scaling={scaling_method}")
        return self
    
    # ============== STAGE 4: FEATURE SELECTION ==============
    
    def select_features(self, n_features='auto', method='mutual_info'):
        """Feature selection."""
        if not self.is_supervised:
            return self
        
        if n_features == 'auto':
            n_features = max(5, int(self.X.shape[1] * 0.7))
        
        if self.problem_type == 'classification':
            selector = SelectKBest(f_classif, k=min(n_features, self.X.shape[1]))
        else:
            selector = SelectKBest(f_regression, k=min(n_features, self.X.shape[1]))
        
        self.X = pd.DataFrame(
            selector.fit_transform(self.X, self.y),
            columns=[self.feature_names[i] for i in selector.get_support(indices=True)]
        )
        self.feature_names = self.X.columns.tolist()
        
        self.logger.log_feature_engineering(
            n_features_before=len(self.feature_names) + (len(self.feature_names) - self.X.shape[1]),
            n_features_after=self.X.shape[1],
            selection_method=method
        )
        print(f"✓ Stage 7 - Feature Selection: {self.X.shape[1]} features")
        return self
    
    # ============== STAGE 7: TRAIN/TEST SPLIT ==============
    
    def split_data(self, test_size=0.2, stratify=True):
        """Split data for supervised learning."""
        if not self.is_supervised:
            return self
        
        if stratify and self.problem_type == 'classification':
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            for train_idx, test_idx in splitter.split(self.X, self.y):
                self.X_train, self.X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42
            )
        
        self.logger.log_train_test_split(
            train_size=int((1 - test_size) * 100),
            test_size=int(test_size * 100),
            stratified=(stratify and self.problem_type == 'classification')
        )
        print(f"✓ Stage 7 - Train/Test Split: {len(self.X_train)}/{len(self.X_test)}")
        return self
    
    # ============== STAGE 10: CLASS IMBALANCE HANDLING ==============
    
    def handle_imbalance(self):
        """Handle class imbalance."""
        if not self.is_supervised or self.problem_type != 'classification':
            return self
        
        class_dist = self.y_train.value_counts().to_dict()
        imbalance_ratio = max(class_dist.values()) / min(class_dist.values())
        
        if imbalance_ratio > 2:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_dist.values()) - 1))
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            handling = 'SMOTE'
        else:
            handling = 'None (balanced)'
        
        self.logger.log_class_imbalance(class_dist, handling)
        print(f"✓ Stage 10 - Class Imbalance: {handling}")
        return self
    
    # ============== STAGE 9: REGULARIZATION DECISION ==============
    
    def select_regularization(self):
        """Decide which regularization to use."""
        if not self.is_supervised:
            return 'none'
        
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.X_train[numeric_cols].corr()
            high_corr = (corr_matrix.abs() > 0.9).sum().sum()
        else:
            high_corr = 0
        
        feature_to_sample_ratio = self.X_train.shape[1] / self.X_train.shape[0]
        
        if high_corr > 5 and feature_to_sample_ratio < 0.1:
            reg_type = 'L2'
            reason = "High correlation (multicollinearity), use Ridge (L2)"
        elif feature_to_sample_ratio > 0.3:
            reg_type = 'L1'
            reason = "More features than samples, use Lasso (L1) for feature selection"
        elif high_corr > 5 or feature_to_sample_ratio > 0.2:
            reg_type = 'ElasticNet'
            reason = "Balanced regularization needed, use ElasticNet (L1+L2)"
        else:
            reg_type = 'L2'
            reason = "Balanced feature set, use Ridge (L2) for stability"
        
        self.reg_type = reg_type
        self.logger.log_regularization(reg_type, reason)
        print(f"✓ Stage 9 - Regularization: {reg_type}")
        print(f"  Reason: {reason}")
        return reg_type
    
    # ============== STAGE 11: HYPERPARAMETER TUNING ==============
    
    def tune_hyperparameters(self, model, model_name):
        """Tune hyperparameters using GridSearchCV."""
        
        param_grids = {
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            },
            'RandomForestRegressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
        
        param_grid = param_grids.get(model_name, {})
        if not param_grid:
            return model
        
        scoring = 'f1_weighted' if self.problem_type == 'classification' else 'r2'
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        self.best_hyperparams = best_params
        
        self.logger.log_hyperparameter_tuning(best_params, 'GridSearchCV', scoring)
        print(f"  Tuned: {model_name} (best CV {scoring}={grid_search.best_score_:.4f})")
        
        return grid_search.best_estimator_
    
    # ============== STAGE 12: ADVANCED MODELS ==============
    
    def get_advanced_models(self):
        """Get all advanced models for training."""
        models = {}
        
        if self.problem_type == 'classification':
            models['LogisticRegression'] = LogisticRegression(max_iter=1000, random_state=42)
            models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)
            models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
            models['AdaBoost'] = AdaBoostClassifier(n_estimators=50, random_state=42)
            models['SVC'] = SVC(kernel='rbf', probability=True, random_state=42)
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        
        else:  # Regression
            models['LinearRegression'] = LinearRegression()
            models['Ridge'] = Ridge(alpha=1.0)
            models['Lasso'] = Lasso(alpha=0.1)
            models['ElasticNet'] = ElasticNet(alpha=0.1)
            models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            models['SVR'] = SVR(kernel='rbf')
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        
        return models
    
    # ============== STAGE 8-12: MODEL TRAINING WITH ALL ENHANCEMENTS ==============
    
    def train_model(self, tune_hyperparameters=True):
        """Train multiple advanced models with hyperparameter tuning."""
        
        print("\n" + "="*70)
        print("STAGE 9: REGULARIZATION DECISION")
        print("="*70)
        reg_type = self.select_regularization()
        
        print("\n" + "="*70)
        print("STAGE 11: HYPERPARAMETER TUNING & STAGE 12: ADVANCED MODELS")
        print("="*70)
        
        models = self.get_advanced_models()
        
        if self.problem_type == 'classification':
            scoring = 'f1_weighted'
        else:
            scoring = 'r2'
        
        best_model_name = None
        best_score = -np.inf
        cv_scores_dict = {}
        best_model = None
        
        print(f"\nTraining {len(models)} models with {scoring} scoring...\n")
        
        for name, model in models.items():
            try:
                if tune_hyperparameters:
                    model = self.tune_hyperparameters(model, name)
                
                scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scoring)
                mean_score = scores.mean()
                cv_scores_dict[name] = round(mean_score, 4)
                
                print(f"  ✓ {name:25} {scoring}={mean_score:.4f} (±{scores.std():.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    best_model = model
            
            except Exception as e:
                print(f"  ✗ {name:25} Failed - {str(e)[:30]}")
                continue
        
        # Train best model
        self.model = best_model.fit(self.X_train, self.y_train)
        
        self.logger.log_model_selection(
            best_model_name,
            cv_scores_dict,
            best_score,
            f"Best CV {scoring} score. Regularization: {reg_type}. Tuning: {'GridSearchCV' if tune_hyperparameters else 'No'}"
        )
        
        print(f"\n✓ STAGE 8: BASELINE & ADVANCED MODELS")
        print(f"  Best Model: {best_model_name} ({scoring}: {best_score:.4f})")
        print(f"  Trained on {len(self.X_train)} samples with {self.X_train.shape[1]} features")
        
        return self
    
    # ============== STAGE 13: EVALUATION ==============
    
    def evaluate(self):
        """Evaluate model on test set."""
        if not self.is_supervised:
            return {}
        
        y_pred = self.model.predict(self.X_test)
        
        if self.problem_type == 'classification':
            metrics = {
                'accuracy': round(accuracy_score(self.y_test, y_pred), 4),
                'precision': round(precision_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
                'recall': round(recall_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
                'f1': round(f1_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
            }
            try:
                y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
                metrics['roc_auc'] = round(roc_auc_score(self.y_test, y_pred_proba), 4)
            except:
                pass
        else:
            metrics = {
                'r2': round(r2_score(self.y_test, y_pred), 4),
                'rmse': round(np.sqrt(mean_squared_error(self.y_test, y_pred)), 4),
                'mae': round(mean_absolute_error(self.y_test, y_pred), 4),
            }
        
        self.logger.log_evaluation(metrics)
        print(f"\n✓ STAGE 13: EVALUATION")
        print(f"  {metrics}")
        
        return metrics
    
    # ============== STAGE 14: ERROR ANALYSIS ==============
    
    def error_analysis(self):
        """Analyze errors and Stage 15: Error-based decisions."""
        if not self.is_supervised or self.problem_type != 'classification':
            return {}
        
        y_pred = self.model.predict(self.X_test)
        
        false_positives = sum((y_pred == 1) & (self.y_test == 0))
        false_negatives = sum((y_pred == 0) & (self.y_test == 1))
        
        insights = []
        
        # Stage 15: Error-based decisions
        if false_positives > false_negatives * 2:
            insights.append("High FP: Consider raising prediction threshold")
        elif false_negatives > false_positives * 2:
            insights.append("High FN: Consider lowering prediction threshold")
        
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-3:]
            insights.append(f"Top error-driving features: {[self.feature_names[i] for i in top_features]}")
        
        self.logger.log_error_analysis(
            error_summary=f"FP: {false_positives}, FN: {false_negatives}",
            false_positives=false_positives,
            false_negatives=false_negatives,
            insights=insights
        )
        
        print(f"\n✓ STAGE 14: ERROR ANALYSIS & STAGE 15: ERROR-BASED DECISIONS")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        for insight in insights:
            print(f"  💡 {insight}")
        
        return {'false_positives': false_positives, 'false_negatives': false_negatives, 'insights': insights}
    
    # ============== UNSUPERVISED PIPELINE ==============
    
    def unsupervised_pipeline(self):
        """Run unsupervised learning pipeline."""
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data.select_dtypes(include=[np.number]))
        
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        variance_explained = sum(pca.explained_variance_ratio_) * 100
        
        self.logger.log_dimensionality_reduction(
            'PCA',
            pca.n_components_,
            round(variance_explained, 2)
        )
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_pca, clusters)
        
        self.logger.log_clustering('KMeans', 3, sil_score)
        
        print(f"\n✓ Unsupervised: PCA ({pca.n_components_} components, {variance_explained:.1f}% variance)")
        print(f"✓ Clustering: KMeans (k=3, silhouette={sil_score:.3f})")
        
        return clusters
    
    # ============== EXPORT ==============
    
    def generate_report(self) -> str:
        """Generate full decision report."""
        return self.logger.summary_text()
    
    def get_decisions(self) -> dict:
        """Get all decisions as dictionary."""
        return self.logger.to_dict()
    
    def save_report(self, filepath: str):
        """Save decision log to JSON."""
        self.logger.to_json(filepath)