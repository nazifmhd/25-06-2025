"""
Model Monitoring and Drift Detection System
===========================================

This module implements advanced model monitoring capabilities including:
- Data drift detection
- Model performance monitoring
- Automated alerts and reporting
- MLOps integration
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from evidently.report import Report
    from evidently.metric_suite import DataDriftMetricSuite, DataQualityMetricSuite
    from evidently.test_suite import TestSuite
    from evidently.tests import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

class ModelMonitor:
    """Advanced model monitoring and drift detection"""
    
    def __init__(self, reference_data=None):
        self.reference_data = reference_data
        self.monitoring_history = []
        
    def detect_data_drift(self, current_data, reference_data=None):
        """Detect data drift using statistical methods"""
        if reference_data is None:
            reference_data = self.reference_data
            
        if reference_data is None:
            raise ValueError("Reference data must be provided")
        
        drift_results = {}
        
        # Get numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_data.columns:
                # Kolmogorov-Smirnov test
                from scipy.stats import ks_2samp
                ks_stat, p_value = ks_2samp(reference_data[col].dropna(), 
                                          current_data[col].dropna())
                
                drift_results[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05,
                    'drift_severity': 'High' if p_value < 0.01 else 'Medium' if p_value < 0.05 else 'Low'
                }
        
        return drift_results
    
    def generate_monitoring_report(self, current_data, model_predictions=None):
        """Generate comprehensive monitoring report"""
        if not EVIDENTLY_AVAILABLE:
            return self._simple_monitoring_report(current_data, model_predictions)
        
        try:
            # Data drift report
            data_drift_report = Report(metrics=[DataDriftMetricSuite()])
            data_drift_report.run(reference_data=self.reference_data, current_data=current_data)
            
            # Data quality report
            data_quality_report = Report(metrics=[DataQualityMetricSuite()])
            data_quality_report.run(reference_data=self.reference_data, current_data=current_data)
            
            # Save reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_drift_report.save_html(f"reports/data_drift_report_{timestamp}.html")
            data_quality_report.save_html(f"reports/data_quality_report_{timestamp}.html")
            
            return {
                'timestamp': timestamp,
                'drift_report_path': f"reports/data_drift_report_{timestamp}.html",
                'quality_report_path': f"reports/data_quality_report_{timestamp}.html"
            }
            
        except Exception as e:
            logging.warning(f"Could not generate Evidently reports: {e}")
            return self._simple_monitoring_report(current_data, model_predictions)
    
    def _simple_monitoring_report(self, current_data, model_predictions=None):
        """Simple monitoring report without Evidently"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'rows': len(current_data),
                'columns': len(current_data.columns),
                'missing_values': current_data.isnull().sum().to_dict(),
                'data_types': current_data.dtypes.astype(str).to_dict()
            }
        }
        
        if self.reference_data is not None:
            # Basic drift detection
            drift_results = self.detect_data_drift(current_data)
            report['drift_detection'] = drift_results
            
            # Summary statistics comparison
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            stats_comparison = {}
            
            for col in numeric_cols:
                if col in self.reference_data.columns:
                    stats_comparison[col] = {
                        'reference_mean': float(self.reference_data[col].mean()),
                        'current_mean': float(current_data[col].mean()),
                        'reference_std': float(self.reference_data[col].std()),
                        'current_std': float(current_data[col].std()),
                        'mean_drift': abs(current_data[col].mean() - self.reference_data[col].mean()) / self.reference_data[col].std()
                    }
            
            report['statistics_comparison'] = stats_comparison
        
        if model_predictions is not None:
            report['prediction_summary'] = {
                'total_predictions': len(model_predictions),
                'positive_predictions': int(np.sum(model_predictions == 1)),
                'negative_predictions': int(np.sum(model_predictions == 0)),
                'positive_rate': float(np.mean(model_predictions == 1))
            }
        
        return report
    
    def check_model_performance_degradation(self, current_accuracy, baseline_accuracy, threshold=0.05):
        """Check if model performance has degraded significantly"""
        performance_drop = baseline_accuracy - current_accuracy
        
        return {
            'performance_degradation': performance_drop > threshold,
            'performance_drop': performance_drop,
            'severity': 'Critical' if performance_drop > 0.1 else 'High' if performance_drop > threshold else 'Normal',
            'recommendation': 'Retrain model immediately' if performance_drop > 0.1 else 'Monitor closely' if performance_drop > threshold else 'Continue monitoring'
        }
    
    def automated_alerts(self, monitoring_results):
        """Generate automated alerts based on monitoring results"""
        alerts = []
        
        # Check for data drift
        if 'drift_detection' in monitoring_results:
            drift_cols = [col for col, result in monitoring_results['drift_detection'].items() 
                         if result['drift_detected']]
            if drift_cols:
                alerts.append({
                    'type': 'Data Drift',
                    'severity': 'High',
                    'message': f"Data drift detected in columns: {', '.join(drift_cols)}",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Check for missing values
        if 'data_summary' in monitoring_results:
            missing_cols = [col for col, count in monitoring_results['data_summary']['missing_values'].items() 
                           if count > 0]
            if missing_cols:
                alerts.append({
                    'type': 'Data Quality',
                    'severity': 'Medium',
                    'message': f"Missing values found in columns: {', '.join(missing_cols)}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts

class ExperimentTracker:
    """Track ML experiments and model versions"""
    
    def __init__(self, experiment_name="customer_purchase_prediction"):
        self.experiment_name = experiment_name
        self.experiments_file = "experiments/experiment_log.json"
        self.experiments = self._load_experiments()
    
    def _load_experiments(self):
        """Load existing experiments"""
        try:
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _save_experiments(self):
        """Save experiments to file"""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def log_experiment(self, model_name, parameters, metrics, notes=""):
        """Log a new experiment"""
        experiment = {
            'id': len(self.experiments) + 1,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'parameters': parameters,
            'metrics': metrics,
            'notes': notes
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
        
        return experiment['id']
    
    def get_best_experiment(self, metric='roc_auc'):
        """Get the best experiment based on a metric"""
        if not self.experiments:
            return None
        
        best_exp = max(self.experiments, 
                      key=lambda x: x['metrics'].get(metric, 0))
        return best_exp
    
    def compare_experiments(self, experiment_ids):
        """Compare multiple experiments"""
        experiments = [exp for exp in self.experiments if exp['id'] in experiment_ids]
        
        comparison = {
            'experiments': experiments,
            'metrics_comparison': {}
        }
        
        # Get all metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp['metrics'].keys())
        
        # Compare metrics
        for metric in all_metrics:
            comparison['metrics_comparison'][metric] = [
                exp['metrics'].get(metric, 0) for exp in experiments
            ]
        
        return comparison

class ABTestFramework:
    """A/B testing framework for model comparisons"""
    
    def __init__(self):
        self.tests = {}
    
    def create_test(self, test_name, model_a, model_b, traffic_split=0.5):
        """Create a new A/B test"""
        self.tests[test_name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': [],
            'created_at': datetime.now().isoformat()
        }
    
    def assign_user_to_test(self, test_name, user_id):
        """Assign a user to either model A or B"""
        if test_name not in self.tests:
            raise ValueError(f"Test {test_name} does not exist")
        
        # Simple hash-based assignment for consistency
        user_hash = hash(str(user_id)) % 100
        threshold = self.tests[test_name]['traffic_split'] * 100
        
        return 'A' if user_hash < threshold else 'B'
    
    def record_result(self, test_name, variant, outcome, user_id=None):
        """Record a test result"""
        if test_name not in self.tests:
            raise ValueError(f"Test {test_name} does not exist")
        
        result = {
            'outcome': outcome,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        
        if variant == 'A':
            self.tests[test_name]['results_a'].append(result)
        else:
            self.tests[test_name]['results_b'].append(result)
    
    def analyze_test(self, test_name):
        """Analyze A/B test results"""
        if test_name not in self.tests:
            raise ValueError(f"Test {test_name} does not exist")
        
        test = self.tests[test_name]
        results_a = test['results_a']
        results_b = test['results_b']
        
        if not results_a or not results_b:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate conversion rates
        conversions_a = sum(1 for r in results_a if r['outcome'] == 1)
        conversions_b = sum(1 for r in results_b if r['outcome'] == 1)
        
        conversion_rate_a = conversions_a / len(results_a)
        conversion_rate_b = conversions_b / len(results_b)
        
        # Simple statistical significance test (Chi-square)
        from scipy.stats import chi2_contingency
        
        contingency_table = [
            [conversions_a, len(results_a) - conversions_a],
            [conversions_b, len(results_b) - conversions_b]
        ]
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'model_a_conversion_rate': conversion_rate_a,
            'model_b_conversion_rate': conversion_rate_b,
            'lift': (conversion_rate_b - conversion_rate_a) / conversion_rate_a * 100,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b),
            'winner': 'B' if conversion_rate_b > conversion_rate_a else 'A'
        }

# Example usage and testing
if __name__ == "__main__":
    # Example monitoring workflow
    print("🔍 Model Monitoring and MLOps Demo")
    
    # Create sample reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'Age': np.random.normal(35, 10, 1000),
        'Salary': np.random.normal(50000, 15000, 1000),
        'Gender_Encoded': np.random.choice([0, 1], 1000)
    })
    
    # Create current data with slight drift
    current_data = pd.DataFrame({
        'Age': np.random.normal(37, 12, 500),  # Slight drift
        'Salary': np.random.normal(52000, 16000, 500),  # Slight drift
        'Gender_Encoded': np.random.choice([0, 1], 500)
    })
    
    # Initialize monitor
    monitor = ModelMonitor(reference_data)
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report(current_data)
    print("✅ Monitoring report generated")
    
    # Generate alerts
    alerts = monitor.automated_alerts(report)
    if alerts:
        print(f"⚠️ {len(alerts)} alerts generated")
        for alert in alerts:
            print(f"  - {alert['type']}: {alert['message']}")
    
    # Experiment tracking demo
    tracker = ExperimentTracker()
    
    # Log sample experiments
    exp_id = tracker.log_experiment(
        model_name="XGBoost",
        parameters={"n_estimators": 100, "max_depth": 6},
        metrics={"accuracy": 0.85, "roc_auc": 0.90},
        notes="Baseline model"
    )
    print(f"✅ Experiment {exp_id} logged")
    
    # A/B testing demo
    ab_test = ABTestFramework()
    ab_test.create_test("model_comparison", "xgboost", "lightgbm")
    
    # Simulate test assignment and results
    for user_id in range(100):
        variant = ab_test.assign_user_to_test("model_comparison", user_id)
        outcome = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% conversion rate
        ab_test.record_result("model_comparison", variant, outcome, user_id)
    
    # Analyze test
    analysis = ab_test.analyze_test("model_comparison")
    print(f"📊 A/B Test Results: Model {analysis['winner']} wins with {analysis['lift']:.1f}% lift")
    
    print("🎉 MLOps demo completed successfully!")
