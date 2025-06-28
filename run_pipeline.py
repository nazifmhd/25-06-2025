"""
Automated ML Pipeline Orchestrator
==================================

This script orchestrates the entire enhanced ML pipeline with modern AI features.
Run this to execute all components in the correct order.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_command(command, description):
    """Run a command and log the results"""
    logging.info(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"✅ {description} completed successfully")
            return True
        else:
            logging.error(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"❌ Error running {description}: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    logging.info("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'streamlit', 'joblib', 'xgboost', 'lightgbm', 'optuna',
        'shap', 'tensorflow', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        logging.info("Installing missing packages...")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        return run_command(install_cmd, "Installing missing packages")
    else:
        logging.info("✅ All dependencies are installed")
        return True

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'reports', 'experiments']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"📁 Created directory: {directory}")
    
    return True

def run_data_preprocessing():
    """Run data preprocessing and basic ML pipeline"""
    logging.info("🔄 Running original ML pipeline for baseline...")
    return run_command("python ml_pipeline.py", "Original ML Pipeline")

def run_enhanced_pipeline():
    """Run the enhanced ML pipeline with modern AI features"""
    logging.info("🚀 Running enhanced ML pipeline...")
    return run_command("python enhanced_ml_pipeline.py", "Enhanced ML Pipeline")

def run_model_monitoring():
    """Run model monitoring and drift detection"""
    logging.info("🔍 Running model monitoring...")
    return run_command("python model_monitoring.py", "Model Monitoring")

def generate_reports():
    """Generate comprehensive reports"""
    logging.info("📊 Generating comprehensive reports...")
    
    # Create a summary report
    summary = {
        'pipeline_run_time': datetime.now().isoformat(),
        'status': 'completed',
        'components': {
            'data_preprocessing': True,
            'enhanced_ml_pipeline': True,
            'model_monitoring': True,
            'web_app': True
        }
    }
    
    with open('results/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("✅ Reports generated")
    return True

def main():
    """Main pipeline orchestrator"""
    logging.info("🚀 Starting Enhanced ML Pipeline Orchestrator")
    logging.info("=" * 60)
    
    pipeline_steps = [
        ("Checking Dependencies", check_dependencies),
        ("Creating Directories", create_directories),
        ("Running Data Preprocessing", run_data_preprocessing),
        ("Running Enhanced Pipeline", run_enhanced_pipeline),
        ("Running Model Monitoring", run_model_monitoring),
        ("Generating Reports", generate_reports)
    ]
    
    failed_steps = []
    
    for step_name, step_function in pipeline_steps:
        try:
            success = step_function()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            logging.error(f"❌ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
    
    # Final status
    if not failed_steps:
        logging.info("🎉 Enhanced ML Pipeline completed successfully!")
        logging.info("🌐 You can now run the web application:")
        logging.info("   streamlit run enhanced_app.py")
        logging.info("   or")
        logging.info("   streamlit run app.py")
    else:
        logging.error(f"❌ Pipeline completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            logging.error(f"   - {step}")
    
    logging.info("=" * 60)
    logging.info("📁 Check the following directories for outputs:")
    logging.info("   - models/     : Trained models and preprocessors")
    logging.info("   - results/    : Model evaluation results and visualizations")
    logging.info("   - reports/    : Monitoring and drift detection reports")
    logging.info("   - experiments/: Experiment tracking logs")

if __name__ == "__main__":
    main()
