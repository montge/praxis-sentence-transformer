"""
Visualization utilities for analysis results
"""

import os
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from ..logger import setup_logging

logger = setup_logging("visualization-utils", logging.DEBUG)

def plot_results(results_dict: Dict, paths: Dict[str, str], model_name: str, alpha: float):
    """
    Plot analysis results with enhanced visualization
    
    Parameters:
        results_dict (Dict): Dictionary containing threshold results
        paths (Dict[str, str]): Dictionary containing output paths
        model_name (str): Name of the model being analyzed
        alpha (float): Alpha value used for analysis
    """
    plt.figure(figsize=(12, 8))
    
    metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'balanced_accuracy']
    thresholds = list(results_dict.keys())
    
    for metric in metrics:
        values = [results_dict[t][metric] for t in thresholds]
        plt.plot(thresholds, values, marker='o', label=metric.replace('_', ' ').title())
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Metrics vs Threshold\nModel: {model_name}\nAlpha: {alpha}')
    plt.legend()
    
    save_path = os.path.join(
        paths['visualizations'], 
        f'threshold_analysis_alpha{alpha}.png'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path}") 

def plot_recall_comparison(all_results: Dict, paths: Dict[str, str], alpha: float):
    """
    Create comparative visualization of recall across models and thresholds
    
    Parameters:
        all_results (Dict): Dictionary containing results for all models
        paths (Dict[str, str]): Dictionary containing output paths
        alpha (float): Alpha value for the comparison
    """
    try:
        alpha = float(alpha)
        logger.debug(f"Starting plot_recall_comparison with alpha={alpha}")
        
        if not all_results:
            logger.error("No results data provided for comparison")
            return
        
        # Pre-validate data availability for this alpha
        valid_models = {}
        for model_name, model_results in all_results.items():
            if not model_results:
                logger.warning(f"Empty results for model {model_name}")
                continue
                
            try:
                alpha_results = model_results.get(alpha)
                if alpha_results and isinstance(alpha_results, dict):
                    valid_models[model_name] = alpha_results
                    logger.debug(f"Valid data found for {model_name} at alpha={alpha}")
                else:
                    logger.warning(f"No valid data for {model_name} at alpha={alpha}")
            except Exception as e:
                logger.error(f"Error processing {model_name} results: {str(e)}")
                continue
        
        if not valid_models:
            logger.warning(f"No valid models found for alpha={alpha}, skipping plot")
            return
            
        plt.figure(figsize=(15, 10))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        
        legend_handles = []
        legend_labels = []
        
        for (model_name, model_data), marker in zip(valid_models.items(), markers):
            try:
                model_display = str(model_name).split('/')[-1]
                recall = model_data.get('recall')
                precision = model_data.get('precision')
                f1_score = model_data.get('f1_score')
                
                if all(v is not None for v in [recall, precision, f1_score]):
                    scatter = plt.scatter(precision, recall, marker=marker, s=100)
                    legend_handles.append(scatter)
                    legend_labels.append(f"{model_display} (F1={f1_score:.3f})")
                else:
                    logger.warning(f"Missing metrics for {model_name} at alpha={alpha}")
            except Exception as e:
                logger.error(f"Error plotting {model_name}: {str(e)}")
                continue
        
        if not legend_handles:
            logger.error("No data was successfully plotted")
            return
            
        plt.grid(True, alpha=0.3)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(f'Precision-Recall Comparison Across Models\nAlpha: {alpha}')
        
        plt.legend(legend_handles, legend_labels,
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0.)
        
        save_path = os.path.join(paths['comparisons'], f'precision_recall_comparison_alpha{alpha:.2f}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved precision-recall comparison visualization to {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating recall comparison plot: {str(e)}")
        logger.exception("Detailed error trace:")
        plt.close()

def save_metrics_to_csv(results_dict: Dict, paths: Dict[str, str], alpha: float):
    """
    Save metrics to CSV file
    
    Parameters:
        results_dict (Dict): Dictionary containing threshold results
        paths (Dict[str, str]): Dictionary containing output paths
        alpha (float): Alpha value used for analysis
    """
    data = []
    for threshold, metrics in results_dict.items():
        row = {
            'threshold': threshold,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_negatives': metrics['false_negatives']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(paths['data'], f'metrics_alpha{alpha}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

def print_results_table(threshold_results: Dict):
    """
    Print formatted results table with additional metrics for requirement coverage and TNs
    
    Parameters:
        threshold_results (Dict): Dictionary containing threshold results
    """
    print("\nAnalysis Results:")
    print("{:<9} | {:<9} | {:<6} | {:<8} | {:<7} | {:<7} | {:<7} | {:<4} | {:<12}".format(
        "Threshold", "Precision", "Recall", "F1-Score", "FN Rate", "TN Rate", "TNs", "FPs", "Req Coverage"
    ))
    print("-" * 95)
    
    for threshold, metrics in sorted(threshold_results.items()):
        try:
            req_coverage = 1 - metrics.get('false_negative_rate', 0)
            tn = metrics.get('true_negatives', 0)
            fp = metrics.get('false_positives', 0)
            tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print("{:<9.2f} | {:<9.3f} | {:<6.3f} | {:<8.3f} | {:<7.3f} | {:<7.3f} | {:<7d} | {:<4d} | {:<12.3f}".format(
                threshold,
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('false_negative_rate', 0),
                tn_rate,
                tn,
                fp,
                req_coverage
            ))

            # Coverage threshold message
            min_coverage = float(os.getenv('MIN_COVERAGE_THRESHOLD', '0.95'))
            if req_coverage >= min_coverage:
                print("\n*** Met {:.0%} coverage threshold ***".format(min_coverage))
                print("At threshold {:.2f}:".format(threshold))
                print("- {:.1%} of true requirements are found".format(req_coverage))
                print("- {:.1%} of true non-requirements are correctly identified".format(tn_rate))
                print("- True Negatives: {} out of {} possible".format(tn, tn + fp))
                print("- False Positives: {}".format(fp))
                if tn > 0:
                    print("- For every correct non-requirement, we have {:.2f} false positives\n".format(fp/tn))
                else:
                    print("- No true negatives found in this threshold\n")
                    
        except Exception as e:
            logger.error("Error processing results for threshold {}: {}".format(threshold, str(e)))
            continue

def save_model_summary(model_name: str, model_results: Dict, paths: Dict) -> None:
    """Save summary of model performance"""
    try:
        # Find best threshold based on F1 score
        thresholds = list(model_results.values())[0].keys()  # Get thresholds from first alpha value
        best_threshold = max(
            thresholds,
            key=lambda x: list(model_results.values())[0][x]['metrics']['f1_score']
        )
        
        # Get results for best threshold
        best_results = list(model_results.values())[0][best_threshold]
        
        summary = {
            'model_name': model_name,
            'best_threshold': best_threshold,
            'metrics': best_results['metrics'],
            'counts': best_results['counts']
        }
        
        # Save summary to file
        summary_file = os.path.join(paths['results'], 'model_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Model summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving model summary: {str(e)}")
        logger.exception("Detailed error trace:")

def save_final_comparison(all_results: Dict, paths: Dict[str, str], min_coverage: float):
    """
    Save final comparison of all models and configurations
    
    Parameters:
        all_results (Dict): Dictionary containing results for all models
        paths (Dict[str, str]): Dictionary containing output paths
        min_coverage (float): Minimum coverage threshold
    """
    try:
        os.makedirs(paths['results'], exist_ok=True)
        comparison_file = os.path.join(paths['results'], 'final_comparison.txt')
        
        logger.debug(f"Input all_results structure: {json.dumps(all_results, indent=2)}")
        
        with open(comparison_file, 'w') as f:
            f.write("Final Analysis Comparison\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Minimum Coverage Threshold: {min_coverage:.2%}\n\n")
            
            headers = [
                "Model", "Alpha", "Threshold", "Precision", "Recall", "F1", 
                "Accuracy", "Coverage", "TP", "FP", "FN", "TN"
            ]
            f.write(" | ".join(f"{h:<12}" for h in headers) + "\n")
            f.write("-" * (13 * len(headers)) + "\n")
            
            for model_name, model_results in all_results.items():
                model_display = model_name.split('/')[-1]
                logger.debug(f"Processing model: {model_display}")
                
                for alpha, alpha_results in model_results.items():
                    if not alpha_results:
                        logger.warning(f"No results found for {model_display} with alpha={alpha}")
                        continue
                        
                    for threshold, metrics in alpha_results.items():
                        try:
                            precision = metrics.get('precision', 0.0)
                            recall = metrics.get('recall', 0.0)
                            f1_score = metrics.get('f1_score', 0.0)
                            accuracy = metrics.get('accuracy', 0.0)
                            
                            fn_rate = metrics.get('false_negative_rate', 0.0)
                            coverage = 1.0 - fn_rate
                            
                            tp = metrics.get('true_positives', 0)
                            fp = metrics.get('false_positives', 0)
                            fn = metrics.get('false_negatives', 0)
                            tn = metrics.get('true_negatives', 0)
                            
                            if coverage >= min_coverage:
                                result_line = " | ".join([
                                    f"{model_display[:12]:<12}",
                                    f"{alpha:<12.2f}",
                                    f"{threshold:<12.2f}",
                                    f"{precision:<12.3f}",
                                    f"{recall:<12.3f}",
                                    f"{f1_score:<12.3f}",
                                    f"{accuracy:<12.3f}",
                                    f"{coverage:<12.3%}",
                                    f"{tp:<12d}",
                                    f"{fp:<12d}",
                                    f"{fn:<12d}",
                                    f"{tn:<12d}"
                                ])
                                f.write(result_line + "\n")
                            else:
                                logger.debug(f"Skipping results below coverage threshold: {coverage:.2%} < {min_coverage:.2%}")
                                
                        except Exception as e:
                            logger.error(f"Error processing threshold {threshold} results: {str(e)}")
                            logger.debug(f"Problematic metrics: {metrics}")
                            continue
                    
                    f.write("-" * (13 * len(headers)) + "\n")
            
            f.write("\nSummary:\n")
            f.write(f"Total models processed: {len(all_results)}\n")
            f.write(f"Coverage threshold: {min_coverage:.2%}\n")
            
        logger.info(f"Saved final comparison to {comparison_file}")
        
    except Exception as e:
        logger.error(f"Error in save_final_comparison: {str(e)}")
        logger.exception("Detailed error trace:")
        raise