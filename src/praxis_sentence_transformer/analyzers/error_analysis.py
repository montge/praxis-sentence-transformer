import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_curve
from typing import Dict, Tuple, List
from dataclasses import dataclass
import os
from datetime import datetime
import json
from pathlib import Path
import logging
from logger import get_logger, handle_exception

logger = get_logger(__name__)

# Get the current file's directory and construct relative path
CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR.parent / "results" / "error_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Saving results to: {RESULTS_DIR}")

@dataclass
class ProcessingStageResults:
    """Store and calculate metrics for each processing stage"""
    tp: int  # True Positives
    fp: int  # False Positives
    tn: int  # True Negatives
    fn: int  # False Negatives
    name: str  # Stage identifier

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    @property
    def fnr(self) -> float:
        """Calculate False Negative Rate"""
        return self.fn / (self.fn + self.tp)

    @property
    def fpr(self) -> float:
        """Calculate False Positive Rate"""
        return self.fp / (self.fp + self.tn)

@dataclass
class CostAnalysis:
    """
    Analyze costs associated with different processing approaches.
    
    Args:
        review_cost_per_hour: Hourly cost for review work
        engineering_cost_per_hour: Hourly cost for engineering work
        rework_hours: Hours needed to fix a missed requirement
        integration_cost: Additional integration cost per missed requirement
        delay_cost_per_day: Cost of schedule delay per day
        typical_delay_days: Typical delay days per missed requirement
        quick_review_minutes: Time for quick review of likely non-matches
        detailed_review_hours: Time for detailed review of potential matches
        fp_review_hours: Hours needed to review false positives
        fp_overhead_cost: Additional overhead cost per false positive
    """
    review_cost_per_hour: float
    engineering_cost_per_hour: float
    rework_hours: float
    integration_cost: float
    delay_cost_per_day: float
    typical_delay_days: float
    quick_review_minutes: float
    detailed_review_hours: float
    fp_review_hours: float  # Added: Time to review false positives
    fp_overhead_cost: float  # Added: Overhead cost per false positive
    
    def calculate_impact_cost(self, missed_requirements: int, false_positives: int) -> float:
        """
        Calculate total impact cost including false positive overhead
        
        Args:
            missed_requirements: Number of false negatives
            false_positives: Number of false positives
        """
        # Original impact costs from missed requirements
        rework_cost = (self.engineering_cost_per_hour * self.rework_hours * missed_requirements)
        delay_impact = (self.delay_cost_per_day * self.typical_delay_days * missed_requirements)
        integration_impact = (self.integration_cost * missed_requirements)
        
        # Additional costs from false positives
        fp_review_cost = (self.review_cost_per_hour * self.fp_review_hours * false_positives)
        fp_overhead = (self.fp_overhead_cost * false_positives)
        
        return rework_cost + delay_impact + integration_impact + fp_review_cost + fp_overhead

class ProcessingApproach:
    """
    Represents a specific processing approach with its metrics and costs.
    
    Args:
        name: Name of the processing approach
        true_positives: Number of true positive results
        false_positives: Number of false positive results
        true_negatives: Number of true negative results
        false_negatives: Number of false negative results
        processing_time_hours: Time required for processing
        cost_per_hour: Hourly cost of processing
        requires_exhaustive_review: Whether approach requires reviewing all pairs
    """
    def __init__(self, 
                 name: str,
                 true_positives: int,
                 false_positives: int,
                 true_negatives: int,
                 false_negatives: int,
                 processing_time_hours: float,
                 cost_per_hour: float,
                 requires_exhaustive_review: bool):
        self.name = name
        self.tp = true_positives
        self.fp = false_positives
        self.tn = true_negatives
        self.fn = false_negatives
        self.processing_time = processing_time_hours
        self.cost_per_hour = cost_per_hour
        self.requires_exhaustive_review = requires_exhaustive_review
        
    def calculate_processing_cost(self, total_requirements: int) -> float:
        """Calculate total processing cost for this approach"""
        return self.processing_time * self.cost_per_hour * total_requirements

class MultiStageLLMAnalysis:
    def __init__(self, 
                 tfidf_results: Dict[str, int],
                 claude21_results: Dict[str, int],
                 claude35_results: Dict[str, int],
                 model_info: Dict[str, str] = None):
        """
        Initialize analysis with results from all processing stages.
        
        Args:
            tfidf_results: Results from TF-IDF + Transformer stage
            claude21_results: Results from Claude 2.1
            claude35_results: Results from Claude 3.5
            model_info: Dictionary containing model details and versions
        """
        self.model_info = model_info or {
            'tfidf_transformer': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'claude21': 'anthropic/claude-2.1',
            'claude35': 'anthropic/claude-3.5-sonnet'
        }
        
        # Initialize each stage with its results
        self.stages = {
            'tfidf': ProcessingStageResults(**tfidf_results, 
                name=f'TF-IDF + {self.model_info["tfidf_transformer"].split("/")[-1]}'),
            'claude21': ProcessingStageResults(**claude21_results, 
                name=self.model_info['claude21']),
            'claude35': ProcessingStageResults(**claude35_results, 
                name=self.model_info['claude35'])
        }
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.calculate_combined_metrics()

    def save_results(self):
        """Save analysis results to JSON file"""
        results = {
            'timestamp': self.timestamp,
            'model_info': self.model_info,
            'metrics': {
                stage_name: {
                    'precision': stage.precision,
                    'recall': stage.recall,
                    'fnr': stage.fnr,
                    'fpr': stage.fpr,
                    'raw_counts': {
                        'tp': stage.tp,
                        'fp': stage.fp,
                        'tn': stage.tn,
                        'fn': stage.fn
                    }
                }
                for stage_name, stage in self.stages.items()
            },
            'combined_metrics': {
                'fnr_combined_21': self.fnr_combined_21,
                'fnr_combined_35': self.fnr_combined_35
            }
        }
        
        output_file = RESULTS_DIR / f'analysis_results_{self.timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {output_file}")

    def calculate_combined_metrics(self):
        """Calculate error rates for different processing pipelines"""
        # Calculate combined FNR for TF-IDF → Claude 2.1
        self.fnr_combined_21 = 1 - (1 - self.stages['tfidf'].fnr) * (1 - self.stages['claude21'].fnr)
        
        # Calculate combined FNR for TF-IDF → Claude 3.5
        self.fnr_combined_35 = 1 - (1 - self.stages['tfidf'].fnr) * (1 - self.stages['claude35'].fnr)

    def calculate_confidence_interval(self, 
                                   stage_key: str,
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for success rate of a specific stage.
        
        Args:
            stage_key: Key identifying the processing stage
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        stage = self.stages[stage_key]
        z_score = stats.norm.ppf((1 + confidence) / 2)
        p_hat = stage.tp / (stage.tp + stage.fn)
        n = stage.tp + stage.fn
        
        margin_of_error = z_score * np.sqrt((p_hat * (1 - p_hat)) / n)
        return (p_hat - margin_of_error, p_hat + margin_of_error)

    def plot_comparative_metrics(self):
        """Create a comprehensive comparison of metrics across all stages"""
        metrics = {
            'Precision': [stage.precision for stage in self.stages.values()],
            'Recall': [stage.recall for stage in self.stages.values()],
            'FNR': [stage.fnr for stage in self.stages.values()],
            'FPR': [stage.fpr for stage in self.stages.values()]
        }
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Add timestamp and model info to title
        title = (f"Comparative Analysis ({self.timestamp})\n"
                f"TF-IDF Transformer: {self.model_info['tfidf_transformer'].split('/')[-1]}")
        fig.suptitle(title, fontsize=10)
        
        # Plot precision and recall
        x = np.arange(len(self.stages))
        width = 0.35
        
        axes[0].bar(x - width/2, metrics['Precision'], width, label='Precision')
        axes[0].bar(x + width/2, metrics['Recall'], width, label='Recall')
        axes[0].set_ylabel('Rate')
        axes[0].set_title('Precision and Recall by Processing Stage')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([stage.name for stage in self.stages.values()])
        axes[0].legend()
        
        # Plot error rates
        axes[1].bar(x - width/2, metrics['FNR'], width, label='False Negative Rate')
        axes[1].bar(x + width/2, metrics['FPR'], width, label='False Positive Rate')
        axes[1].set_ylabel('Rate')
        axes[1].set_title('Error Rates by Processing Stage')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([stage.name for stage in self.stages.values()])
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot with detailed filename
        output_file = RESULTS_DIR / f'stage_comparison_{self.timestamp}.png'
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        logger.info(f"Comparative metrics plot saved to {output_file}")
        return fig

    def plot_pipeline_comparison(self):
        """Compare the performance of different processing pipelines"""
        # Prepare data for visualization
        pipeline_data = {
            'TF-IDF → Claude 2.1': self.fnr_combined_21,
            'TF-IDF → Claude 3.5': self.fnr_combined_35
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add timestamp and model info to title
        title = (f"Pipeline Performance Comparison ({self.timestamp})\n"
                f"Using {self.model_info['tfidf_transformer'].split('/')[-1]} for TF-IDF stage")
        ax.set_title(title, fontsize=10)
        
        # Create bar plot
        bars = ax.bar(pipeline_data.keys(), pipeline_data.values())
        ax.set_ylabel('Combined False Negative Rate')
        ax.set_title('Pipeline Performance Comparison')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot with detailed filename
        output_file = RESULTS_DIR / f'pipeline_comparison_{self.timestamp}.png'
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        logger.info(f"Pipeline comparison plot saved to {output_file}")
        return fig

    def calculate_buffer_requirements(self, 
                                   cost_ratio: float,
                                   confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate required buffers for different pipeline configurations.
        
        Args:
            cost_ratio: Cost ratio of missing requirement vs review
            confidence: Confidence level
            
        Returns:
            Dictionary of pipeline names and their required buffers
        """
        buffers = {}
        
        # Calculate confidence intervals for each LLM stage
        _, margin_21 = self.calculate_confidence_interval('claude21', confidence)
        _, margin_35 = self.calculate_confidence_interval('claude35', confidence)
        
        # Calculate buffers for each pipeline
        buffers['TF-IDF → Claude 2.1'] = self.fnr_combined_21 * cost_ratio * (1 + margin_21)
        buffers['TF-IDF → Claude 3.5'] = self.fnr_combined_35 * cost_ratio * (1 + margin_35)
        
        return buffers
    
    def calculate_cost_adjusted_buffer(self, 
                                    impact_cost: float,
                                    occurrence_prob: float,
                                    review_cost_per_hour: float,
                                    review_hours: float,
                                    confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Calculate cost-adjusted buffers comparing TF-IDF alone vs adding LLM stages.
        
        Args:
            impact_cost: Cost of missing a critical requirement ($)
            occurrence_prob: Probability of occurrence (0-1)
            review_cost_per_hour: Hourly cost of review ($)
            review_hours: Average hours needed for review
            confidence: Confidence level for statistical calculations
            
        Returns:
            Dictionary containing buffer calculations and intermediate values for each approach
        """
        try:
            # Calculate cost ratio as per ERROR_ANALYSIS.md formula
            cost_ratio = (impact_cost * occurrence_prob) / (review_cost_per_hour * review_hours)
            
            # Get confidence intervals
            tfidf_ci = self.calculate_confidence_interval('tfidf', confidence)
            tfidf_margin = tfidf_ci[1] - ((tfidf_ci[1] + tfidf_ci[0]) / 2)
            
            # Calculate buffers for TF-IDF alone
            tfidf_buffer = self.stages['tfidf'].fnr * cost_ratio * (1 + tfidf_margin)
            
            results = {
                'timestamp': self.timestamp,
                'cost_parameters': {
                    'impact_cost': impact_cost,
                    'occurrence_probability': occurrence_prob,
                    'review_cost_per_hour': review_cost_per_hour,
                    'review_hours': review_hours,
                    'calculated_cost_ratio': cost_ratio
                },
                'model_info': self.model_info,
                'approaches': {
                    'tfidf_only': {
                        'fnr': self.stages['tfidf'].fnr,
                        'buffer': tfidf_buffer,
                        'precision': self.stages['tfidf'].precision,
                        'recall': self.stages['tfidf'].recall,
                        'raw_counts': {
                            'tp': self.stages['tfidf'].tp,
                            'fp': self.stages['tfidf'].fp,
                            'tn': self.stages['tfidf'].tn,
                            'fn': self.stages['tfidf'].fn
                        }
                    },
                    'with_claude21': {
                        'fnr': self.fnr_combined_21,
                        'buffer': self.fnr_combined_21 * cost_ratio * (1 + tfidf_margin),
                        'precision': self.stages['claude21'].precision,
                        'recall': self.stages['claude21'].recall,
                        'raw_counts': {
                            'tp': self.stages['claude21'].tp,
                            'fp': self.stages['claude21'].fp,
                            'tn': self.stages['claude21'].tn,
                            'fn': self.stages['claude21'].fn
                        }
                    },
                    'with_claude35': {
                        'fnr': self.fnr_combined_35,
                        'buffer': self.fnr_combined_35 * cost_ratio * (1 + tfidf_margin),
                        'precision': self.stages['claude35'].precision,
                        'recall': self.stages['claude35'].recall,
                        'raw_counts': {
                            'tp': self.stages['claude35'].tp,
                            'fp': self.stages['claude35'].fp,
                            'tn': self.stages['claude35'].tn,
                            'fn': self.stages['claude35'].fn
                        }
                    }
                }
            }
            
            # Save results to file
            output_file = RESULTS_DIR / f'cost_analysis_{self.timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Cost analysis results saved to {output_file}")
            return results
            
        except Exception as e:
            handle_exception(e)
            return {}

    def calculate_cost_ratio(self,
                            impact_cost: float,
                            occurrence_prob: float,
                            review_cost_per_hour: float,
                            review_hours: float) -> float:
        """
        Calculate the cost ratio as defined in ERROR_ANALYSIS.md
        
        Args:
            impact_cost: Cost of missing a critical requirement ($)
            occurrence_prob: Probability of such an occurrence (0-1)
            review_cost_per_hour: Hourly cost of review ($)
            review_hours: Average hours needed for review
            
        Returns:
            float: Calculated cost ratio
        """
        try:
            return (impact_cost * occurrence_prob) / (review_cost_per_hour * review_hours)
        except Exception as e:
            handle_exception(e)
            return 0.0

    def calculate_required_buffer(self,
                            stage_key: str,
                            cost_ratio: float,
                            confidence: float = 0.95,
                            risk_premium: float = 0.0) -> float:
        """
        Calculate required buffer based on ERROR_ANALYSIS.md formula with enhanced consideration
        for FP vs FN impacts and pipeline effects.
        
        Args:
            stage_key: Key identifying the processing stage
            cost_ratio: Calculated cost ratio
            confidence: Confidence level for margin of error
            risk_premium: Additional risk premium (0.5-1.0 for critical requirements)
            
        Returns:
            float: Required buffer size
        """
        try:
            stage = self.stages[stage_key]
            
            # Calculate base error rates
            fnr = stage.fnr  # False Negative Rate
            fpr = stage.fpr  # False Positive Rate
            
            # Get margin of error from confidence interval
            _, upper = self.calculate_confidence_interval(stage_key, confidence)
            margin = upper - fnr
            
            # Calculate FN impact component (more expensive due to rework)
            fn_impact = fnr * cost_ratio * (1 + margin)
            
            # Calculate FP impact component (review overhead)
            fp_review_ratio = 0.2  # FP review cost relative to FN cost
            fp_impact = fpr * cost_ratio * fp_review_ratio
            
            # For LLM stages, consider pipeline effect
            if stage_key in ['claude21', 'claude35']:
                # Consider combined effect with TF-IDF
                tfidf_fnr = self.stages['tfidf'].fnr
                pipeline_fnr = 1 - (1 - tfidf_fnr) * (1 - fnr)
                fn_impact = pipeline_fnr * cost_ratio * (1 + margin)
            
            # Combine impacts with weights favoring FN prevention
            fn_weight = 0.8  # Higher weight for FN due to rework cost
            fp_weight = 0.2  # Lower weight for FP due to upfront review
            base_buffer = (fn_impact * fn_weight + fp_impact * fp_weight)
            
            # Apply risk premium if specified
            return base_buffer * (1 + risk_premium)
            
        except Exception as e:
            handle_exception(e)
            return 0.0

    def analyze_cost_impacts(self, total_pairs: int, cost_analysis: CostAnalysis) -> Dict:
        """
        Analyze cost impacts with enhanced buffer calculation
        """
        try:
            # Calculate base cost ratio with FN emphasis
            fn_impact_cost = (cost_analysis.rework_hours * cost_analysis.engineering_cost_per_hour +
                             cost_analysis.integration_cost +
                             cost_analysis.delay_cost_per_day * cost_analysis.typical_delay_days)
            
            fp_impact_cost = (cost_analysis.fp_review_hours * cost_analysis.review_cost_per_hour +
                             cost_analysis.fp_overhead_cost)
            
            # Weighted cost ratio favoring FN prevention
            base_cost_ratio = (fn_impact_cost * 0.8 + fp_impact_cost * 0.2) * 0.1 / \
                             (cost_analysis.review_cost_per_hour * cost_analysis.detailed_review_hours)
            
            # Calculate buffers for different risk levels and stages
            buffers = {
                'tfidf': {
                    'standard': self.calculate_required_buffer('tfidf', base_cost_ratio),
                    'critical': self.calculate_required_buffer('tfidf', base_cost_ratio, risk_premium=0.5),
                    'low_risk': self.calculate_required_buffer('tfidf', base_cost_ratio) * 0.75
                },
                'claude21': {
                    'standard': self.calculate_required_buffer('claude21', base_cost_ratio),
                    'critical': self.calculate_required_buffer('claude21', base_cost_ratio, risk_premium=0.5),
                    'low_risk': self.calculate_required_buffer('claude21', base_cost_ratio) * 0.75
                },
                'claude35': {
                    'standard': self.calculate_required_buffer('claude35', base_cost_ratio),
                    'critical': self.calculate_required_buffer('claude35', base_cost_ratio, risk_premium=0.5),
                    'low_risk': self.calculate_required_buffer('claude35', base_cost_ratio) * 0.75
                }
            }
            
            # Create ProcessingApproach instances for each stage
            approaches = [
                ProcessingApproach(
                    "TF-IDF + Transformer",
                    self.stages['tfidf'].tp,
                    self.stages['tfidf'].fp,
                    self.stages['tfidf'].tn,
                    self.stages['tfidf'].fn,
                    processing_time_hours=0.0002,  # ~0.7 sec per pair
                    cost_per_hour=50,
                    requires_exhaustive_review=True
                ),
                ProcessingApproach(
                    "TF-IDF + Claude 2.1",
                    self.stages['claude21'].tp,
                    self.stages['claude21'].fp,
                    self.stages['claude21'].tn,
                    self.stages['claude21'].fn,
                    processing_time_hours=0.2,
                    cost_per_hour=100,
                    requires_exhaustive_review=False
                ),
                ProcessingApproach(
                    "TF-IDF + Claude 3.5",
                    self.stages['claude35'].tp,
                    self.stages['claude35'].fp,
                    self.stages['claude35'].tn,
                    self.stages['claude35'].fn,
                    processing_time_hours=0.25,
                    cost_per_hour=150,
                    requires_exhaustive_review=False
                )
            ]
            
            results = []
            for approach in approaches:
                # Calculate processing costs
                if approach.requires_exhaustive_review:
                    processing_cost = approach.calculate_processing_cost(total_pairs)
                else:
                    # For LLM approaches, we first process with TF-IDF then apply LLM
                    tfidf_cost = 0.0002 * 50 * total_pairs  # TF-IDF processing
                    llm_cost = approach.calculate_processing_cost(approach.tp + approach.fp)
                    processing_cost = tfidf_cost + llm_cost
                
                # Calculate review costs
                if approach.requires_exhaustive_review:
                    # Quick review for true negatives
                    quick_review_cost = (approach.tn * cost_analysis.quick_review_minutes / 60 * 
                                       cost_analysis.review_cost_per_hour)
                    
                    # Detailed review for true positives and false positives
                    detailed_review_cost = ((approach.tp + approach.fp) * 
                                          cost_analysis.detailed_review_hours * 
                                          cost_analysis.review_cost_per_hour)
                else:
                    # For LLM approaches:
                    # 1. Quick review of TF-IDF results
                    quick_review_cost = (self.stages['tfidf'].tn * cost_analysis.quick_review_minutes / 60 * 
                                       cost_analysis.review_cost_per_hour)
                    
                    # 2. Detailed review of LLM results
                    detailed_review_cost = ((approach.tp + approach.fp) * 
                                          cost_analysis.detailed_review_hours * 
                                          cost_analysis.review_cost_per_hour)
                
                # Additional FP review costs (applies to all approaches)
                fp_review_cost = (approach.fp * cost_analysis.fp_review_hours * 
                                  cost_analysis.review_cost_per_hour)
                
                total_review_cost = quick_review_cost + detailed_review_cost + fp_review_cost
                
                # Calculate impact costs
                impact_cost = cost_analysis.calculate_impact_cost(approach.fn, approach.fp)
                
                results.append({
                    'Approach': approach.name,
                    'Processing Cost': processing_cost,
                    'Review Cost': total_review_cost,
                    'Impact Cost': impact_cost,
                    'Total Cost': processing_cost + total_review_cost + impact_cost,
                    'Quick Review Cost': quick_review_cost,
                    'Detailed Review Cost': detailed_review_cost,
                    'FP Review Cost': fp_review_cost,
                    'False Positives': approach.fp,
                    'False Negatives': approach.fn
                })
            
            # Save results
            results_df = pd.DataFrame(results)
            output_file = RESULTS_DIR / f'cost_impact_analysis_{self.timestamp}.json'
            results_df.to_json(output_file, orient='records', indent=4)
            logger.info(f"Cost impact analysis saved to {output_file}")
            
            # Add True Positives to the DataFrame for cost effectiveness calculation
            results_df['True Positives'] = [approach.tp for approach in approaches]
            
            # Create detailed cost comparisons
            self.plot_detailed_cost_comparisons(results_df, cost_analysis)
            
            # Add buffer calculations to results
            results_df['Required_Buffer'] = results_df['Approach'].map(
                lambda x: buffers['tfidf']['standard'] if 'TF-IDF' in x else buffers['tfidf']['critical']
            )
            
            # Save buffer analysis
            buffer_analysis = {
                'timestamp': self.timestamp,
                'cost_ratio': base_cost_ratio,
                'buffers': buffers,
                'parameters': {
                    'confidence_level': 0.95,
                    'occurrence_probability': 0.1,
                    'risk_premium_critical': 0.5
                }
            }
            
            buffer_file = RESULTS_DIR / f'buffer_analysis_{self.timestamp}.json'
            with open(buffer_file, 'w') as f:
                json.dump(buffer_analysis, f, indent=4)
            
            logger.info(f"Buffer analysis saved to {buffer_file}")
            
            # Create buffer analysis diagram
            self.plot_buffer_analysis_diagram(cost_analysis)
            
            # Create cost and time comparison plots
            self.plot_cost_and_time_comparison(results_df, cost_analysis)
            
            return results_df.to_dict('records')
            
        except Exception as e:
            handle_exception(e)
            return {}

    def plot_cost_stacked_comparison(self, df: pd.DataFrame):
        """Create stacked bar chart comparing total costs"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create stacked bar chart
            bottom = np.zeros(len(df))
            
            # Define cost categories and colors
            categories = ['Processing Cost', 'Review Cost', 'Impact Cost']
            colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            for cat, color in zip(categories, colors):
                plt.bar(df['Approach'], df[cat], bottom=bottom, label=cat, color=color)
                bottom += df[cat]
            
            # Customize the plot
            plt.title('Total Cost Comparison by Approach\n(Processing, Review, and Impact Costs)', 
                     pad=20, fontsize=12)
            plt.xlabel('Approach')
            plt.ylabel('Cost ($)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on each segment
            for i in range(len(df)):
                y_bottom = 0
                for cat in categories:
                    value = df.iloc[i][cat]
                    y_middle = y_bottom + (value / 2)
                    plt.text(i, y_middle, f'${value:,.0f}', 
                            ha='center', va='center')
                    y_bottom += value
            
            # Add total cost labels at the top
            for i, total in enumerate(df['Total Cost']):
                plt.text(i, bottom[i], f'Total: ${total:,.0f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            output_file = RESULTS_DIR / f'cost_stacked_comparison_{self.timestamp}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Cost stacked comparison plot saved to {output_file}")
            
            return plt.gcf()
            
        except Exception as e:
            handle_exception(e)
            return None

    def plot_buffer_analysis_diagram(self, cost_analysis: CostAnalysis):
        """
        Create a visual diagram showing:
        1. Process flow (top) - left to right flow
        2. Detailed formulas (middle)
        3. Calculations and practical implications (bottom)
        """
        try:
            plt.figure(figsize=(24, 20))
            gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2])
            
            # Calculate impact cost
            impact_cost = (cost_analysis.rework_hours * cost_analysis.engineering_cost_per_hour +
                          cost_analysis.integration_cost +
                          cost_analysis.delay_cost_per_day * cost_analysis.typical_delay_days)

            # Top: Process Flow (left to right)
            ax_flow = plt.subplot(gs[0])
            ax_flow.axis('off')
            
            stages = [
                (0.1, 0.5, 'Base FNR\n(False Negative Rate)'),
                (0.3, 0.5, 'Cost Ratio\nAdjustment'),
                (0.5, 0.5, 'Confidence\nInterval Margin'),
                (0.7, 0.5, 'Risk Premium\n(for Critical Reqs)'),
                (0.9, 0.5, 'Final Buffer')
            ]
            
            # Draw process boxes and arrows
            for i in range(len(stages)-1):
                # Draw box
                ax_flow.add_patch(plt.Rectangle((stages[i][0]-0.08, 0.3), 
                                              0.16, 0.4, facecolor='lightblue', alpha=0.3))
                ax_flow.text(stages[i][0], 0.5, stages[i][2], 
                            ha='center', va='center')
                
                # Draw arrow
                ax_flow.annotate('', xy=(stages[i+1][0]-0.08, 0.5),
                               xytext=(stages[i][0]+0.08, 0.5),
                               arrowprops=dict(arrowstyle='->'))
            
            # Draw final box
            ax_flow.add_patch(plt.Rectangle((stages[-1][0]-0.08, 0.3),
                                          0.16, 0.4, facecolor='lightgreen', alpha=0.3))
            ax_flow.text(stages[-1][0], 0.5, stages[-1][2],
                        ha='center', va='center')

            # Middle: Detailed Formulas
            ax_formulas = plt.subplot(gs[1])
            ax_formulas.axis('off')
            
            formula_text = (
                "Detailed Formulas:\n\n"
                "1. Base FNR = FN / (FN + TP)\n"
                "2. Cost Ratio = (Impact × Prob) / (Review Cost × Hours)\n"
                "3. Confidence Margin = z₀.₉₅ × √(FNR × (1-FNR) / n)\n"
                "4. Final Buffer = FNR × Cost Ratio × (1 + Margin) × (1 + Premium)\n\n"
                f"Impact Cost Components:\n"
                f"• Rework: {cost_analysis.rework_hours}h × ${cost_analysis.engineering_cost_per_hour}/h = ${cost_analysis.rework_hours * cost_analysis.engineering_cost_per_hour:,}\n"
                f"• Integration: ${cost_analysis.integration_cost:,}\n"
                f"• Delay: {cost_analysis.typical_delay_days} days × ${cost_analysis.delay_cost_per_day}/day = ${cost_analysis.delay_cost_per_day * cost_analysis.typical_delay_days:,}"
            )
            ax_formulas.text(0.5, 0.5, formula_text, fontsize=12, ha='center', va='center',
                            bbox=dict(facecolor='lightgray', alpha=0.1))

            # Bottom: Calculations and Practical Implications
            ax_calcs = plt.subplot(gs[2])
            ax_calcs.axis('off')
            
            approaches = ['TF-IDF', 'Claude 2.1', 'Claude 3.5']
            for idx, approach_name in enumerate(approaches):
                stage = {'TF-IDF': 'tfidf', 'Claude 2.1': 'claude21', 'Claude 3.5': 'claude35'}[approach_name]
                
                # Calculate values
                base_fnr = self.stages[stage].fnr
                _, upper = self.calculate_confidence_interval(stage, 0.95)
                margin = upper - base_fnr
                cost_ratio = impact_cost * 0.1 / (cost_analysis.review_cost_per_hour * cost_analysis.detailed_review_hours)
                
                # Calculate final buffer percentage
                buffer_pct = base_fnr * cost_ratio * (1 + margin) * 1.5
                
                # Translate buffer to practical terms
                total_requirements = self.stages[stage].tp + self.stages[stage].fn
                buffer_requirements = int(total_requirements * buffer_pct)
                buffer_hours = buffer_requirements * cost_analysis.detailed_review_hours
                buffer_cost = buffer_hours * cost_analysis.review_cost_per_hour
                
                x_pos = 0.25 + idx * 0.25
                calc_text = (
                    f"{approach_name} Approach:\n\n"
                    f"Base FNR = {base_fnr:.3f}\n"
                    f"Cost Ratio = {cost_ratio:.2f}\n"
                    f"Margin = {margin:.3f}\n"
                    f"Risk Premium = 50%\n\n"
                    f"Buffer Required: {buffer_pct:.1%}\n"
                    f"Practical Implementation:\n"
                    f"• Additional Requirements to Review: {buffer_requirements}\n"
                    f"• Additional Review Hours: {buffer_hours:.1f}h\n"
                    f"• Additional Cost: ${buffer_cost:,.2f}"
                )
                
                ax_calcs.text(x_pos, 0.5, calc_text, fontsize=12, ha='center', va='center',
                             bbox=dict(facecolor='lightblue', alpha=0.1))

            plt.suptitle('Buffer Analysis Methodology and Practical Implementation', 
                        fontsize=14, y=0.95)
            
            # Save diagram and data
            output_file = RESULTS_DIR / f'buffer_analysis_diagram_{self.timestamp}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Buffer analysis diagram saved to {output_file}")
            
            # Save calculation details including practical translations
            calculation_data = {
                'timestamp': self.timestamp,
                'impact_cost_components': {
                    'rework': cost_analysis.rework_hours * cost_analysis.engineering_cost_per_hour,
                    'integration': cost_analysis.integration_cost,
                    'delay': cost_analysis.delay_cost_per_day * cost_analysis.typical_delay_days,
                    'total': impact_cost
                },
                'approaches': {
                    name: {
                        'base_fnr': self.stages[stage].fnr,
                        'cost_ratio': cost_ratio,
                        'margin': margin,
                        'buffer_percentage': buffer_pct,
                        'practical_implementation': {
                            'additional_requirements': buffer_requirements,
                            'additional_hours': buffer_hours,
                            'additional_cost': buffer_cost
                        }
                    }
                    for name, stage in {'TF-IDF': 'tfidf', 'Claude 2.1': 'claude21', 'Claude 3.5': 'claude35'}.items()
                }
            }
            
            data_file = RESULTS_DIR / f'buffer_calculation_details_{self.timestamp}.json'
            with open(data_file, 'w') as f:
                json.dump(calculation_data, f, indent=4)
            logger.info(f"Buffer calculation details saved to {data_file}")
            
            return plt.gcf()
            
        except Exception as e:
            handle_exception(e)
            return None

    def plot_cost_and_time_comparison(self, df: pd.DataFrame, cost_analysis: CostAnalysis):
        """
        Create stacked bar charts comparing total costs and time estimates
        
        Args:
            df: DataFrame containing cost and approach data
            cost_analysis: CostAnalysis instance with cost parameters
        """
        try:
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Left subplot: Cost Comparison
            bottom = np.zeros(len(df))
            cost_categories = ['Processing Cost', 'Review Cost', 'Impact Cost']
            cost_colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            # Get x positions
            x_positions = np.arange(len(df))
            
            for cat, color in zip(cost_categories, cost_colors):
                ax1.bar(x_positions, df[cat], bottom=bottom, label=cat, color=color)
                bottom += df[cat]
            
            # Customize cost plot
            ax1.set_title('Total Cost Comparison by Approach', pad=20, fontsize=12)
            ax1.set_xlabel('Approach')
            ax1.set_ylabel('Cost ($)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Fix: Set ticks and labels properly
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(df['Approach'], rotation=45, ha='right')
            
            # Add value labels on cost segments
            for i in range(len(df)):
                y_bottom = 0
                for cat in cost_categories:
                    value = df.iloc[i][cat]
                    y_middle = y_bottom + (value / 2)
                    ax1.text(i, y_middle, f'${value:,.0f}', 
                            ha='center', va='center')
                    y_bottom += value
            
            # Add total cost labels
            for i, total in enumerate(df['Total Cost']):
                ax1.text(i, bottom[i], f'Total: ${total:,.0f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            # Right subplot: Time Estimation
            bottom_hours = np.zeros(len(df))
            time_categories = ['Processing Hours', 'Review Hours', 'Rework Hours']
            time_colors = ['#9b59b6', '#f1c40f', '#e67e22']
            
            # Calculate time components
            df['Processing Hours'] = df.apply(
                lambda x: x['Processing Cost'] / 50 if 'TF-IDF' in x['Approach'] 
                else x['Processing Cost'] / 100, axis=1
            )
            df['Review Hours'] = df.apply(
                lambda x: (x['Review Cost'] / cost_analysis.review_cost_per_hour), axis=1
            )
            df['Rework Hours'] = df.apply(
                lambda x: (x['Impact Cost'] / cost_analysis.engineering_cost_per_hour), axis=1
            )
            
            for cat, color in zip(time_categories, time_colors):
                ax2.bar(x_positions, df[cat], bottom=bottom_hours, 
                       label=cat, color=color)
                bottom_hours += df[cat]
            
            # Customize time plot
            ax2.set_title('Time Estimation by Approach', pad=20, fontsize=12)
            ax2.set_xlabel('Approach')
            ax2.set_ylabel('Hours')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Fix: Set ticks and labels properly
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(df['Approach'], rotation=45, ha='right')
            
            # Add value labels on time segments
            for i in range(len(df)):
                y_bottom = 0
                for cat in time_categories:
                    value = df.iloc[i][cat]
                    y_middle = y_bottom + (value / 2)
                    ax2.text(i, y_middle, f'{value:.1f}h', 
                            ha='center', va='center')
                    y_bottom += value
            
            # Add total time labels
            total_hours = bottom_hours
            for i, total in enumerate(total_hours):
                ax2.text(i, total, f'Total: {total:.1f}h', 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            output_file = RESULTS_DIR / f'cost_and_time_comparison_{self.timestamp}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Cost and time comparison plot saved to {output_file}")
            
            # Save data for future analysis
            analysis_data = {
                'timestamp': self.timestamp,
                'cost_analysis': {
                    'categories': cost_categories,
                    'data': df[cost_categories].to_dict('records')
                },
                'time_analysis': {
                    'categories': time_categories,
                    'data': df[time_categories].to_dict('records')
                },
                'approaches': df['Approach'].tolist(),
                'totals': {
                    'cost': df['Total Cost'].tolist(),
                    'hours': total_hours.tolist()
                },
                'raw_data': df.to_dict('records')
            }
            
            # Save analysis data to JSON
            data_file = RESULTS_DIR / f'cost_and_time_analysis_{self.timestamp}.json'
            with open(data_file, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            logger.info(f"Cost and time analysis data saved to {data_file}")
            
            return fig
            
        except Exception as e:
            handle_exception(e)
            return None

    def plot_detailed_cost_comparisons(self, df: pd.DataFrame, cost_analysis: CostAnalysis):
        """
        Create four separate visualizations for detailed cost analysis:
        1. Direct Costs (Processing + Review)
        2. Impact Costs (Rework + Integration + Delay)
        3. Time Distribution
        4. Cost-Effectiveness Metrics (Cost per True Match)
        """
        try:
            # Create a figure with 2x2 subplots
            fig = plt.figure(figsize=(20, 16))
            gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

            # 1. Direct Costs Plot
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_direct_costs(ax1, df)

            # 2. Impact Costs Plot
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_impact_costs(ax2, df)

            # 3. Time Distribution Plot
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_time_distribution(ax3, df, cost_analysis)

            # 4. Cost-Effectiveness Metrics
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_cost_effectiveness(ax4, df)

            plt.suptitle('Detailed Cost Analysis Comparison', fontsize=16, y=0.95)
            
            # Save plot
            output_file = RESULTS_DIR / f'detailed_cost_analysis_{self.timestamp}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Detailed cost analysis saved to {output_file}")

            # Save data for future analysis
            analysis_data = {
                'timestamp': self.timestamp,
                'direct_costs': {
                    'processing': df['Processing Cost'].to_dict(),
                    'review': df['Review Cost'].to_dict()
                },
                'impact_costs': {
                    'rework': df['Impact Cost'].to_dict(),
                    'integration': (df['Impact Cost'] * 0.2).to_dict(),  # Estimated split
                    'delay': (df['Impact Cost'] * 0.3).to_dict()  # Estimated split
                },
                'time_distribution': {
                    'processing_hours': df['Processing Hours'].to_dict(),
                    'review_hours': df['Review Hours'].to_dict(),
                    'rework_hours': df['Rework Hours'].to_dict()
                },
                'cost_effectiveness': {
                    'cost_per_true_match': (df['Total Cost'] / df['True Positives']).to_dict(),
                    'cost_per_requirement': (df['Total Cost'] / len(df)).to_dict()
                },
                'raw_data': df.to_dict('records')
            }

            # Save analysis data to JSON
            data_file = RESULTS_DIR / f'detailed_cost_analysis_{self.timestamp}.json'
            with open(data_file, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            logger.info(f"Detailed cost analysis data saved to {data_file}")

            return fig

        except Exception as e:
            handle_exception(e)
            return None

    def _plot_direct_costs(self, ax, df):
        """Plot direct costs breakdown"""
        x = np.arange(len(df))
        width = 0.35

        ax.bar(x - width/2, df['Processing Cost'], width, label='Processing',
               color='#2ecc71')
        ax.bar(x + width/2, df['Review Cost'], width, label='Review',
               color='#3498db')

        ax.set_title('Direct Costs Breakdown')
        ax.set_xlabel('Approach')
        ax.set_ylabel('Cost ($)')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Approach'], rotation=45, ha='right')
        ax.legend()

        # Add value labels
        for i in x:
            ax.text(i - width/2, df['Processing Cost'].iloc[i],
                    f'${df["Processing Cost"].iloc[i]:,.0f}',
                    ha='center', va='bottom')
            ax.text(i + width/2, df['Review Cost'].iloc[i],
                    f'${df["Review Cost"].iloc[i]:,.0f}',
                    ha='center', va='bottom')

    def _plot_impact_costs(self, ax, df):
        """Plot impact costs breakdown"""
        bottom = np.zeros(len(df))
        impact_categories = ['Rework', 'Integration', 'Delay']
        colors = ['#e74c3c', '#e67e22', '#f1c40f']

        # Estimate the breakdown of impact costs
        impact_splits = {
            'Rework': 0.5,
            'Integration': 0.2,
            'Delay': 0.3
        }

        for cat, color in zip(impact_categories, colors):
            values = df['Impact Cost'] * impact_splits[cat]
            ax.bar(df['Approach'], values, bottom=bottom,
                   label=cat, color=color)
            bottom += values

        ax.set_title('Impact Costs Breakdown')
        ax.set_xlabel('Approach')
        ax.set_ylabel('Cost ($)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    def _plot_time_distribution(self, ax, df, cost_analysis):
        """Plot time distribution"""
        x = np.arange(len(df))
        width = 0.25

        ax.bar(x - width, df['Processing Hours'], width, label='Processing',
               color='#9b59b6')
        ax.bar(x, df['Review Hours'], width, label='Review',
               color='#f1c40f')
        ax.bar(x + width, df['Rework Hours'], width, label='Rework',
               color='#e67e22')

        ax.set_title('Time Distribution by Activity')
        ax.set_xlabel('Approach')
        ax.set_ylabel('Hours')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Approach'], rotation=45, ha='right')
        ax.legend()

    def _plot_cost_effectiveness(self, ax, df):
        """Plot cost effectiveness metrics"""
        x = np.arange(len(df))
        
        # Calculate cost per true match
        df['Cost per True Match'] = df['Total Cost'] / df['True Positives']
        
        # Create line plot for cost effectiveness
        ax2 = ax.twinx()
        
        # Bar plot for total cost
        bars = ax.bar(x, df['Total Cost'], alpha=0.3, color='#95a5a6',
                      label='Total Cost')
        ax.set_ylabel('Total Cost ($)')
        
        # Line plot for cost per match
        line = ax2.plot(x, df['Cost per True Match'], color='#e74c3c',
                        marker='o', label='Cost per True Match')
        ax2.set_ylabel('Cost per True Match ($)')
        
        ax.set_title('Cost Effectiveness Analysis')
        ax.set_xlabel('Approach')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Approach'], rotation=45, ha='right')
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

def main():
    try:
        # Sample data for all stages
        tfidf_results = {
            'tp': 318,
            'fp': 10156,
            'tn': 41183,
            'fn': 43
        }
        
        claude21_results = {
            'tp': 108,
            'fp': 443,
            'tn': 50896,
            'fn': 253
        }
        
        claude35_results = {
            'tp': 152,
            'fp': 566,
            'tn': 50773,
            'fn': 209
        }
        
        # Initialize analysis with model info
        model_info = {
            'tfidf_transformer': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'claude21': 'anthropic/claude-2.1',
            'claude35': 'anthropic/claude-3.5-sonnet'
        }
        
        analysis = MultiStageLLMAnalysis(
            tfidf_results, 
            claude21_results, 
            claude35_results,
            model_info
        )
        
        # Generate and save visualizations
        analysis.plot_comparative_metrics()
        analysis.plot_pipeline_comparison()
        
        # Save numerical results
        analysis.save_results()
        
        # Calculate and save buffer requirements
        cost_ratios = [5, 10, 15, 20]
        buffer_results = {
            'timestamp': analysis.timestamp,
            'cost_ratio_analysis': {}
        }
        
        for ratio in cost_ratios:
            buffers = analysis.calculate_buffer_requirements(ratio)
            buffer_results['cost_ratio_analysis'][f'ratio_{ratio}'] = buffers
        
        buffer_file = RESULTS_DIR / f'buffer_requirements_{analysis.timestamp}.json'
        with open(buffer_file, 'w') as f:
            json.dump(buffer_results, f, indent=4)
        
        logger.info(f"Buffer requirements saved to {buffer_file}")

        # Add cost-adjusted buffer analysis
        cost_scenarios = [
            {
                'impact_cost': 100000,  # $100k impact
                'occurrence_prob': 0.1,  # 10% probability
                'review_cost_per_hour': 150,  # $150/hour
                'review_hours': 2  # 2 hours average review
            },
            {
                'impact_cost': 50000,   # $50k impact
                'occurrence_prob': 0.2,  # 20% probability
                'review_cost_per_hour': 150,
                'review_hours': 2
            }
        ]
        
        buffer_results = {
            'timestamp': analysis.timestamp,
            'scenarios': {}
        }
        
        for i, scenario in enumerate(cost_scenarios, 1):
            results = analysis.calculate_cost_adjusted_buffer(**scenario)
            buffer_results['scenarios'][f'scenario_{i}'] = results
        
        buffer_file = RESULTS_DIR / f'buffer_analysis_{analysis.timestamp}.json'
        with open(buffer_file, 'w') as f:
            json.dump(buffer_results, f, indent=4)
        
        logger.info(f"Buffer analysis saved to {buffer_file}")
        
        # Add cost analysis with false positive considerations
        cost_analysis = CostAnalysis(
            review_cost_per_hour=150,        # Senior engineer review cost
            engineering_cost_per_hour=150,    # Implementation engineer cost
            rework_hours=40,                  # Hours to fix a missed requirement
            integration_cost=5000,            # Additional integration cost per miss
            delay_cost_per_day=1000,         # Cost of schedule delay per day
            typical_delay_days=5,            # Typical delay days per missed requirement
            quick_review_minutes=1,          # Time for quick review of likely non-matches
            detailed_review_hours=2,         # Time for detailed review of potential matches
            fp_review_hours=1,              # Additional hours to review each false positive
            fp_overhead_cost=500            # Overhead cost per false positive (meetings, documentation, etc.)
        )
        
        # Total requirement pairs
        total_pairs = 51700  # 235 * 220
        
        # Perform cost analysis
        cost_results = analysis.analyze_cost_impacts(total_pairs, cost_analysis)
        
        logger.info("Cost analysis completed successfully")
        
    except Exception as e:
        handle_exception(e)

if __name__ == "__main__":
    main()