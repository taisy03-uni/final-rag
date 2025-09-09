"""
In-Depth Analysis of Metadata Guided RAG Technique
==================================================

This script provides comprehensive analysis of the best performing metadata-based RAG technique,
including detailed performance breakdowns, metadata extraction analysis, and comparative insights.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple, Any
import os
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_detailed_results():
    """Load the detailed results from the metadata RAG evaluation"""
    with open('evaluation/evaluation4/generative_detailed_results_chunks.json', 'r') as f:
        return json.load(f)

def load_evaluation_data():
    """Load the original evaluation dataset"""
    with open('evaluation/evaluation3/eval3.json', 'r') as f:
        return json.load(f)

class MetadataGuidedAnalyzer:
    def __init__(self):
        self.results_data = load_detailed_results()
        self.eval_data = load_evaluation_data()
        self.metadata_guided_results = self.results_data['metadata_guided']['results']
        self.baseline_results = self.results_data['baseline_chunks']['results']
        
    def extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract detailed performance metrics for metadata guided technique"""
        
        print("🔍 EXTRACTING PERFORMANCE METRICS")
        print("=" * 50)
        
        results = self.metadata_guided_results
        
        # Basic metrics
        f1_scores = [r['f1_score'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        
        metrics = {
            'total_queries': len(results),
            'successful_queries': sum(1 for f1 in f1_scores if f1 > 0),
            'avg_f1': np.mean(f1_scores),
            'median_f1': np.median(f1_scores),
            'std_f1': np.std(f1_scores),
            'max_f1': np.max(f1_scores),
            'min_f1': np.min(f1_scores),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'success_rate': sum(1 for f1 in f1_scores if f1 > 0) / len(f1_scores)
        }
        
        # Performance distribution
        metrics['f1_distribution'] = {
            'excellent (>0.5)': sum(1 for f1 in f1_scores if f1 > 0.5),
            'good (0.3-0.5)': sum(1 for f1 in f1_scores if 0.3 <= f1 <= 0.5),
            'moderate (0.1-0.3)': sum(1 for f1 in f1_scores if 0.1 <= f1 < 0.3),
            'poor (0-0.1)': sum(1 for f1 in f1_scores if 0 < f1 < 0.1),
            'failed (0)': sum(1 for f1 in f1_scores if f1 == 0)
        }
        
        # Query-level analysis
        query_analysis = []
        for i, result in enumerate(results):
            query_data = self.eval_data[i]
            analysis = {
                'query_id': i + 1,
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall'],
                'law_type': query_data.get('law_type', 'Unknown'),
                'expected_count': len(result['expected_uri']),
                'retrieved_count': len(result['retrieved_uri']),
                'matched_count': len(set(result['expected_uri']) & set(result['retrieved_uri'])),
                'query_length': len(result['query'].split()),
                'scenario_complexity': 'complex' if len(query_data['scenario'].split()) > 50 else 'simple'
            }
            query_analysis.append(analysis)
        
        metrics['query_analysis'] = query_analysis
        
        print(f"Total Queries Analyzed: {metrics['total_queries']}")
        print(f"Successful Queries: {metrics['successful_queries']} ({metrics['success_rate']:.1%})")
        print(f"Average F1-Score: {metrics['avg_f1']:.4f}")
        print(f"Performance Range: {metrics['min_f1']:.4f} - {metrics['max_f1']:.4f}")
        
        return metrics
    
    def analyze_vs_baseline(self) -> Dict[str, Any]:
        """Compare metadata guided technique with baseline performance"""
        
        print("\nCOMPARING WITH BASELINE")
        print("=" * 50)
        
        mg_results = self.metadata_guided_results
        bl_results = self.baseline_results
        
        comparison = {
            'metadata_guided': {
                'avg_f1': np.mean([r['f1_score'] for r in mg_results]),
                'success_rate': sum(1 for r in mg_results if r['f1_score'] > 0) / len(mg_results),
                'avg_precision': np.mean([r['precision'] for r in mg_results]),
                'avg_recall': np.mean([r['recall'] for r in mg_results])
            },
            'baseline': {
                'avg_f1': np.mean([r['f1_score'] for r in bl_results]),
                'success_rate': sum(1 for r in bl_results if r['f1_score'] > 0) / len(bl_results),
                'avg_precision': np.mean([r['precision'] for r in bl_results]),
                'avg_recall': np.mean([r['recall'] for r in bl_results])
            }
        }
        
        # Calculate improvements
        improvements = {}
        for metric in ['avg_f1', 'success_rate', 'avg_precision', 'avg_recall']:
            baseline_val = comparison['baseline'][metric]
            metadata_val = comparison['metadata_guided'][metric]
            
            if baseline_val > 0:
                improvement = ((metadata_val - baseline_val) / baseline_val) * 100
            else:
                improvement = float('inf') if metadata_val > 0 else 0
                
            improvements[metric] = improvement
        
        comparison['improvements'] = improvements
        
        # Query-by-query comparison
        query_comparisons = []
        for i in range(len(mg_results)):
            mg_f1 = mg_results[i]['f1_score']
            bl_f1 = bl_results[i]['f1_score']
            
            query_comp = {
                'query_id': i + 1,
                'metadata_f1': mg_f1,
                'baseline_f1': bl_f1,
                'improvement': mg_f1 - bl_f1,
                'relative_improvement': ((mg_f1 - bl_f1) / bl_f1 * 100) if bl_f1 > 0 else float('inf') if mg_f1 > 0 else 0,
                'law_type': self.eval_data[i].get('law_type', 'Unknown')
            }
            query_comparisons.append(query_comp)
        
        comparison['query_comparisons'] = query_comparisons
        
        print(f"F1-Score Improvement: {improvements['avg_f1']:.1f}%")
        print(f"Success Rate Improvement: {improvements['success_rate']:.1f}%")
        print(f"Precision Improvement: {improvements['avg_precision']:.1f}%")
        print(f"Recall Improvement: {improvements['avg_recall']:.1f}%")
        
        # Count wins/losses
        wins = sum(1 for q in query_comparisons if q['improvement'] > 0)
        losses = sum(1 for q in query_comparisons if q['improvement'] < 0)
        ties = sum(1 for q in query_comparisons if q['improvement'] == 0)
        
        print(f"Query-by-Query: {wins} wins, {losses} losses, {ties} ties")
        
        return comparison
    
    def analyze_by_legal_domain(self) -> Dict[str, Any]:
        """Analyze performance by legal domain"""
        
        print("\nANALYZING BY LEGAL DOMAIN")
        
        domain_performance = defaultdict(list)
        
        for i, result in enumerate(self.metadata_guided_results):
            law_type = self.eval_data[i].get('law_type', 'Unknown')
            domain_performance[law_type].append({
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall'],
                'query_id': i + 1
            })
        
        # Calculate domain statistics
        domain_stats = {}
        for domain, results in domain_performance.items():
            f1_scores = [r['f1_score'] for r in results]
            domain_stats[domain] = {
                'count': len(results),
                'avg_f1': np.mean(f1_scores),
                'success_rate': sum(1 for f1 in f1_scores if f1 > 0) / len(f1_scores),
                'max_f1': np.max(f1_scores),
                'std_f1': np.std(f1_scores),
                'query_ids': [r['query_id'] for r in results]
            }
        
        # Sort by performance
        sorted_domains = sorted(domain_stats.items(), key=lambda x: x[1]['avg_f1'], reverse=True)
        
        print("Domain Performance Ranking:")
        for i, (domain, stats) in enumerate(sorted_domains, 1):
            print(f"{i}. {domain}: F1={stats['avg_f1']:.4f}, Success={stats['success_rate']:.1%}, Count={stats['count']}")
        
        return {
            'domain_performance': domain_performance,
            'domain_stats': domain_stats,
            'sorted_domains': sorted_domains
        }
    
    def identify_success_failure_patterns(self) -> Dict[str, Any]:
        """Identify patterns in successful vs failed queries"""
        
        print("\nIDENTIFYING SUCCESS/FAILURE PATTERNS")
        print("=" * 50)
        
        successful_queries = []
        failed_queries = []
        
        for i, result in enumerate(self.metadata_guided_results):
            query_data = self.eval_data[i]
            
            query_info = {
                'query_id': i + 1,
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall'],
                'law_type': query_data.get('law_type', 'Unknown'),
                'scenario_length': len(query_data['scenario'].split()),
                'question_length': len(query_data['question'].split()),
                'total_length': len((query_data['scenario'] + ' ' + query_data['question']).split()),
                'expected_cases': len(result['expected_uri']),
                'retrieved_cases': len(result['retrieved_uri']),
                'scenario': query_data['scenario'][:100] + "..." if len(query_data['scenario']) > 100 else query_data['scenario'],
                'question': query_data['question']
            }
            
            if result['f1_score'] > 0:
                successful_queries.append(query_info)
            else:
                failed_queries.append(query_info)
        
        # Analyze patterns
        patterns = {
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'success_patterns': self._analyze_query_patterns(successful_queries),
            'failure_patterns': self._analyze_query_patterns(failed_queries)
        }
        
        print(f"Successful Queries: {len(successful_queries)}")
        print(f"Failed Queries: {len(failed_queries)}")
        
        if successful_queries:
            print(f"Avg Length (Successful): {np.mean([q['total_length'] for q in successful_queries]):.1f} words")
            print(f"Most Common Domain (Successful): {Counter([q['law_type'] for q in successful_queries]).most_common(1)[0]}")
        
        if failed_queries:
            print(f"Avg Length (Failed): {np.mean([q['total_length'] for q in failed_queries]):.1f} words")
            print(f"Most Common Domain (Failed): {Counter([q['law_type'] for q in failed_queries]).most_common(1)[0]}")
        
        return patterns
    
    def _analyze_query_patterns(self, queries: List[Dict]) -> Dict[str, Any]:
        """Helper method to analyze patterns in query groups"""
        if not queries:
            return {}
        
        return {
            'avg_scenario_length': np.mean([q['scenario_length'] for q in queries]),
            'avg_question_length': np.mean([q['question_length'] for q in queries]),
            'avg_total_length': np.mean([q['total_length'] for q in queries]),
            'law_type_distribution': Counter([q['law_type'] for q in queries]),
            'expected_cases_avg': np.mean([q['expected_cases'] for q in queries]),
            'length_range': {
                'min': min([q['total_length'] for q in queries]),
                'max': max([q['total_length'] for q in queries])
            }
        }
    
    def create_comprehensive_visualizations(self, metrics: Dict, comparison: Dict, domain_analysis: Dict, patterns: Dict):
        """Create detailed visualizations explaining the metadata guided technique"""
        
        print("\nCREATING COMPREHENSIVE VISUALIZATIONS")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        f1_scores = [r['f1_score'] for r in self.metadata_guided_results]
        
        ax1.hist(f1_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.4f}')
        ax1.axvline(np.median(f1_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(f1_scores):.4f}')
        ax1.set_xlabel('F1-Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('F1-Score Distribution\n(Metadata Guided Technique)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Metadata vs Baseline Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        metrics_names = ['F1-Score', 'Precision', 'Recall', 'Success Rate']
        metadata_vals = [comparison['metadata_guided']['avg_f1'], comparison['metadata_guided']['avg_precision'], 
                        comparison['metadata_guided']['avg_recall'], comparison['metadata_guided']['success_rate']]
        baseline_vals = [comparison['baseline']['avg_f1'], comparison['baseline']['avg_precision'], 
                        comparison['baseline']['avg_recall'], comparison['baseline']['success_rate']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, metadata_vals, width, label='Metadata Guided', alpha=0.8, color='lightgreen')
        bars2 = ax2.bar(x + width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Metadata Guided vs Baseline\nPerformance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Query-by-Query Improvement
        ax3 = fig.add_subplot(gs[0, 2])
        query_comparisons = comparison['query_comparisons']
        improvements = [q['improvement'] for q in query_comparisons]
        query_ids = [q['query_id'] for q in query_comparisons]
        
        colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
        bars = ax3.bar(query_ids, improvements, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Query ID')
        ax3.set_ylabel('F1-Score Improvement')
        ax3.set_title('Query-by-Query Improvement\nover Baseline')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance by Legal Domain
        ax4 = fig.add_subplot(gs[1, 0])
        domain_stats = domain_analysis['domain_stats']
        domains = list(domain_stats.keys())
        domain_f1s = [stats['avg_f1'] for stats in domain_stats.values()]
        
        bars = ax4.barh(domains, domain_f1s, alpha=0.8, color='lightblue')
        ax4.set_xlabel('Average F1-Score')
        ax4.set_title('Performance by Legal Domain')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, f1) in enumerate(zip(bars, domain_f1s)):
            width = bar.get_width()
            ax4.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{f1:.3f}', ha='left', va='center', fontsize=8)
        
        # 5. Success vs Failure Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Performance categories
        categories = list(metrics['f1_distribution'].keys())
        counts = list(metrics['f1_distribution'].values())
        colors_cat = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        
        wedges, texts, autotexts = ax5.pie(counts, labels=categories, autopct='%1.1f%%', 
                                          colors=colors_cat, startangle=90)
        ax5.set_title('Performance Distribution\nCategories')
        
        # 6. Query Length vs Performance
        ax6 = fig.add_subplot(gs[1, 2])
        query_lengths = [len(r['query'].split()) for r in self.metadata_guided_results]
        f1_scores = [r['f1_score'] for r in self.metadata_guided_results]
        
        scatter = ax6.scatter(query_lengths, f1_scores, alpha=0.6, s=50, color='purple')
        
        # Add trend line
        z = np.polyfit(query_lengths, f1_scores, 1)
        p = np.poly1d(z)
        ax6.plot(query_lengths, p(query_lengths), "r--", alpha=0.8)
        
        ax6.set_xlabel('Query Length (words)')
        ax6.set_ylabel('F1-Score')
        ax6.set_title('Query Length vs Performance')
        ax6.grid(True, alpha=0.3)
        
        # 7. Detailed Performance Heatmap
        ax7 = fig.add_subplot(gs[2, :])
        
        # Create heatmap data
        query_metrics = []
        for i, result in enumerate(self.metadata_guided_results):
            query_metrics.append([
                result['f1_score'],
                result['precision'], 
                result['recall'],
                1 if result['f1_score'] > 0 else 0  # Success indicator
            ])
        
        heatmap_data = np.array(query_metrics).T
        
        im = ax7.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax7.set_yticks([0, 1, 2, 3])
        ax7.set_yticklabels(['F1-Score', 'Precision', 'Recall', 'Success'])
        ax7.set_xlabel('Query ID')
        ax7.set_title('Performance Heatmap Across All Queries\n(Green = Better Performance)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Score')
        
        # 8. Success Pattern Analysis
        ax8 = fig.add_subplot(gs[3, 0])
        
        if patterns['successful_queries'] and patterns['failed_queries']:
            success_lengths = [q['total_length'] for q in patterns['successful_queries']]
            failure_lengths = [q['total_length'] for q in patterns['failed_queries']]
            
            ax8.hist(success_lengths, alpha=0.6, label='Successful', bins=10, color='green')
            ax8.hist(failure_lengths, alpha=0.6, label='Failed', bins=10, color='red')
            ax8.set_xlabel('Query Length (words)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Query Length Distribution\nSuccess vs Failure')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Expected vs Retrieved Cases
        ax9 = fig.add_subplot(gs[3, 1])
        expected_counts = [len(r['expected_uri']) for r in self.metadata_guided_results]
        retrieved_counts = [len(r['retrieved_uri']) for r in self.metadata_guided_results]
        
        ax9.scatter(expected_counts, retrieved_counts, alpha=0.6, s=50)
        
        # Perfect match line
        max_count = max(max(expected_counts), max(retrieved_counts))
        ax9.plot([0, max_count], [0, max_count], 'r--', alpha=0.5, label='Perfect Match')
        
        ax9.set_xlabel('Expected Cases')
        ax9.set_ylabel('Retrieved Cases')
        ax9.set_title('Expected vs Retrieved Cases\nper Query')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Technique Effectiveness Summary
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.axis('off')
        
        # Create summary text
        summary_text = f"""
        METADATA GUIDED TECHNIQUE SUMMARY
        {'='*35}

        OVERALL PERFORMANCE:
        • F1-Score: {metrics['avg_f1']:.4f}
        • Success Rate: {metrics['success_rate']:.1%}
        • Queries Improved: {sum(1 for q in query_comparisons if q['improvement'] > 0)}/{len(query_comparisons)}

        VS BASELINE IMPROVEMENT:
        • F1-Score: +{comparison['improvements']['avg_f1']:.1f}%
        • Success Rate: +{comparison['improvements']['success_rate']:.1f}%
        • Precision: +{comparison['improvements']['avg_precision']:.1f}%
        • Recall: +{comparison['improvements']['avg_recall']:.1f}%

        ⚖️ BEST DOMAINS:
        """
        
        # Add top 3 domains
        for i, (domain, stats) in enumerate(domain_analysis['sorted_domains'][:3]):
            summary_text += f"   {i+1}. {domain}: {stats['avg_f1']:.3f}\n"
        
        summary_text += f"""
            KEY INSIGHTS:
        • {len(patterns['successful_queries'])} successful queries
        • {len(patterns['failed_queries'])} failed queries  
        • Best F1-Score: {metrics['max_f1']:.4f}
        • Performance Range: {metrics['min_f1']:.3f} - {metrics['max_f1']:.3f}
        """
        
        ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Metadata Guided RAG Technique - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # Save the visualization
        output_path = 'evaluation/evaluation4/deep_analysis/metadata_guided_comprehensive_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive visualization saved to: {output_path}")
        
        return output_path
    

def main():
    """Main execution function"""
    
    print("STARTING METADATA GUIDED TECHNIQUE DEEP ANALYSIS")
    print("=" * 60)
    
    analyzer = MetadataGuidedAnalyzer()
    
    # Step 1: Extract performance metrics  
    print("\n" + "="*60)
    metrics = analyzer.extract_performance_metrics()
    
    # Step 2: Compare with baseline
    print("\n" + "="*60) 
    comparison = analyzer.analyze_vs_baseline()
    
    # Step 3: Analyze by legal domain
    print("\n" + "="*60)
    domain_analysis = analyzer.analyze_by_legal_domain()
    
    # Step 4: Identify success/failure patterns  
    print("\n" + "="*60)
    patterns = analyzer.identify_success_failure_patterns()
    
    # Step 5: Create comprehensive visualizations
    print("\n" + "="*60)
    viz_path = analyzer.create_comprehensive_visualizations(metrics, comparison, domain_analysis, patterns)
    
    # Save analysis data
    analysis_data = {
        'metrics': metrics,
        'comparison': comparison, 
        'domain_analysis': domain_analysis,
        'patterns': patterns,
        'summary': {
            'best_f1': metrics['max_f1'],
            'avg_f1': metrics['avg_f1'],
            'success_rate': metrics['success_rate'],
            'improvement_over_baseline': comparison['improvements']['avg_f1'],
            'total_queries': metrics['total_queries'],
            'successful_queries': metrics['successful_queries']
        }
    }
    
    data_path = 'evaluation/evaluation4/deep_analysis/metadata_guided_analysis_data.json'
    with open(data_path, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"   • Best F1-Score: {metrics['max_f1']:.4f}")
    print(f"   • Average F1-Score: {metrics['avg_f1']:.4f}")  
    print(f"   • Success Rate: {metrics['success_rate']:.1%}")
    print(f"   • Improvement over Baseline: +{comparison['improvements']['avg_f1']:.1f}%")
    print(f"   • Queries Improved: {sum(1 for q in comparison['query_comparisons'] if q['improvement'] > 0)}/{len(comparison['query_comparisons'])}")

if __name__ == "__main__":
    main()