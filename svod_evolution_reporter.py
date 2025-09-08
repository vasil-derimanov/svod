"""
SVOD Version Evolution Report Generator
Comprehensive reporting system for SVOD development evolution and insights

Version: 1.0.0
Date: September 8, 2025
Author: SVOD Development Team

Features:
- Historical accuracy and performance trend analysis
- YOLOv8 vs YOLOv4 adoption and impact tracking
- Detection method evolution visualization
- Cross-platform performance insights
- Regression detection and improvement validation
- Model effectiveness comparison across versions
"""

import json
import csv
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import argparse
import re
from collections import defaultdict, Counter

__version__ = "1.0.0" 
__release_date__ = "2025-09-08"

class SVODEvolutionReporter:
    """Generate comprehensive evolution reports for SVOD development"""
    
    def __init__(self, svod_root: Path):
        self.svod_root = Path(svod_root)
        self.db_path = self.svod_root / "svod_statistics.db"
        self.reports = {}
        
    def load_historical_data(self) -> Dict[str, Any]:
        """Load historical data from database and files"""
        historical_data = {
            'versions': [],
            'benchmarks': [],
            'comparisons': [],
            'statistics': []
        }
        
        # Load from SQLite database if exists
        if self.db_path.exists():
            historical_data['statistics'] = self._load_from_database()
        
        # Load benchmark results
        historical_data['benchmarks'] = self._load_benchmark_files()
        
        # Load comparison results
        historical_data['comparisons'] = self._load_comparison_files()
        
        # Load version information from git if available
        historical_data['versions'] = self._load_git_history()
        
        return historical_data
    
    def _load_from_database(self) -> List[Dict[str, Any]]:
        """Load statistics from SQLite database"""
        stats = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get version statistics with detection usage
            cursor.execute('''
                SELECT vs.version, vs.release_date, vs.timestamp, vs.accuracy,
                       vs.avg_processing_time, vs.total_videos, vs.yolo_version, vs.notes,
                       GROUP_CONCAT(du.method_name || ':' || du.usage_count) as detection_methods
                FROM version_stats vs
                LEFT JOIN detection_usage du ON vs.id = du.version_id
                GROUP BY vs.id
                ORDER BY vs.timestamp
            ''')
            
            for row in cursor.fetchall():
                detection_methods = {}
                if row[8]:  # detection_methods
                    for method_data in row[8].split(','):
                        if ':' in method_data:
                            method, count = method_data.split(':', 1)
                            detection_methods[method] = int(count)
                
                stats.append({
                    'version': row[0],
                    'release_date': row[1],
                    'timestamp': row[2],
                    'accuracy': row[3],
                    'avg_processing_time': row[4],
                    'total_videos': row[5],
                    'yolo_version': row[6],
                    'notes': row[7],
                    'detection_methods': detection_methods
                })
            
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not load from database: {e}")
        
        return stats
    
    def _load_benchmark_files(self) -> List[Dict[str, Any]]:
        """Load benchmark result files"""
        benchmarks = []
        
        for benchmark_file in self.svod_root.glob("svod_benchmark_*.json"):
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)
                    benchmark_data['source_file'] = benchmark_file.name
                    benchmarks.append(benchmark_data)
            except Exception as e:
                print(f"Warning: Could not load benchmark file {benchmark_file}: {e}")
        
        # Sort by timestamp
        benchmarks.sort(key=lambda x: x.get('timestamp', ''))
        return benchmarks
    
    def _load_comparison_files(self) -> List[Dict[str, Any]]:
        """Load version comparison files"""
        comparisons = []
        
        for comparison_file in self.svod_root.glob("svod_version_comparison_*.json"):
            try:
                with open(comparison_file, 'r', encoding='utf-8') as f:
                    comparison_data = json.load(f)
                    comparison_data['source_file'] = comparison_file.name
                    comparisons.append(comparison_data)
            except Exception as e:
                print(f"Warning: Could not load comparison file {comparison_file}: {e}")
        
        # Sort by timestamp
        comparisons.sort(key=lambda x: x.get('timestamp', ''))
        return comparisons
    
    def _load_git_history(self) -> List[Dict[str, Any]]:
        """Load git commit history for version tracking"""
        versions = []
        
        try:
            import subprocess
            
            # Get git log with version-related commits
            result = subprocess.run([
                'git', 'log', '--oneline', '--grep=v[0-9]', '--grep=version', 
                '--grep=feat:', '--grep=fix:', '--since=30 days ago'
            ], cwd=self.svod_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        commit_hash, message = line.split(' ', 1)
                        
                        # Extract version if present
                        version_match = re.search(r'v?(\d+\.\d+\.\d+)', message)
                        version = version_match.group(1) if version_match else None
                        
                        versions.append({
                            'commit_hash': commit_hash,
                            'message': message,
                            'version': version,
                            'type': self._classify_commit_type(message)
                        })
                        
        except Exception as e:
            print(f"Warning: Could not load git history: {e}")
        
        return versions
    
    def _classify_commit_type(self, message: str) -> str:
        """Classify commit type based on message"""
        message_lower = message.lower()
        
        if 'feat:' in message_lower or 'feature' in message_lower:
            return 'feature'
        elif 'fix:' in message_lower or 'bug' in message_lower:
            return 'bugfix'
        elif 'yolo' in message_lower:
            return 'model_update'
        elif 'performance' in message_lower or 'speed' in message_lower:
            return 'performance'
        elif 'accuracy' in message_lower:
            return 'accuracy_improvement'
        else:
            return 'other'
    
    def generate_accuracy_trend_report(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate accuracy trend analysis"""
        print("📈 Generating accuracy trend report...")
        
        accuracy_data = []
        
        # Collect accuracy data from various sources
        for stat in historical_data['statistics']:
            if stat['accuracy']:
                accuracy_data.append({
                    'version': stat['version'],
                    'timestamp': stat['timestamp'],
                    'accuracy': stat['accuracy'],
                    'source': 'database'
                })
        
        for benchmark in historical_data['benchmarks']:
            if benchmark.get('success_rate'):
                accuracy_data.append({
                    'version': benchmark.get('svod_version', 'unknown'),
                    'timestamp': benchmark.get('timestamp'),
                    'accuracy': benchmark['success_rate'],
                    'source': 'benchmark'
                })
        
        # Sort by timestamp
        accuracy_data.sort(key=lambda x: x['timestamp'])
        
        report = {
            'data_points': len(accuracy_data),
            'accuracy_timeline': accuracy_data,
            'summary': {}
        }
        
        if accuracy_data:
            accuracies = [d['accuracy'] for d in accuracy_data]
            report['summary'] = {
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies),
                'latest_accuracy': accuracies[-1],
                'first_accuracy': accuracies[0],
                'total_improvement': accuracies[-1] - accuracies[0],
                'avg_accuracy': sum(accuracies) / len(accuracies),
                'trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining' if accuracies[-1] < accuracies[0] else 'stable'
            }
        
        return report
    
    def generate_performance_trend_report(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance trend analysis"""
        print("⏱️ Generating performance trend report...")
        
        performance_data = []
        
        # Collect performance data
        for stat in historical_data['statistics']:
            if stat['avg_processing_time']:
                performance_data.append({
                    'version': stat['version'],
                    'timestamp': stat['timestamp'],
                    'avg_time': stat['avg_processing_time'],
                    'source': 'database'
                })
        
        for benchmark in historical_data['benchmarks']:
            avg_time = benchmark.get('summary', {}).get('avg_execution_time')
            if avg_time:
                performance_data.append({
                    'version': benchmark.get('svod_version', 'unknown'),
                    'timestamp': benchmark.get('timestamp'),
                    'avg_time': avg_time,
                    'source': 'benchmark'
                })
        
        # Sort by timestamp
        performance_data.sort(key=lambda x: x['timestamp'])
        
        report = {
            'data_points': len(performance_data),
            'performance_timeline': performance_data,
            'summary': {}
        }
        
        if performance_data:
            times = [d['avg_time'] for d in performance_data]
            report['summary'] = {
                'min_time': min(times),
                'max_time': max(times),
                'latest_time': times[-1],
                'first_time': times[0],
                'speed_improvement': times[0] - times[-1],  # Positive = faster
                'avg_time': sum(times) / len(times),
                'trend': 'faster' if times[-1] < times[0] else 'slower' if times[-1] > times[0] else 'stable'
            }
        
        return report
    
    def generate_yolo_evolution_report(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate YOLO version evolution analysis"""
        print("🎯 Generating YOLO evolution report...")
        
        yolo_usage = defaultdict(list)
        yolo_timeline = []
        
        # Collect YOLO version usage data
        for stat in historical_data['statistics']:
            if stat['yolo_version'] and stat['yolo_version'] != 'unknown':
                yolo_usage[stat['yolo_version']].append({
                    'version': stat['version'],
                    'timestamp': stat['timestamp'],
                    'accuracy': stat['accuracy']
                })
                yolo_timeline.append({
                    'timestamp': stat['timestamp'],
                    'version': stat['version'],
                    'yolo_version': stat['yolo_version']
                })
        
        for benchmark in historical_data['benchmarks']:
            yolo_versions = benchmark.get('summary', {}).get('yolo_version_usage', {})
            if yolo_versions:
                timestamp = benchmark.get('timestamp')
                svod_version = benchmark.get('svod_version', 'unknown')
                
                for yolo_ver, count in yolo_versions.items():
                    yolo_usage[yolo_ver].append({
                        'version': svod_version,
                        'timestamp': timestamp,
                        'usage_count': count
                    })
                    yolo_timeline.append({
                        'timestamp': timestamp,
                        'version': svod_version,
                        'yolo_version': yolo_ver,
                        'usage_count': count
                    })
        
        # Sort timeline
        yolo_timeline.sort(key=lambda x: x['timestamp'])
        
        report = {
            'yolo_versions_tracked': list(yolo_usage.keys()),
            'timeline': yolo_timeline,
            'version_analysis': {},
            'transition_points': []
        }
        
        # Analyze each YOLO version
        for yolo_ver, data_points in yolo_usage.items():
            if data_points:
                accuracies = [d['accuracy'] for d in data_points if d.get('accuracy')]
                report['version_analysis'][yolo_ver] = {
                    'usage_periods': len(data_points),
                    'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else None,
                    'first_seen': min(d['timestamp'] for d in data_points),
                    'last_seen': max(d['timestamp'] for d in data_points)
                }
        
        # Detect transition points
        prev_yolo = None
        for entry in yolo_timeline:
            if prev_yolo and prev_yolo != entry['yolo_version']:
                report['transition_points'].append({
                    'timestamp': entry['timestamp'],
                    'from_version': prev_yolo,
                    'to_version': entry['yolo_version'],
                    'svod_version': entry['version']
                })
            prev_yolo = entry['yolo_version']
        
        return report
    
    def generate_detection_method_evolution(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detection method usage evolution analysis"""
        print("🔍 Generating detection method evolution report...")
        
        method_timeline = []
        method_stats = defaultdict(list)
        
        # Collect detection method data
        for stat in historical_data['statistics']:
            if stat['detection_methods']:
                timestamp = stat['timestamp']
                version = stat['version']
                
                for method, count in stat['detection_methods'].items():
                    method_timeline.append({
                        'timestamp': timestamp,
                        'version': version,
                        'method': method,
                        'usage_count': count
                    })
                    method_stats[method].append({
                        'timestamp': timestamp,
                        'version': version,
                        'count': count
                    })
        
        # Sort timeline
        method_timeline.sort(key=lambda x: x['timestamp'])
        
        report = {
            'methods_tracked': list(method_stats.keys()),
            'timeline': method_timeline,
            'method_analysis': {},
            'usage_trends': {}
        }
        
        # Analyze each detection method
        for method, data_points in method_stats.items():
            if data_points:
                counts = [d['count'] for d in data_points]
                report['method_analysis'][method] = {
                    'total_usage_points': len(data_points),
                    'avg_usage': sum(counts) / len(counts),
                    'max_usage': max(counts),
                    'min_usage': min(counts),
                    'first_seen': min(d['timestamp'] for d in data_points),
                    'last_seen': max(d['timestamp'] for d in data_points),
                    'trend': 'increasing' if counts[-1] > counts[0] else 'decreasing' if counts[-1] < counts[0] else 'stable'
                }
        
        return report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evolution report"""
        print(f"\n📋 === SVOD Evolution Report Generator v{__version__} ===")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load all historical data
        historical_data = self.load_historical_data()
        
        # Generate individual reports
        comprehensive_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': __version__,
                'data_sources': {
                    'statistics_db': self.db_path.exists(),
                    'benchmark_files': len(historical_data['benchmarks']),
                    'comparison_files': len(historical_data['comparisons']),
                    'git_commits': len(historical_data['versions'])
                }
            },
            'accuracy_trends': self.generate_accuracy_trend_report(historical_data),
            'performance_trends': self.generate_performance_trend_report(historical_data),
            'yolo_evolution': self.generate_yolo_evolution_report(historical_data),
            'detection_methods': self.generate_detection_method_evolution(historical_data),
            'executive_summary': {}
        }
        
        # Generate executive summary
        comprehensive_report['executive_summary'] = self._generate_executive_summary(comprehensive_report)
        
        return comprehensive_report
    
    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of key findings"""
        summary = {
            'key_findings': [],
            'recommendations': [],
            'data_quality': 'good'
        }
        
        # Accuracy findings
        acc_summary = report['accuracy_trends'].get('summary', {})
        if acc_summary:
            latest_acc = acc_summary.get('latest_accuracy', 0)
            improvement = acc_summary.get('total_improvement', 0)
            
            summary['key_findings'].append(f"Current accuracy: {latest_acc:.1f}%")
            
            if improvement > 0:
                summary['key_findings'].append(f"Accuracy improved by {improvement:.1f}% over time")
            elif improvement < 0:
                summary['key_findings'].append(f"Accuracy declined by {abs(improvement):.1f}% over time")
        
        # Performance findings
        perf_summary = report['performance_trends'].get('summary', {})
        if perf_summary:
            latest_time = perf_summary.get('latest_time', 0)
            speed_improvement = perf_summary.get('speed_improvement', 0)
            
            summary['key_findings'].append(f"Current avg processing time: {latest_time:.2f}s")
            
            if speed_improvement > 0:
                summary['key_findings'].append(f"Processing speed improved by {speed_improvement:.2f}s")
            elif speed_improvement < 0:
                summary['key_findings'].append(f"Processing speed decreased by {abs(speed_improvement):.2f}s")
        
        # YOLO evolution findings
        yolo_versions = report['yolo_evolution'].get('yolo_versions_tracked', [])
        if yolo_versions:
            summary['key_findings'].append(f"YOLO versions used: {', '.join(yolo_versions)}")
            
            transitions = report['yolo_evolution'].get('transition_points', [])
            if transitions:
                latest_transition = transitions[-1]
                summary['key_findings'].append(
                    f"Latest YOLO transition: {latest_transition['from_version']} → {latest_transition['to_version']}"
                )
        
        # Generate recommendations
        if acc_summary.get('trend') == 'declining':
            summary['recommendations'].append("Investigate accuracy regression causes")
        
        if perf_summary.get('trend') == 'slower':
            summary['recommendations'].append("Optimize processing performance")
        
        data_points = sum([
            report['accuracy_trends'].get('data_points', 0),
            report['performance_trends'].get('data_points', 0)
        ])
        
        if data_points < 5:
            summary['data_quality'] = 'limited'
            summary['recommendations'].append("Collect more historical data for better trend analysis")
        
        return summary
    
    def save_report(self, report: Dict[str, Any], output_file: Optional[Path] = None):
        """Save evolution report to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.svod_root / f"svod_evolution_report_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"💾 Evolution report saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the evolution report"""
        print(f"\n📊 === SVOD Evolution Summary ===")
        
        summary = report.get('executive_summary', {})
        
        if summary.get('key_findings'):
            print("\n🔍 Key Findings:")
            for finding in summary['key_findings']:
                print(f"  • {finding}")
        
        if summary.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        # Data sources summary
        metadata = report.get('metadata', {})
        sources = metadata.get('data_sources', {})
        print(f"\n📈 Data Sources:")
        print(f"  • Statistics DB: {'✅' if sources.get('statistics_db') else '❌'}")
        print(f"  • Benchmark files: {sources.get('benchmark_files', 0)}")
        print(f"  • Comparison files: {sources.get('comparison_files', 0)}")
        print(f"  • Git commits: {sources.get('git_commits', 0)}")
        
        quality = summary.get('data_quality', 'unknown')
        quality_icon = '✅' if quality == 'good' else '⚠️' if quality == 'limited' else '❌'
        print(f"  • Data quality: {quality_icon} {quality}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="SVOD Evolution Report Generator")
    parser.add_argument("--svod-root", type=Path, default=Path.cwd(),
                       help="Path to SVOD root directory")
    parser.add_argument("--output", type=Path,
                       help="Output file for report")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary to console")
    parser.add_argument("--save", action="store_true",
                       help="Save full report to file")
    
    args = parser.parse_args()
    
    reporter = SVODEvolutionReporter(args.svod_root)
    report = reporter.generate_comprehensive_report()
    
    if args.summary or not (args.save or args.output):
        reporter.print_summary(report)
    
    if args.save or args.output:
        reporter.save_report(report, args.output)

if __name__ == "__main__":
    main()