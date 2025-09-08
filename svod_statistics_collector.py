"""
SVOD Statistics Collector and Historical Analysis
Advanced statistics collection for SVOD version evolution tracking

Version: 1.0.0
Date: September 8, 2025
Author: SVOD Development Team

Features:
- Historical version performance tracking
- YOLOv8 vs YOLOv4 usage statistics
- Detection method effectiveness analysis
- Cross-platform performance comparison
- Regression detection and improvement validation
- Model precision and recall tracking
"""

import csv
import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import sqlite3

__version__ = "1.0.0"
__release_date__ = "2025-09-08"

class SVODStatisticsCollector:
    """Comprehensive statistics collector for SVOD evolution analysis"""
    
    def __init__(self, svod_root: Path):
        self.svod_root = Path(svod_root)
        self.db_path = self.svod_root / "svod_statistics.db"
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database for historical tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Version tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS version_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                release_date TEXT,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                avg_processing_time REAL,
                total_videos INTEGER,
                total_frames INTEGER,
                yolo_version TEXT,
                notes TEXT
            )
        ''')
        
        # Detection method usage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id INTEGER,
                method_name TEXT NOT NULL,
                usage_count INTEGER,
                success_rate REAL,
                avg_confidence REAL,
                FOREIGN KEY (version_id) REFERENCES version_stats (id)
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id INTEGER,
                model_name TEXT NOT NULL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                false_positive_rate REAL,
                processing_time_ms REAL,
                FOREIGN KEY (version_id) REFERENCES version_stats (id)
            )
        ''')
        
        # Video analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id INTEGER,
                video_name TEXT NOT NULL,
                expected_orientation TEXT,
                predicted_orientation TEXT,
                is_correct INTEGER,
                analysis_time REAL,
                confidence_score REAL,
                detection_methods TEXT,
                FOREIGN KEY (version_id) REFERENCES version_stats (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def extract_version_from_script(self) -> Tuple[str, str, str]:
        """Extract version information from SVOD script"""
        script_path = self.svod_root / "video_orientation_detector.py"
        version = "unknown"
        release_date = "unknown"
        release_name = "unknown"
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract version
            version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                version = version_match.group(1)
                
            # Extract release date
            date_match = re.search(r'__release_date__\s*=\s*["\']([^"\']+)["\']', content)
            if date_match:
                release_date = date_match.group(1)
                
            # Extract release name
            name_match = re.search(r'__release_name__\s*=\s*["\']([^"\']+)["\']', content)
            if name_match:
                release_name = name_match.group(1)
                
        except Exception as e:
            print(f"Warning: Could not extract version info: {e}")
            
        return version, release_date, release_name
    
    def collect_current_statistics(self) -> Dict[str, Any]:
        """Collect statistics from latest SVOD run"""
        stats = {
            'version_info': self.extract_version_from_script(),
            'accuracy': None,
            'performance': {},
            'detection_usage': defaultdict(int),
            'model_stats': defaultdict(dict),
            'video_results': [],
            'yolo_version': 'unknown'
        }
        
        # Find latest result files
        batch_files = sorted(
            self.svod_root.glob("batch_report_*.txt"), 
            key=lambda p: p.stat().st_mtime, 
            reverse=True
        )
        
        detailed_files = sorted(
            self.svod_root.glob("detailed_votes_*.csv"), 
            key=lambda p: p.stat().st_mtime, 
            reverse=True
        )
        
        speed_files = sorted(
            self.svod_root.glob("speed_results_*.csv"), 
            key=lambda p: p.stat().st_mtime, 
            reverse=True
        )
        
        # Extract accuracy from batch report
        if batch_files:
            stats['accuracy'] = self._extract_accuracy_from_batch(batch_files[0])
            
        # Extract detailed detection statistics
        if detailed_files:
            detailed_stats = self._analyze_detailed_votes(detailed_files[0])
            stats['detection_usage'] = detailed_stats['model_usage']
            stats['yolo_version'] = detailed_stats['yolo_version']
            stats['video_results'] = detailed_stats['video_results']
            
        # Extract performance statistics
        if speed_files:
            stats['performance'] = self._analyze_speed_results(speed_files[0])
            
        return stats
    
    def _extract_accuracy_from_batch(self, batch_file: Path) -> Optional[float]:
        """Extract accuracy from batch report"""
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for accuracy pattern
            accuracy_match = re.search(r'Accuracy:\s*(\d+\.?\d*)%', content)
            if accuracy_match:
                return float(accuracy_match.group(1))
                
            # Alternative pattern
            accuracy_match = re.search(r'(\d+)/(\d+)\s*\((\d+\.?\d*)%\)', content)
            if accuracy_match:
                return float(accuracy_match.group(3))
                
        except Exception as e:
            print(f"Warning: Could not extract accuracy from {batch_file}: {e}")
            
        return None
    
    def _analyze_detailed_votes(self, detailed_file: Path) -> Dict[str, Any]:
        """Analyze detailed votes for detection statistics"""
        stats = {
            'model_usage': defaultdict(int),
            'yolo_version': 'unknown',
            'video_results': [],
            'total_frames': 0
        }
        
        try:
            with open(detailed_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                current_video = None
                video_votes = []
                
                for row in reader:
                    stats['total_frames'] += 1
                    video = row.get('Video', 'unknown')
                    
                    # Track model usage
                    for model in ['Face', 'YOLO', 'MobileNet', 'Hough', 'Aspect']:
                        if row.get(model) and row[model].strip():
                            stats['model_usage'][model] += 1
                    
                    # Track YOLO version
                    if 'YOLOVersion' in row and row['YOLOVersion']:
                        stats['yolo_version'] = row['YOLOVersion']
                    
                    # Collect video-level results
                    if video != current_video:
                        if current_video and video_votes:
                            # Process previous video
                            self._process_video_votes(current_video, video_votes, stats)
                        current_video = video
                        video_votes = []
                    
                    video_votes.append(row)
                
                # Process last video
                if current_video and video_votes:
                    self._process_video_votes(current_video, video_votes, stats)
                    
        except Exception as e:
            print(f"Warning: Could not analyze detailed votes from {detailed_file}: {e}")
            
        return stats
    
    def _process_video_votes(self, video: str, votes: List[Dict], stats: Dict):
        """Process votes for a single video"""
        # Determine final prediction (most common vote)
        predictions = [vote.get('FinalPrediction', '') for vote in votes if vote.get('FinalPrediction')]
        if predictions:
            final_prediction = Counter(predictions).most_common(1)[0][0]
            
            # Calculate confidence (consistency of votes)
            confidence = predictions.count(final_prediction) / len(predictions) if predictions else 0
            
            # Extract expected orientation from reference
            expected = self._get_expected_orientation(video)
            is_correct = (final_prediction.upper() == expected.upper()) if expected else None
            
            stats['video_results'].append({
                'video_name': video,
                'expected_orientation': expected,
                'predicted_orientation': final_prediction,
                'is_correct': is_correct,
                'confidence_score': confidence,
                'total_votes': len(votes)
            })
    
    def _get_expected_orientation(self, video: str) -> Optional[str]:
        """Get expected orientation from reference file"""
        ref_file = self.svod_root / "reference_orientations.csv"
        if not ref_file.exists():
            return None
            
        try:
            with open(ref_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Video') == video:
                        return row.get('ExpectedOrientation', '')
        except Exception:
            pass
            
        return None
    
    def _analyze_speed_results(self, speed_file: Path) -> Dict[str, Any]:
        """Analyze speed performance results"""
        performance = {
            'avg_processing_time': 0,
            'total_videos': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'per_video_times': []
        }
        
        try:
            with open(speed_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                times = []
                for row in reader:
                    if 'AnalysisTimeSec' in row and row['AnalysisTimeSec']:
                        try:
                            time_sec = float(row['AnalysisTimeSec'])
                            times.append(time_sec)
                            performance['min_time'] = min(performance['min_time'], time_sec)
                            performance['max_time'] = max(performance['max_time'], time_sec)
                            performance['per_video_times'].append({
                                'video': row.get('Video', ''),
                                'time': time_sec
                            })
                        except ValueError:
                            continue
                
                if times:
                    performance['avg_processing_time'] = sum(times) / len(times)
                    performance['total_videos'] = len(times)
                    performance['min_time'] = performance['min_time'] if performance['min_time'] != float('inf') else 0
                    
        except Exception as e:
            print(f"Warning: Could not analyze speed results from {speed_file}: {e}")
            
        return performance
    
    def store_statistics(self, stats: Dict[str, Any]) -> int:
        """Store collected statistics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert version statistics
            version, release_date, release_name = stats['version_info']
            cursor.execute('''
                INSERT INTO version_stats 
                (version, release_date, timestamp, accuracy, avg_processing_time, 
                 total_videos, total_frames, yolo_version, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version,
                release_date,
                datetime.now().isoformat(),
                stats['accuracy'],
                stats['performance'].get('avg_processing_time'),
                stats['performance'].get('total_videos', 0),
                sum(stats['detection_usage'].values()),
                stats['yolo_version'],
                release_name
            ))
            
            version_id = cursor.lastrowid
            
            # Insert detection usage statistics
            for method, count in stats['detection_usage'].items():
                cursor.execute('''
                    INSERT INTO detection_usage (version_id, method_name, usage_count)
                    VALUES (?, ?, ?)
                ''', (version_id, method, count))
            
            # Insert video results
            for result in stats['video_results']:
                cursor.execute('''
                    INSERT INTO video_results 
                    (version_id, video_name, expected_orientation, predicted_orientation,
                     is_correct, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    version_id,
                    result['video_name'],
                    result['expected_orientation'],
                    result['predicted_orientation'],
                    1 if result['is_correct'] else 0,
                    result['confidence_score']
                ))
            
            conn.commit()
            print(f"✅ Statistics stored for version {version} (ID: {version_id})")
            return version_id
            
        except Exception as e:
            print(f"❌ Error storing statistics: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def generate_evolution_report(self) -> Dict[str, Any]:
        """Generate comprehensive evolution report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'version_history': [],
            'accuracy_trend': [],
            'speed_trend': [],
            'model_evolution': {},
            'summary': {}
        }
        
        try:
            # Get version history
            cursor.execute('''
                SELECT version, release_date, timestamp, accuracy, avg_processing_time,
                       total_videos, yolo_version, notes
                FROM version_stats
                ORDER BY timestamp
            ''')
            
            for row in cursor.fetchall():
                version_data = {
                    'version': row[0],
                    'release_date': row[1],
                    'timestamp': row[2],
                    'accuracy': row[3],
                    'avg_processing_time': row[4],
                    'total_videos': row[5],
                    'yolo_version': row[6],
                    'notes': row[7]
                }
                report['version_history'].append(version_data)
                
                if version_data['accuracy']:
                    report['accuracy_trend'].append({
                        'version': version_data['version'],
                        'accuracy': version_data['accuracy']
                    })
                    
                if version_data['avg_processing_time']:
                    report['speed_trend'].append({
                        'version': version_data['version'],
                        'avg_time': version_data['avg_processing_time']
                    })
            
            # Calculate summary statistics
            if report['accuracy_trend']:
                accuracies = [item['accuracy'] for item in report['accuracy_trend']]
                report['summary']['accuracy_range'] = {
                    'min': min(accuracies),
                    'max': max(accuracies),
                    'latest': accuracies[-1],
                    'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
                }
            
            if report['speed_trend']:
                times = [item['avg_time'] for item in report['speed_trend']]
                report['summary']['speed_range'] = {
                    'min': min(times),
                    'max': max(times),
                    'latest': times[-1],
                    'improvement': times[0] - times[-1] if len(times) > 1 else 0  # Negative means slower
                }
                
        except Exception as e:
            print(f"❌ Error generating evolution report: {e}")
        finally:
            conn.close()
            
        return report
    
    def print_evolution_summary(self):
        """Print a summary of SVOD evolution"""
        report = self.generate_evolution_report()
        
        print(f"\n🔬 === SVOD Evolution Summary ===")
        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if report['version_history']:
            print(f"\n📈 Version History ({len(report['version_history'])} versions tracked):")
            for version in report['version_history'][-5:]:  # Show last 5 versions
                print(f"  v{version['version']}: {version['notes']} ({version['timestamp'][:10]})")
        
        if 'accuracy_range' in report['summary']:
            acc = report['summary']['accuracy_range']
            print(f"\n🎯 Accuracy Evolution:")
            print(f"  Range: {acc['min']:.1f}% - {acc['max']:.1f}%")
            print(f"  Latest: {acc['latest']:.1f}%")
            print(f"  Total improvement: {acc['improvement']:+.1f}%")
        
        if 'speed_range' in report['summary']:
            speed = report['summary']['speed_range']
            print(f"\n⏱️ Speed Evolution:")
            print(f"  Range: {speed['min']:.2f}s - {speed['max']:.2f}s")
            print(f"  Latest: {speed['latest']:.2f}s")
            if speed['improvement'] > 0:
                print(f"  Speed improvement: {speed['improvement']:.2f}s faster")
            elif speed['improvement'] < 0:
                print(f"  Speed change: {abs(speed['improvement']):.2f}s slower")
            else:
                print(f"  Speed: No significant change")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="SVOD Statistics Collector")
    parser.add_argument("--svod-root", type=Path, default=Path.cwd(),
                       help="Path to SVOD root directory")
    parser.add_argument("--collect", action="store_true",
                       help="Collect and store current statistics")
    parser.add_argument("--report", action="store_true",
                       help="Generate evolution report")
    parser.add_argument("--summary", action="store_true",
                       help="Print evolution summary")
    
    args = parser.parse_args()
    
    collector = SVODStatisticsCollector(args.svod_root)
    
    if args.collect:
        stats = collector.collect_current_statistics()
        collector.store_statistics(stats)
    
    if args.report:
        report = collector.generate_evolution_report()
        output_file = args.svod_root / f"svod_evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"💾 Evolution report saved to: {output_file}")
    
    if args.summary:
        collector.print_evolution_summary()
    
    # Default behavior: show summary
    if not any([args.collect, args.report, args.summary]):
        collector.print_evolution_summary()

if __name__ == "__main__":
    main()