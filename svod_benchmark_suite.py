"""
SVOD Automated Benchmark Suite
Comprehensive benchmarking system for SVOD version validation and performance testing

Version: 1.0.0
Date: September 8, 2025
Author: SVOD Development Team

Features:
- Automated testing across Windows PowerShell, WSL Linux, and macOS
- Version comparison with standardized test datasets
- Performance regression detection
- YOLOv8 vs YOLOv4 benchmarking
- Cross-platform compatibility validation
- Automated model download and cleanup for vanilla testing
"""

import os
import sys
import subprocess
import json
import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import platform
import csv

__version__ = "1.0.0"
__release_date__ = "2025-09-08"

class SVODBenchmarkSuite:
    """Automated benchmark suite for SVOD testing and validation"""
    
    def __init__(self, svod_root: Path, test_videos_dir: Optional[Path] = None):
        self.svod_root = Path(svod_root)
        self.test_videos_dir = test_videos_dir or Path("C:/Users/boris/Videos")
        self.script_path = self.svod_root / "video_orientation_detector.py"
        self.benchmark_results = {}
        self.platform_info = self._get_platform_info()
        
    def _get_platform_info(self) -> Dict[str, str]:
        """Get current platform information"""
        return {
            'system': platform.system(),
            'machine': platform.machine(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'is_wsl': self._is_wsl(),
            'shell': self._get_shell_type()
        }
    
    def _is_wsl(self) -> bool:
        """Check if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def _get_shell_type(self) -> str:
        """Determine shell type"""
        if platform.system() == "Windows":
            return "PowerShell" if "pwsh" in os.environ.get("SHELL", "") else "cmd"
        else:
            return os.environ.get("SHELL", "bash").split("/")[-1]
    
    def find_test_videos(self, max_videos: int = 5, max_duration: int = 5) -> List[Path]:
        """Find suitable test videos for benchmarking"""
        print(f"🎥 Searching for test videos in {self.test_videos_dir}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        test_videos = []
        
        if not self.test_videos_dir.exists():
            print(f"⚠️ Test videos directory not found: {self.test_videos_dir}")
            return []
        
        for video_file in self.test_videos_dir.iterdir():
            if video_file.suffix.lower() in video_extensions and len(test_videos) < max_videos:
                # Check video duration if possible
                duration = self._get_video_duration(video_file)
                if duration <= max_duration or duration == -1:  # -1 means couldn't determine
                    test_videos.append(video_file)
                    print(f"  ✅ Selected: {video_file.name} ({duration}s)")
                else:
                    print(f"  ⏭️ Skipped: {video_file.name} (too long: {duration}s)")
        
        print(f"📊 Found {len(test_videos)} suitable test videos")
        return test_videos
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe if available, otherwise return -1"""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(video_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return -1  # Unknown duration
    
    def clean_environment(self, preserve_reference: bool = True):
        """Clean SVOD environment for vanilla testing"""
        print("🧹 Cleaning SVOD environment for vanilla testing...")
        
        # Files to remove (model files that should be re-downloaded)
        files_to_remove = [
            "mobilenet-v2.xml",
            "mobilenet-v2.bin", 
            "mobilenet-v2.onnx",
            "res10_300x300_ssd_iter_140000.caffemodel",
            "deploy.prototxt",
            "yolov4.weights",
            "yolov4.cfg",
            "yolov8n.pt",
            "coco.names",
            "lbfmodel.yaml"
        ]
        
        # Directories to remove
        dirs_to_remove = [
            "public",
            "models",
            "__pycache__"
        ]
        
        # Result files to remove (unless preserving reference)
        result_patterns = [
            "orientation_results_*.csv",
            "detailed_votes_*.csv", 
            "speed_results_*.csv",
            "batch_report_*.txt",
            "batch_report_*.json",
            "annotated_*.mp4"
        ]
        
        removed_count = 0
        
        # Remove model files
        for file_name in files_to_remove:
            file_path = self.svod_root / file_name
            if file_path.exists():
                file_path.unlink()
                removed_count += 1
                print(f"  🗑️ Removed: {file_name}")
        
        # Remove directories
        for dir_name in dirs_to_remove:
            dir_path = self.svod_root / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                removed_count += 1
                print(f"  🗑️ Removed directory: {dir_name}")
        
        # Remove result files
        if not preserve_reference:
            for pattern in result_patterns:
                for file_path in self.svod_root.glob(pattern):
                    file_path.unlink()
                    removed_count += 1
                    print(f"  🗑️ Removed: {file_path.name}")
        
        print(f"✅ Environment cleaned ({removed_count} items removed)")
    
    def run_svod_test(self, video_path: Path, test_name: str = "", timeout: int = 300) -> Dict[str, Any]:
        """Run SVOD on a single video and collect results"""
        print(f"🎬 Testing SVOD with {video_path.name}")
        
        start_time = time.time()
        result = {
            'video_name': video_path.name,
            'test_name': test_name,
            'platform': self.platform_info,
            'start_time': datetime.now().isoformat(),
            'success': False,
            'execution_time': 0,
            'prediction': None,
            'confidence': None,
            'error_message': None,
            'model_downloads': [],
            'yolo_version': 'unknown'
        }
        
        try:
            # Run SVOD with the test video
            cmd = [sys.executable, str(self.script_path), str(video_path), "--max-seconds", "5"]
            
            process = subprocess.run(
                cmd,
                cwd=self.svod_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            
            if process.returncode == 0:
                result['success'] = True
                
                # Parse output for prediction and model info
                output_lines = process.stdout.split('\n')
                for line in output_lines:
                    if 'Final prediction:' in line:
                        result['prediction'] = line.split('Final prediction:')[1].strip()
                    elif 'YOLOv8' in line:
                        result['yolo_version'] = 'YOLOv8'
                    elif 'YOLOv4' in line:
                        result['yolo_version'] = 'YOLOv4'
                    elif 'Downloaded' in line or 'Installing' in line:
                        result['model_downloads'].append(line.strip())
                
                print(f"  ✅ Success: {result['prediction']} ({execution_time:.2f}s)")
            else:
                result['error_message'] = process.stderr
                print(f"  ❌ Failed: {process.stderr[:100]}...")
                
        except subprocess.TimeoutExpired:
            result['error_message'] = f"Timeout after {timeout} seconds"
            print(f"  ⏰ Timeout after {timeout}s")
        except Exception as e:
            result['error_message'] = str(e)
            print(f"  ❌ Error: {e}")
        
        return result
    
    def run_batch_benchmark(self, test_videos: List[Path], clean_before_test: bool = True) -> Dict[str, Any]:
        """Run batch benchmark with multiple videos"""
        print(f"\n🚀 Starting batch benchmark with {len(test_videos)} videos")
        
        if clean_before_test:
            self.clean_environment()
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'platform_info': self.platform_info,
            'svod_version': self._get_svod_version(),
            'test_videos': len(test_videos),
            'total_time': 0,
            'success_rate': 0,
            'individual_results': [],
            'summary': {}
        }
        
        start_time = time.time()
        successful_tests = 0
        
        for i, video_path in enumerate(test_videos, 1):
            print(f"\n--- Test {i}/{len(test_videos)} ---")
            
            test_result = self.run_svod_test(video_path, f"batch_test_{i}")
            benchmark_results['individual_results'].append(test_result)
            
            if test_result['success']:
                successful_tests += 1
        
        total_time = time.time() - start_time
        benchmark_results['total_time'] = total_time
        benchmark_results['success_rate'] = (successful_tests / len(test_videos)) * 100
        
        # Generate summary statistics
        benchmark_results['summary'] = self._generate_summary(benchmark_results)
        
        print(f"\n📊 Batch Benchmark Summary:")
        print(f"  Success Rate: {benchmark_results['success_rate']:.1f}% ({successful_tests}/{len(test_videos)})")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Time: {total_time/len(test_videos):.2f}s per video")
        
        return benchmark_results
    
    def _get_svod_version(self) -> str:
        """Extract SVOD version from script"""
        try:
            with open(self.script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import re
            version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                return version_match.group(1)
        except:
            pass
        
        return "unknown"
    
    def _generate_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results"""
        successful_results = [r for r in benchmark_results['individual_results'] if r['success']]
        
        summary = {
            'successful_tests': len(successful_results),
            'failed_tests': len(benchmark_results['individual_results']) - len(successful_results),
            'avg_execution_time': 0,
            'min_execution_time': 0,
            'max_execution_time': 0,
            'yolo_version_usage': {},
            'predictions': {},
            'model_downloads_needed': 0
        }
        
        if successful_results:
            execution_times = [r['execution_time'] for r in successful_results]
            summary['avg_execution_time'] = sum(execution_times) / len(execution_times)
            summary['min_execution_time'] = min(execution_times)
            summary['max_execution_time'] = max(execution_times)
            
            # Count YOLO version usage
            for result in successful_results:
                yolo_version = result.get('yolo_version', 'unknown')
                summary['yolo_version_usage'][yolo_version] = summary['yolo_version_usage'].get(yolo_version, 0) + 1
            
            # Count predictions
            for result in successful_results:
                prediction = result.get('prediction', 'unknown')
                summary['predictions'][prediction] = summary['predictions'].get(prediction, 0) + 1
        
        # Count tests that needed model downloads
        summary['model_downloads_needed'] = sum(1 for r in benchmark_results['individual_results'] if r['model_downloads'])
        
        return summary
    
    def save_benchmark_results(self, results: Dict[str, Any], output_file: Optional[Path] = None):
        """Save benchmark results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            platform_name = self.platform_info['system'].lower()
            output_file = self.svod_root / f"svod_benchmark_{platform_name}_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Benchmark results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving benchmark results: {e}")
    
    def compare_with_previous(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current benchmark with previous results"""
        print("\n🔍 Comparing with previous benchmark results...")
        
        # Find previous benchmark files
        platform_name = self.platform_info['system'].lower()
        benchmark_files = sorted(
            self.svod_root.glob(f"svod_benchmark_{platform_name}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        comparison = {
            'has_previous': False,
            'previous_file': None,
            'version_change': None,
            'success_rate_change': None,
            'speed_change': None,
            'yolo_usage_change': {}
        }
        
        if len(benchmark_files) > 1:  # Current + at least one previous
            previous_file = benchmark_files[1]  # Second most recent
            comparison['previous_file'] = previous_file.name
            
            try:
                with open(previous_file, 'r', encoding='utf-8') as f:
                    previous_results = json.load(f)
                
                comparison['has_previous'] = True
                
                # Compare versions
                current_version = current_results.get('svod_version', 'unknown')
                previous_version = previous_results.get('svod_version', 'unknown')
                comparison['version_change'] = f"{previous_version} → {current_version}"
                
                # Compare success rates
                current_success = current_results.get('success_rate', 0)
                previous_success = previous_results.get('success_rate', 0)
                comparison['success_rate_change'] = current_success - previous_success
                
                # Compare average execution times
                current_avg_time = current_results.get('summary', {}).get('avg_execution_time', 0)
                previous_avg_time = previous_results.get('summary', {}).get('avg_execution_time', 0)
                if previous_avg_time > 0:
                    comparison['speed_change'] = current_avg_time - previous_avg_time
                
                print(f"  📂 Previous: {previous_file.name}")
                print(f"  🔄 Version: {comparison['version_change']}")
                print(f"  🎯 Success Rate: {comparison['success_rate_change']:+.1f}%")
                if comparison['speed_change'] is not None:
                    print(f"  ⏱️ Speed: {comparison['speed_change']:+.2f}s")
                
            except Exception as e:
                print(f"  ⚠️ Could not load previous results: {e}")
        else:
            print("  ℹ️ No previous benchmark found for comparison")
        
        return comparison
    
    def run_cross_platform_test(self, test_videos: List[Path]) -> Dict[str, Any]:
        """Run tests optimized for current platform"""
        print(f"\n🌐 Running cross-platform test on {self.platform_info['system']}")
        
        # Adjust test parameters based on platform
        if self.platform_info['system'] == "Windows":
            print("  🪟 Windows PowerShell environment detected")
        elif self.platform_info['is_wsl']:
            print("  🐧 WSL Linux environment detected")
        elif self.platform_info['system'] == "Darwin":
            print("  🍎 macOS environment detected")
            if self.platform_info['machine'] == "arm64":
                print("    🚀 Apple Silicon (ARM64) detected")
        
        return self.run_batch_benchmark(test_videos, clean_before_test=True)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="SVOD Automated Benchmark Suite")
    parser.add_argument("--svod-root", type=Path, default=Path.cwd(),
                       help="Path to SVOD root directory")
    parser.add_argument("--test-videos", type=Path,
                       help="Path to test videos directory")
    parser.add_argument("--max-videos", type=int, default=5,
                       help="Maximum number of test videos")
    parser.add_argument("--clean", action="store_true",
                       help="Clean environment before testing")
    parser.add_argument("--no-clean", action="store_true",
                       help="Skip environment cleaning")
    parser.add_argument("--save-results", action="store_true",
                       help="Save benchmark results to file")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with previous results")
    
    args = parser.parse_args()
    
    benchmark = SVODBenchmarkSuite(args.svod_root, args.test_videos)
    
    # Find test videos
    test_videos = benchmark.find_test_videos(args.max_videos)
    if not test_videos:
        print("❌ No suitable test videos found")
        return
    
    # Determine cleaning strategy
    clean_before_test = not args.no_clean
    if args.clean:
        clean_before_test = True
    
    # Run benchmark
    results = benchmark.run_cross_platform_test(test_videos)
    
    # Save results if requested
    if args.save_results:
        benchmark.save_benchmark_results(results)
    
    # Compare with previous results if requested
    if args.compare:
        comparison = benchmark.compare_with_previous(results)

if __name__ == "__main__":
    main()