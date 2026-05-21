"""
SVOD Statistics Mixin

Provides statistics tracking, reporting, and voting stats methods
for the OrientationDetector class.
"""

from typing import Dict


class StatisticsMixin:
    """Mixin providing statistics management for OrientationDetector."""

    def _init_stats(self) -> Dict:
        """Return a fresh statistics dictionary."""
        return {
            "total_frames": 0,
            "frames_processed": 0,
            "frames_with_humans": 0,
            "correct_orientation_frames": 0,
            "incorrect_orientation_frames": 0,
            "uncertain_frames": 0,
            "face_detections": 0,
            "body_detections": 0,
            "close_up_frames": 0,
            "analyzed_duration": 0.0,
            "video_duration": 0.0,
            "mobilenet_votes": 0,
            "hough_votes": 0,
            "aspect_votes": 0,
            "conflict_resolutions": 0,
            "face_correct_votes": 0,
            "face_incorrect_votes": 0,
            "body_correct_votes": 0,
            "body_incorrect_votes": 0,
            "pose_detections": 0,
            "forced_incorrect_frames": 0,
            "forced_landscape_portrait_incorrect": 0,
            "rotation_directions": [],
            "rotation_strengths": [],
            "rotation_left_positions": 0,
            "rotation_right_positions": 0,
            "rotation_position_samples": 0,
            "face_mesh_detections": 0,
            "face_mesh_votes": 0,
            "cascade_skipped_haar": 0,
            "cascade_skipped_tier3": 0,
        }

    def reset_statistics(self):
        """Alias for reset_stats() - reset statistics for new video processing"""
        self.reset_stats()

    def reset_stats(self):
        """Reset statistics for new video processing"""
        self.stats = self._init_stats()

    def get_statistics(self) -> Dict:
        """Get current statistics from the detector."""
        return self.stats.copy()

    def get_statistics_report(self) -> str:
        """Generate a formatted statistics report."""
        stats = self.stats

        report_lines = [
            "VIDEO ORIENTATION DETECTOR STATISTICS REPORT",
            "=" * 50,
            f"Total frames analyzed: {stats['total_frames']}",
            f"Frames with humans: {stats['frames_with_humans']}",
            f"Correct orientation frames: {stats['correct_orientation_frames']}",
            f"Incorrect orientation frames: {stats['incorrect_orientation_frames']}",
            f"Uncertain frames: {stats['uncertain_frames']}",
            f"Face detections: {stats['face_detections']}",
            f"Body detections: {stats['body_detections']}",
            f"Pose detections: {stats.get('pose_detections', 0)}",
            f"Close-up frames: {stats['close_up_frames']}",
            f"Analyzed duration: {stats['analyzed_duration']:.1f}s",
            f"Video duration: {stats['video_duration']:.1f}s",
            f"MobileNet votes: {stats['mobilenet_votes']}",
            f"Hough line votes: {stats['hough_votes']}",
            f"Aspect ratio votes: {stats['aspect_votes']}",
            f"Conflict resolutions: {stats['conflict_resolutions']}",
            f"Face correct votes: {stats['face_correct_votes']}",
            f"Face incorrect votes: {stats['face_incorrect_votes']}",
            f"Body correct votes: {stats['body_correct_votes']}",
            f"Body incorrect votes: {stats['body_incorrect_votes']}",
            f"Forced incorrect frames: {stats.get('forced_incorrect_frames', 0)}",
            f"Forced landscape portrait incorrect: {stats.get('forced_landscape_portrait_incorrect', 0)}",
            "",
            "ADAPTIVE CASCADE:",
            f"Haar cascade skipped: {stats.get('cascade_skipped_haar', 0)}",
            f"Tier-3 models skipped: {stats.get('cascade_skipped_tier3', 0)}",
        ]

        if stats["frames_with_humans"] > 0:
            correct_pct = (stats["correct_orientation_frames"] / stats["frames_with_humans"]) * 100
            incorrect_pct = (
                stats["incorrect_orientation_frames"] / stats["frames_with_humans"]
            ) * 100
            uncertain_pct = (stats["uncertain_frames"] / stats["frames_with_humans"]) * 100

            report_lines.extend(
                [
                    "",
                    "PERCENTAGES:",
                    f"Correct orientation: {correct_pct:.1f}%",
                    f"Incorrect orientation: {incorrect_pct:.1f}%",
                    f"Uncertain: {uncertain_pct:.1f}%",
                ]
            )

        if stats["total_frames"] > 0:
            close_up_pct = (stats["close_up_frames"] / stats["total_frames"]) * 100
            report_lines.append(f"Close-up percentage: {close_up_pct:.1f}%")

        return "\n".join(report_lines)

    def _update_voting_stats(self, votes: Dict):
        """Update face and body voting statistics for balanced weighting."""
        face_correct = votes["face"].count("correct")
        face_incorrect = votes["face"].count("incorrect")
        self.stats["face_correct_votes"] += face_correct
        self.stats["face_incorrect_votes"] += face_incorrect

        body_correct = votes["yolo"].count("correct")
        body_incorrect = votes["yolo"].count("incorrect")
        self.stats["body_correct_votes"] += body_correct
        self.stats["body_incorrect_votes"] += body_incorrect
