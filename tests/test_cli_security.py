"""Path checks that must not depend on the detection models."""

from svod_cli import path_traversal_error


def test_filename_with_double_dot_is_allowed():
    assert path_traversal_error("clip..final.mp4") is None


def test_parent_directory_component_is_rejected():
    assert path_traversal_error("videos/../secret.mp4") is not None


def test_overlong_dot_component_is_rejected():
    assert path_traversal_error("videos/.../secret.mp4") is not None


def test_normal_relative_path_is_allowed():
    assert path_traversal_error("videos/clip.mp4") is None
