"""Tests for docs.py - Documentation generation utilities."""

import json

import pytest

from tacotoolbox._exceptions import TacoDocumentationError
from tacotoolbox.docs import (
    _get_color_variants,
    _simple_markdown_to_html,
    generate_html,
    generate_markdown,
)


class TestGetColorVariants:

    def test_standard_hex_with_hash(self):
        result = _get_color_variants("#4CAF50")
        assert result["primary"] == "#4CAF50"
        assert result["dark"] == "#39833C"
        assert result["light"] == "#C9E7CA"

    def test_hex_without_hash(self):
        result = _get_color_variants("FF5722")
        assert result["primary"] == "#FF5722"

    def test_black_dark_stays_black(self):
        result = _get_color_variants("#000000")
        assert result["dark"] == "#000000"

    def test_white_light_stays_white(self):
        result = _get_color_variants("#FFFFFF")
        assert result["light"] == "#FFFFFF"

    def test_invalid_length_returns_fallback(self):
        result = _get_color_variants("#FFF")
        assert result["primary"] == "#4CAF50"

    def test_lowercase_normalized_to_upper(self):
        result = _get_color_variants("abcdef")
        assert result["primary"] == "#ABCDEF"


class TestSimpleMarkdownToHtml:

    def test_xss_escaped(self):
        malicious = "<script>alert('xss')</script>"
        result = _simple_markdown_to_html(malicious)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_header_becomes_h4(self):
        result = _simple_markdown_to_html("# My Header")
        assert "<h4" in result
        assert "My Header" in result

    def test_list_items(self):
        result = _simple_markdown_to_html("- item one\n- item two")
        assert "<ul" in result
        assert "<li>item one</li>" in result
        assert "<li>item two</li>" in result

    def test_paragraphs_separated(self):
        result = _simple_markdown_to_html("First para\n\nSecond para")
        assert result.count("<p") == 2




class TestGenerateMarkdown:

    @pytest.fixture
    def valid_collection(self, tmp_path):
        path = tmp_path / "COLLECTION.json"
        path.write_text(json.dumps({
            "id": "test-dataset",
            "description": "A test dataset",
            "keywords": ["test", "fixture"],
        }))
        return path

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_markdown(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_documentation_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(TacoDocumentationError, match="Invalid JSON"):
            generate_markdown(bad)

    def test_generates_output_with_dataset_id(self, valid_collection, tmp_path):
        output = tmp_path / "README.md"
        generate_markdown(valid_collection, output)
        content = output.read_text()
        assert "test-dataset" in content


class TestGenerateHtml:

    @pytest.fixture
    def valid_collection(self, tmp_path):
        path = tmp_path / "COLLECTION.json"
        path.write_text(json.dumps({
            "id": "html-test",
            "description": "HTML test dataset",
            "keywords": [],
        }))
        return path

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_html(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_documentation_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("[]]]malformed")
        with pytest.raises(TacoDocumentationError, match="Invalid JSON"):
            generate_html(bad)

    def test_generates_valid_html_with_theme_color(self, valid_collection, tmp_path):
        output = tmp_path / "index.html"
        generate_html(valid_collection, output, theme_color="#FF5722")
        content = output.read_text()
        assert "<!DOCTYPE html>" in content or "<html" in content
        assert "html-test" in content

    def test_inline_deps_produces_larger_file(self, valid_collection, tmp_path):
        inline = tmp_path / "inline.html"
        cdn = tmp_path / "cdn.html"
        generate_html(valid_collection, inline, inline_deps=True)
        generate_html(valid_collection, cdn, inline_deps=False)
        # Inline embeds d3, leaflet, etc - should be significantly larger
        assert inline.stat().st_size > cdn.stat().st_size * 2