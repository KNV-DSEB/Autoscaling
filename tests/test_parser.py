"""
Test Parser Module
==================
Unit tests cho NASALogParser.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.parser import NASALogParser


class TestNASALogParser:
    """Test cases cho NASALogParser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return NASALogParser()

    def test_parse_valid_line(self, parser):
        """Test parsing một dòng log hợp lệ."""
        line = '199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245'
        result = parser.parse_line(line)

        assert result is not None
        assert result['host'] == '199.72.81.55'
        assert result['method'] == 'GET'
        assert result['url'] == '/history/apollo/'
        assert result['status'] == 200
        assert result['bytes'] == 6245

    def test_parse_line_with_dash_bytes(self, parser):
        """Test parsing dòng log với bytes = '-'."""
        line = 'unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 -'
        result = parser.parse_line(line)

        assert result is not None
        assert result['bytes'] == 0  # "-" should be converted to 0

    def test_parse_invalid_line(self, parser):
        """Test parsing dòng log không hợp lệ."""
        invalid_lines = [
            '',
            'invalid line',
            'partial 199.72.81.55 - -',
            None
        ]

        for line in invalid_lines:
            if line is not None:
                result = parser.parse_line(line)
                assert result is None

    def test_parse_timestamp(self, parser):
        """Test parsing timestamp."""
        line = '199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET / HTTP/1.0" 200 100'
        result = parser.parse_line(line)

        assert result is not None
        assert result['timestamp'].year == 1995
        assert result['timestamp'].month == 7
        assert result['timestamp'].day == 1
        assert result['timestamp'].hour == 0
        assert result['timestamp'].minute == 0
        assert result['timestamp'].second == 1

    def test_parse_different_methods(self, parser):
        """Test parsing các HTTP methods khác nhau."""
        methods = ['GET', 'POST', 'HEAD', 'PUT', 'DELETE']

        for method in methods:
            line = f'host - - [01/Jul/1995:00:00:01 -0400] "{method} /path HTTP/1.0" 200 100'
            result = parser.parse_line(line)

            assert result is not None
            assert result['method'] == method

    def test_parse_different_status_codes(self, parser):
        """Test parsing các status codes khác nhau."""
        status_codes = [200, 301, 302, 400, 404, 500, 503]

        for code in status_codes:
            line = f'host - - [01/Jul/1995:00:00:01 -0400] "GET /path HTTP/1.0" {code} 100'
            result = parser.parse_line(line)

            assert result is not None
            assert result['status'] == code

    def test_extract_url_components(self, parser):
        """Test extract URL components."""
        test_cases = [
            ('/history/apollo/', '/history/apollo/', 'apollo'),
            ('/shuttle/countdown/', '/shuttle/countdown/', 'countdown'),
            ('/images/NASA-logo.gif', '/images/', 'images'),
            ('/', '/', None),
        ]

        for url, expected_path, _ in test_cases:
            line = f'host - - [01/Jul/1995:00:00:01 -0400] "GET {url} HTTP/1.0" 200 100'
            result = parser.parse_line(line)

            assert result is not None
            assert result['url'] == url


class TestParserStatistics:
    """Test parser statistics."""

    @pytest.fixture
    def parser(self):
        return NASALogParser()

    def test_stats_tracking(self, parser):
        """Test tracking của parsing statistics."""
        lines = [
            '199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET / HTTP/1.0" 200 100',
            'invalid line',
            '199.72.81.55 - - [01/Jul/1995:00:00:02 -0400] "GET / HTTP/1.0" 200 100',
        ]

        for line in lines:
            parser.parse_line(line)

        stats = parser.get_statistics()
        assert stats['total_lines'] == 3
        assert stats['valid_lines'] == 2
        assert stats['invalid_lines'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
