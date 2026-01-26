"""
NASA Log Parser
===============
Module parse NASA Common Log Format (CLF) web server logs.

Định dạng log:
    host - - [timestamp timezone] "method url protocol" status bytes

Ví dụ:
    199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245

Edge cases xử lý:
    - Bytes = "-" → Convert sang 0
    - Malformed lines → Skip và log lỗi
    - Encoding issues → Sử dụng latin-1
"""

import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class LogEntry:
    """Cấu trúc dữ liệu cho một log entry đã parse."""
    host: str
    timestamp: datetime
    timezone: str
    method: str
    url: str
    protocol: str
    status_code: int
    bytes_sent: int


class NASALogParser:
    """
    Parser cho NASA Common Log Format (CLF) web server logs.

    Xử lý các edge cases:
        - Bytes = "-" (không có response body)
        - Malformed entries
        - Timezone conversion

    Attributes:
        parse_errors (List): Danh sách các dòng lỗi
        stats (Dict): Thống kê parsing (total, success, failed)

    Usage:
        >>> parser = NASALogParser()
        >>> df = parser.parse_file('train.txt')
        >>> print(parser.stats)
    """

    # Regex pattern cho Common Log Format
    # Groups: host, timestamp, timezone, method, url, protocol, status, bytes
    LOG_PATTERN = re.compile(
        r'^(\S+)\s+-\s+-\s+\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})\s+([+-]\d{4})\]\s+'
        r'"(\w+)\s+(\S+)\s*([^"]*)"\s+(\d{3})\s+(\S+)$'
    )

    # Mapping tên tháng sang số
    MONTHS = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    def __init__(self):
        """Khởi tạo parser với stats rỗng."""
        self.parse_errors: List[Tuple[int, str]] = []
        self.stats = {'total': 0, 'success': 0, 'failed': 0}

    def reset_stats(self):
        """Reset thống kê về trạng thái ban đầu."""
        self.parse_errors = []
        self.stats = {'total': 0, 'success': 0, 'failed': 0}

    def parse_timestamp(self, ts_str: str) -> datetime:
        """
        Parse timestamp từ định dạng: 01/Jul/1995:00:00:01

        Args:
            ts_str: Chuỗi timestamp cần parse

        Returns:
            datetime object

        Raises:
            ValueError: Nếu format không đúng
        """
        # Format: DD/Mon/YYYY:HH:MM:SS
        day = int(ts_str[0:2])
        month = self.MONTHS[ts_str[3:6]]
        year = int(ts_str[7:11])
        hour = int(ts_str[12:14])
        minute = int(ts_str[15:17])
        second = int(ts_str[18:20])

        return datetime(year, month, day, hour, minute, second)

    def parse_bytes(self, bytes_str: str) -> int:
        """
        Parse trường bytes - xử lý trường hợp "-".

        Args:
            bytes_str: Giá trị bytes từ log (có thể là "-")

        Returns:
            int: Số bytes, 0 nếu là "-"
        """
        if bytes_str == '-':
            return 0
        try:
            return int(bytes_str)
        except ValueError:
            return 0

    def parse_line(self, line: str, line_num: int = 0) -> Optional[Dict[str, Any]]:
        """
        Parse một dòng log thành dictionary.

        Args:
            line: Dòng log cần parse
            line_num: Số thứ tự dòng (để debug)

        Returns:
            Dict với các fields đã parse, hoặc None nếu lỗi
        """
        self.stats['total'] += 1

        # Thử match với pattern
        match = self.LOG_PATTERN.match(line.strip())

        if not match:
            self.stats['failed'] += 1
            # Lưu lại dòng lỗi (chỉ 100 ký tự đầu)
            self.parse_errors.append((line_num, line[:100] if len(line) > 100 else line.strip()))
            return None

        self.stats['success'] += 1

        # Extract các groups từ regex match
        host, ts_str, tz, method, url, protocol, status, bytes_str = match.groups()

        return {
            'host': host,
            'timestamp': self.parse_timestamp(ts_str),
            'timezone': tz,
            'method': method,
            'url': url,
            'protocol': protocol if protocol else 'HTTP/1.0',
            'status_code': int(status),
            'bytes': self.parse_bytes(bytes_str)
        }

    def parse_file(
        self,
        filepath: str,
        chunk_size: int = 100000,
        show_progress: bool = True,
        encoding: str = 'latin-1'
    ) -> pd.DataFrame:
        """
        Parse toàn bộ file log thành DataFrame.

        Sử dụng chunked reading để xử lý file lớn hiệu quả.

        Args:
            filepath: Đường dẫn tới file log
            chunk_size: Số dòng để báo cáo progress
            show_progress: Hiển thị progress bar
            encoding: Encoding của file (mặc định latin-1 cho ASCII extended)

        Returns:
            DataFrame với các cột: host, timestamp, timezone, method,
                                   url, protocol, status_code, bytes
        """
        self.reset_stats()
        records = []

        # Đếm tổng số dòng trước (để progress bar chính xác)
        total_lines = None
        if show_progress:
            print("Đang đếm số dòng trong file...")
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                total_lines = sum(1 for _ in f)
            print(f"Tổng số dòng: {total_lines:,}")

        # Parse từng dòng
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            iterator = tqdm(enumerate(f, 1), total=total_lines, desc="Parsing logs") if show_progress else enumerate(f, 1)

            for line_num, line in iterator:
                parsed = self.parse_line(line, line_num)
                if parsed:
                    records.append(parsed)

        # Tạo DataFrame
        df = pd.DataFrame(records)

        if len(df) > 0:
            # Convert timestamp sang datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Sort theo timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

        # In thống kê
        print(f"\n{'='*50}")
        print(f"KẾT QUẢ PARSING")
        print(f"{'='*50}")
        print(f"Tổng số dòng:     {self.stats['total']:>12,}")
        print(f"Parse thành công: {self.stats['success']:>12,} ({self.stats['success']/self.stats['total']*100:.2f}%)")
        print(f"Parse thất bại:   {self.stats['failed']:>12,} ({self.stats['failed']/self.stats['total']*100:.2f}%)")
        print(f"{'='*50}")

        if self.stats['failed'] > 0 and len(self.parse_errors) > 0:
            print(f"\nMẫu các dòng lỗi (tối đa 5 dòng đầu):")
            for line_num, content in self.parse_errors[:5]:
                print(f"  Line {line_num}: {content[:80]}...")

        return df

    def get_parse_errors(self, max_errors: int = 100) -> List[Tuple[int, str]]:
        """
        Lấy danh sách các dòng parse lỗi.

        Args:
            max_errors: Số lỗi tối đa trả về

        Returns:
            List của tuples (line_number, line_content)
        """
        return self.parse_errors[:max_errors]

    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê parsing.

        Returns:
            Dict với total, success, failed, success_rate
        """
        success_rate = self.stats['success'] / self.stats['total'] * 100 if self.stats['total'] > 0 else 0
        return {
            **self.stats,
            'success_rate': success_rate
        }


def quick_parse(filepath: str, max_lines: int = 1000) -> pd.DataFrame:
    """
    Hàm tiện ích để parse nhanh một số dòng đầu của file.

    Hữu ích cho việc kiểm tra format và debug.

    Args:
        filepath: Đường dẫn tới file log
        max_lines: Số dòng tối đa cần parse

    Returns:
        DataFrame với các dòng đã parse
    """
    parser = NASALogParser()
    records = []

    with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > max_lines:
                break
            parsed = parser.parse_line(line, line_num)
            if parsed:
                records.append(parsed)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


if __name__ == "__main__":
    # Demo usage
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "DATA/train.txt"

    print(f"Parsing file: {filepath}")
    print("-" * 50)

    parser = NASALogParser()
    df = parser.parse_file(filepath)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nSample data:\n{df.head()}")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
