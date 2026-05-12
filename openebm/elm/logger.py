"""Custom logging helpers used by EBT training and evaluation scripts.

Provides:

- :class:`Tee` — mirror writes to the console and a file at the same time.
- :class:`CustomStreamHandler` / :class:`CustomLogger` — a logger whose
  ``info`` method can optionally suppress console output per call.
- :class:`JsonlLogger` — a DDP-aware JSON-lines structured logger that shards
  files per rank and can later merge them.
"""
import glob
import json
import logging
import os
import sys
from typing import Any, Union


class Tee(object):
    """Duplicate writes to both the terminal and a log file."""

    def __init__(self, terminal: Any, logfile: str) -> None:
        """Initialize the tee.

        :param terminal: Target terminal / stream (e.g. ``sys.stdout``).
        :type terminal: Any
        :param logfile: Path to the file that mirrors the writes.
        :type logfile: str
        """
        self.terminal = terminal
        if not os.path.exists(logfile):
            open(logfile, 'w').close()
        self.log = open(logfile, 'a')

    def write(self, message: str) -> None:
        """Write ``message`` to both the terminal and the log file.

        :param message: Text to write.
        :type message: str
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """Flush the log file buffer."""
        self.log.flush()


class CustomStreamHandler(logging.StreamHandler):
    """Stream handler that honors a ``print_to_console`` flag on records."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit ``record`` only when ``print_to_console`` is truthy.

        :param record: Log record to emit.
        :type record: logging.LogRecord
        """
        print_to_console = getattr(record, 'print_to_console', True)
        if print_to_console:
            super().emit(record)



class CustomLogger(logging.Logger):
    """Logger whose ``info`` method accepts an explicit console toggle."""

    def info(self, msg: str, *args: Any, print_to_console: bool = True, **kwargs: Any) -> None:
        """Log an INFO record with an optional ``print_to_console`` flag.

        :param msg: Log message.
        :type msg: str
        :param print_to_console: When ``False``, the record is written to file
            handlers but suppressed by :class:`CustomStreamHandler`.
        :type print_to_console: bool
        """
        if self.isEnabledFor(logging.INFO):
            if 'extra' in kwargs:
                kwargs['extra']['print_to_console'] = print_to_console
            else:
                kwargs['extra'] = {'print_to_console': print_to_console}
            self._log(logging.INFO, msg, args, **kwargs)

def setup_custom_logger(
    log_filename: str,
    name: str = "custom_logger",
    print_console: bool = True,
    base_lor_dir: str = "./logs/",
) -> CustomLogger:
    """Build a :class:`CustomLogger` wired to a file (and optionally console).

    :param log_filename: File name (relative to ``base_lor_dir``).
    :type log_filename: str
    :param name: Logger name.
    :type name: str
    :param print_console: When ``True``, a :class:`Tee`-backed console handler
        is also attached.
    :type print_console: bool
    :param base_lor_dir: Base log directory.
    :type base_lor_dir: str
    :return: Configured logger.
    :rtype: CustomLogger
    """
    custom_logger = CustomLogger(name=name)
    custom_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(base_lor_dir + log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    custom_logger.addHandler(file_handler)

    if print_console:
        logfile_full_path = os.path.join(base_lor_dir, log_filename)
        tee = Tee(sys.stdout, logfile_full_path)
        console_handler = CustomStreamHandler(stream=tee)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        custom_logger.addHandler(console_handler)

    return custom_logger

class JsonlLogger(logging.Logger):
    """DDP-aware structured logger that appends one JSON object per line."""

    def __init__(self, name: str, log_filename: str, base_log_dir: str = "./logs/") -> None:
        """Open (truncate) the target file and shard it by rank under DDP.

        :param name: Logger name (propagated to ``logging.Logger``).
        :type name: str
        :param log_filename: File name (relative to ``base_log_dir``).
        :type log_filename: str
        :param base_log_dir: Base log directory. Created if missing.
        :type base_log_dir: str
        """
        super().__init__(name)
        os.makedirs(base_log_dir, exist_ok=True)

        # Under DDP, each rank writes to its own shard to avoid concurrent
        # append corruption; the shards can later be merged via
        # :meth:`merge_rank_files`.
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            base, ext = os.path.splitext(log_filename)
            log_filename = f"{base}_rank{rank}{ext}"

        self.log_filename = os.path.join(base_log_dir, log_filename)
        # NOTE: Opening in ``'w'`` truncates the file on every run; switch to
        # append mode if you need cumulative logs across runs.
        self.file = open(self.log_filename, 'w', encoding='utf-8')

    def log_data(self, data: Union[dict, list]) -> None:
        """Write ``data`` as one JSON-encoded line.

        :param data: Any JSON-serializable dict or list.
        :type data: Union[dict, list]
        :raises ValueError: If ``data`` is neither a dict nor a list.
        """
        if not isinstance(data, (dict, list)):
            raise ValueError("Only dictionary or list data can be logged in JSONL format.")

        self.file.write(json.dumps(data) + '\n')
        self.file.flush()

    def close(self) -> None:
        """Close the underlying file descriptor."""
        self.file.close()

    @staticmethod
    def merge_rank_files(base_log_dir: str, log_filename: str = "results.jsonl") -> None:
        """Merge per-rank shard files into a single unified file.

        :param base_log_dir: Directory containing the shards.
        :type base_log_dir: str
        :param log_filename: Final merged file name.
        :type log_filename: str
        """
        base, ext = os.path.splitext(log_filename)
        pattern = os.path.join(base_log_dir, f"{base}_rank*{ext}")
        rank_files = sorted(glob.glob(pattern))
        if not rank_files:
            return
        merged_path = os.path.join(base_log_dir, log_filename)
        with open(merged_path, 'w', encoding='utf-8') as out:
            for rf in rank_files:
                with open(rf, 'r', encoding='utf-8') as inp:
                    for line in inp:
                        out.write(line)
                os.remove(rf)

def setup_jsonl_logger(
    log_filename: str,
    name: str = "jsonl_logger",
    base_log_dir: str = "./logs/",
) -> JsonlLogger:
    """Factory for :class:`JsonlLogger`.

    :param log_filename: File name (relative to ``base_log_dir``).
    :type log_filename: str
    :param name: Logger name.
    :type name: str
    :param base_log_dir: Base log directory.
    :type base_log_dir: str
    :return: A configured :class:`JsonlLogger`.
    :rtype: JsonlLogger
    """
    return JsonlLogger(name=name, log_filename=log_filename, base_log_dir=base_log_dir)
