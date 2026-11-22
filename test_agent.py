import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import mock_open, patch

import agent


class ToolTests(unittest.TestCase):
    def test_shell_syntax_is_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker"
            result = agent.run_shell(f"echo allowed; touch {marker}")
            self.assertFalse(marker.exists())
            self.assertIn("; touch", result["stdout"])
            result = agent.run_shell(f"echo $(touch {marker}) > {marker}")
            self.assertFalse(marker.exists())
            self.assertIn("$(touch", result["stdout"])

    def test_command_validation_and_quoted_arguments(self):
        for command in ("", "   ", "python --version", "echo '"):
            with self.subTest(command=command):
                self.assertIn("error", agent.run_shell(command))
        self.assertEqual(agent.run_shell('echo "hello world"')["stdout"], "hello world\n")

    def test_read_preview_is_bounded(self):
        opened = mock_open(read_data="x" * 4001)
        with patch("builtins.open", opened):
            result = agent.read_file("large.txt")
        opened().read.assert_called_once_with(4001)
        self.assertEqual(result["content"], "x" * 4000)
        self.assertTrue(result["truncated"])

    def test_read_exact_limit_is_not_truncated(self):
        with patch("builtins.open", mock_open(read_data="x" * 4000)):
            self.assertFalse(agent.read_file("exact.txt")["truncated"])

    def test_utf8_byte_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.txt"
            result = agent.write_file(str(path), "é🙂\n")
            self.assertEqual(result["bytes_written"], len(path.read_bytes()))
            self.assertEqual(path.read_text(), "é🙂\n")

    @unittest.skipUnless(os.name == "posix", "uses /dev/zero")
    def test_infinite_output_is_stopped(self):
        result = agent.run_shell("cat /dev/zero")
        self.assertLessEqual(len(result["stdout"]), 4000)
        self.assertTrue(result["output_limit_exceeded"])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires named pipes")
    def test_blocked_command_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "fifo"
            os.mkfifo(fifo)
            with patch.object(agent, "SHELL_TIMEOUT", 0.1):
                result = agent.run_shell(f"cat {fifo}")
            self.assertTrue(result["timed_out"])

    def test_stub_flow(self):
        result = agent.run_agent("List files", agent.StubModel(), verbose=False)
        self.assertTrue(result.startswith("Done."))


if __name__ == "__main__":
    unittest.main()
