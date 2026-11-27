import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from inquiry_loop.providers.hub_models import run_model_command


class HubModelTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.enterContext(patch("inquiry_loop.providers.hub_models.HfApi", return_value=self.api))
        self.output = self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def set_files(self, *names):
        self.api.model_info.return_value = SimpleNamespace(
            sha="revision", siblings=[SimpleNamespace(rfilename=name) for name in names])

    def test_empty_search(self):
        self.api.list_models.return_value = iter([])
        run_model_command("/search", ["missing", "model"])
        self.assertIn("No matching", self.output.getvalue())

    def test_search_alternates_formats_and_limits_results(self):
        self.api.list_models.side_effect = [
            [SimpleNamespace(id=f"owner/{model_format}-{index}") for index in range(1, 6)]
            for model_format in ("mlx", "gguf")
        ]
        run_model_command("/search", ["model"])
        self.assertEqual(self.output.getvalue().splitlines(), [
            "1. owner/mlx-1 (mlx)",
            "2. owner/gguf-1 (gguf)",
            "3. owner/mlx-2 (mlx)",
            "4. owner/gguf-2 (gguf)",
            "5. owner/mlx-3 (mlx)",
        ])

    def test_search_accepts_a_single_format(self):
        self.api.list_models.return_value = [SimpleNamespace(id="owner/mlx")]
        run_model_command("/search", ["--mlx", "some", "model"])
        self.api.list_models.assert_called_once_with(
            search="some model", filter="mlx", limit=5, sort="downloads")
        self.assertIn("owner/mlx (mlx)", self.output.getvalue())

    def test_multiple_ggufs_require_a_selection(self):
        self.set_files("Q4_K_M.gguf", "Q5_K_M.gguf", "README.md")
        run_model_command("/download", ["owner/model"])
        self.assertIn("/download owner/model FILENAME", self.output.getvalue())
        self.assertIn("Q4_K_M.gguf", self.output.getvalue())
        self.assertIn("Q5_K_M.gguf", self.output.getvalue())
        self.api.snapshot_download.assert_not_called()
        self.api.hf_hub_download.assert_not_called()

    def test_single_variant_uses_one_file_and_excludes_projector(self):
        self.set_files("nested/Q4_K_M.gguf", "model-mmproj-f16.gguf", "model-imatrix.gguf")
        run_model_command("/download", ["owner/model"])
        self.api.hf_hub_download.assert_called_once_with("owner/model", "nested/Q4_K_M.gguf", revision="revision")
        self.api.snapshot_download.assert_not_called()

    def test_selecting_split_gguf_downloads_all_shards_of_that_variant(self):
        first = "quant/Q4_K_M-00001-of-00002.gguf"
        second = "quant/Q4_K_M-00002-of-00002.gguf"
        self.set_files(first, second, "Q5_K_M.gguf")
        self.api.hf_hub_download.side_effect = ["/cache/" + first, "/cache/" + second]
        run_model_command("/download", ["owner/model", second])
        self.assertEqual(self.api.hf_hub_download.call_args_list, [
            call("owner/model", first, revision="revision"), call("owner/model", second, revision="revision"),
        ])
        self.assertIn("Cached file: /cache/" + first, self.output.getvalue())
