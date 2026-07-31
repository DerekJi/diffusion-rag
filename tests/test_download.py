"""下载模块单元测试。

使用 mock 避免实际网络请求。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.download import (
    SUPPORTED_DATASETS,
    _check_dataset_cached,
    _check_elf_cached,
    _resolve_cache_dir,
    download_all,
    download_dataset,
    download_elf_weights,
    main,
)


class TestResolveCacheDir:
    """_resolve_cache_dir 单元测试。"""

    def test_default_cache_dir(self) -> None:
        models_dir, data_dir = _resolve_cache_dir(None)
        assert models_dir == Path("models")
        assert data_dir == Path("data")

    def test_custom_cache_dir(self) -> None:
        models_dir, data_dir = _resolve_cache_dir("/my/cache")
        assert models_dir == Path("/my/cache/models")
        assert data_dir == Path("/my/cache/data")

    def test_relative_custom_cache_dir(self) -> None:
        models_dir, data_dir = _resolve_cache_dir("my_cache")
        assert models_dir == Path("my_cache/models")
        assert data_dir == Path("my_cache/data")


class TestCheckElfCached:
    """_check_elf_cached 单元测试。"""

    def test_cached_when_dir_exists_and_nonempty(self, tmp_path: Path) -> None:
        elf_dir = tmp_path / "embedded-language-flows" / "ELF-B-owt-torch"
        elf_dir.mkdir(parents=True)
        (elf_dir / "checkpoint_95085").touch()
        assert _check_elf_cached(tmp_path) is True

    def test_not_cached_when_dir_empty(self, tmp_path: Path) -> None:
        elf_dir = tmp_path / "embedded-language-flows" / "ELF-B-owt-torch"
        elf_dir.mkdir(parents=True)
        assert _check_elf_cached(tmp_path) is False

    def test_not_cached_when_no_dir(self, tmp_path: Path) -> None:
        assert _check_elf_cached(tmp_path) is False

    def test_cached_with_checkpoint_file(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint_95085").touch()
        assert _check_elf_cached(tmp_path) is True


class TestCheckDatasetCached:
    """_check_dataset_cached 单元测试。"""

    def test_cached_with_marker_file(self, tmp_path: Path) -> None:
        marker = tmp_path / ".nfcorpus_downloaded"
        marker.touch()
        assert _check_dataset_cached("nfcorpus", tmp_path) is True

    def test_not_cached_without_marker(self, tmp_path: Path) -> None:
        assert _check_dataset_cached("nfcorpus", tmp_path) is False


class TestDownloadElfWeights:
    """download_elf_weights 单元测试（mock huggingface_hub）。"""

    @patch("huggingface_hub.snapshot_download")
    @patch("src.utils.download._check_elf_cached", return_value=False)
    def test_download_success(
        self,
        mock_check: MagicMock,
        mock_snapshot: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = download_elf_weights(cache_dir=str(tmp_path))

        # snapshot_download 创建目录，返回的路径包含完整 repo 名称
        expected = tmp_path / "models" / "embedded-language-flows" / "ELF-B-owt-torch"
        assert result == expected
        mock_snapshot.assert_called_once()
        assert mock_snapshot.call_args[1]["repo_id"] == "embedded-language-flows/ELF-B-owt-torch"

    @patch("huggingface_hub.snapshot_download")
    @patch("src.utils.download._check_elf_cached", return_value=False)
    def test_download_skipped_when_cached(
        self,
        mock_check: MagicMock,
        mock_snapshot: MagicMock,
        tmp_path: Path,
    ) -> None:
        # 模拟已缓存 — 并创建 repo 目录使其真实存在
        mock_check.return_value = True
        repo_dir = tmp_path / "models" / "embedded-language-flows" / "ELF-B-owt-torch"
        repo_dir.mkdir(parents=True)
        (repo_dir / "checkpoint_95085").touch()

        result = download_elf_weights(cache_dir=str(tmp_path))

        # 不应调用 snapshot_download
        mock_snapshot.assert_not_called()
        assert result == repo_dir

    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.hf_hub_download")
    @patch("src.utils.download._check_elf_cached", return_value=False)
    def test_download_fallback_to_checkpoint(
        self,
        mock_check: MagicMock,
        mock_hf_hub: MagicMock,
        mock_snapshot: MagicMock,
        tmp_path: Path,
    ) -> None:
        # snapshot_download 失败
        mock_snapshot.side_effect = RuntimeError("Network error")
        mock_hf_hub.return_value = str(tmp_path / "models" / "checkpoint_95085")

        result = download_elf_weights(cache_dir=str(tmp_path))

        # 应回退到 checkpoint 下载
        mock_hf_hub.assert_called_once()
        mock_hf_hub.assert_called_once_with(
            repo_id="embedded-language-flows/ELF-B-owt-torch",
            filename="checkpoint_95085",
            local_dir=str(tmp_path / "models"),
            resume_download=True,
        )
        assert result == tmp_path / "models"


class TestDownloadDataset:
    """download_dataset 单元测试（mock datasets）。"""

    @patch("datasets.load_dataset")
    @patch("src.utils.download._check_dataset_cached", return_value=False)
    def test_download_success(
        self,
        mock_check: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        # 模拟 load_dataset 返回
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 100
        mock_load.return_value = {"queries": mock_ds}

        download_dataset("nfcorpus", cache_dir=str(tmp_path))

        # 应调用 load_dataset 三次（主数据集 + qrels + 验证查询）
        assert mock_load.call_count == 3
        calls = [c[0] for c in mock_load.call_args_list]
        assert any("BeIR/nfcorpus" in str(c) and "queries" not in str(c) for c in calls)
        assert any("BeIR/nfcorpus-qrels" in str(c) for c in calls)

    @patch("datasets.load_dataset")
    @patch("src.utils.download._check_dataset_cached", return_value=False)
    def test_download_verification(
        self,
        mock_check: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """验证下载后会创建标记文件。"""
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 100
        mock_load.return_value = {"queries": mock_ds}

        download_dataset("nfcorpus", cache_dir=str(tmp_path))

        # 标记文件应存在
        marker = tmp_path / "data" / ".nfcorpus_downloaded"
        assert marker.is_file()

    @patch("src.utils.download._check_dataset_cached", return_value=True)
    def test_download_skipped_when_cached(
        self,
        mock_check: MagicMock,
        tmp_path: Path,
    ) -> None:
        # 使用无网络上下文，确保不真正调用 load_dataset
        with patch("datasets.load_dataset") as mock_load:
            download_dataset("nfcorpus", cache_dir=str(tmp_path))
            mock_load.assert_not_called()

    def test_unsupported_dataset(self) -> None:
        with pytest.raises(ValueError, match="不支持的数据集"):
            download_dataset("nonexistent")

    @patch("datasets.load_dataset")
    @patch("src.utils.download._check_dataset_cached", return_value=False)
    def test_download_all_retries_exhausted(
        self,
        mock_check: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """所有重试均失败时抛出 RuntimeError。"""
        mock_load.side_effect = RuntimeError("Connection refused")

        with pytest.raises(RuntimeError, match="下载失败"):
            download_dataset("nfcorpus", cache_dir=str(tmp_path))

        # 应重试 _MAX_RETRIES 次
        assert mock_load.call_count >= 3


class TestDownloadAll:
    """download_all 单元测试。"""

    @patch("src.utils.download.download_elf_weights")
    @patch("src.utils.download.download_dataset")
    def test_download_all_calls_both(
        self,
        mock_ds: MagicMock,
        mock_elf: MagicMock,
    ) -> None:
        download_all()

        mock_elf.assert_called_once()
        # 应为每个支持的数据集调用一次
        assert mock_ds.call_count == len(SUPPORTED_DATASETS)


class TestMainCLI:
    """CLI 入口 main() 单元测试。"""

    @patch("src.utils.download.download_all")
    def test_main_all(self, mock_all: MagicMock) -> None:
        main(["--all"])
        mock_all.assert_called_once()

    @patch("src.utils.download.download_elf_weights")
    def test_main_elf_weights(self, mock_elf: MagicMock) -> None:
        main(["--elf-weights"])
        mock_elf.assert_called_once()

    @patch("src.utils.download.download_dataset")
    def test_main_dataset(self, mock_ds: MagicMock) -> None:
        main(["--dataset", "nfcorpus"])
        mock_ds.assert_called_once_with("nfcorpus", cache_dir=None)

    @patch("src.utils.download.download_dataset")
    def test_main_dataset_with_cache_dir(self, mock_ds: MagicMock) -> None:
        main(["--dataset", "nfcorpus", "--cache-dir", "/my/cache"])
        mock_ds.assert_called_once_with("nfcorpus", cache_dir="/my/cache")

    @patch("src.utils.download.download_all")
    def test_main_default_to_all(self, mock_all: MagicMock) -> None:
        """无参数时默认下载全部。"""
        main([])
        mock_all.assert_called_once()

    @patch("src.utils.download.download_dataset")
    def test_main_help(self, mock_ds: MagicMock) -> None:
        """--help 应打印帮助并退出。"""
        with pytest.raises(SystemExit):
            main(["--help"])
        mock_ds.assert_not_called()
