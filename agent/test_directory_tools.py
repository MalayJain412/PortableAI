from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import windows_tools
from security import validate_directory_path


class ManagedDirectoryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = Path(__file__).resolve().parent.parent / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=cache)
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace_patch = patch("security.WORKSPACE_DIR", self.workspace)
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temporary_directory.cleanup()

    def test_create_list_summarize_and_delete_directory(self) -> None:
        created = windows_tools.create_directory("projects/demo/nested")
        self.assertTrue(created["created"])

        demo = self.workspace / "projects" / "demo"
        (demo / "readme.txt").write_text("hello", encoding="utf-8")
        (demo / "nested" / "data.json").write_text("{}", encoding="utf-8")

        listing = windows_tools.list_directory("projects/demo")
        self.assertEqual(
            [(entry["name"], entry["type"]) for entry in listing["entries"]],
            [("nested", "directory"), ("readme.txt", "file")],
        )

        summary = windows_tools.summarize_directory("projects/demo")
        self.assertEqual(summary["directories"], 1)
        self.assertEqual(summary["files"], 2)
        self.assertEqual(summary["total_bytes"], 7)
        self.assertEqual(summary["extensions"], {".json": 1, ".txt": 1})

        deleted = windows_tools.delete_directory("projects/demo")
        self.assertFalse(demo.exists())
        self.assertEqual(deleted["deleted_directories"], 2)
        self.assertEqual(deleted["deleted_files"], 2)

    def test_paths_outside_workspace_are_rejected(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        with self.assertRaises(Exception):
            windows_tools.create_directory(str(outside))

    def test_external_local_directory_can_be_listed_and_summarized(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        (outside / "folder-one").mkdir(parents=True)
        (outside / "file-one.txt").write_text("content", encoding="utf-8")

        listing = windows_tools.list_directory(str(outside), include="directories")
        self.assertEqual(
            [(entry["name"], entry["type"]) for entry in listing["entries"]],
            [("folder-one", "directory")],
        )

        summary = windows_tools.summarize_directory(str(outside))
        self.assertEqual(summary["directories"], 1)
        self.assertEqual(summary["files"], 1)

    def test_spoken_drive_name_is_normalized(self) -> None:
        self.assertEqual(validate_directory_path("E drive"), "E:\\")
        self.assertEqual(validate_directory_path("e disk\\Projects"), "E:\\Projects")

    def test_workspace_root_cannot_be_deleted(self) -> None:
        windows_tools.create_directory("keep")
        with self.assertRaises(Exception):
            windows_tools.delete_directory(".")


if __name__ == "__main__":
    unittest.main()
