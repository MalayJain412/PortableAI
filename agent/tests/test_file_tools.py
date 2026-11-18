from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import windows_tools
from config import CACHE_DIR


class ManagedFileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temporary_directory.name)
        self.workspace = self.root / "workspace"
        self.note_cache = self.root / "cache"
        self.notes = self.note_cache / "notepad"
        self.workspace.mkdir(parents=True)
        self.notes.mkdir(parents=True)
        self.default_note = self.notes / "portable_ai_current_note.txt"

        self.patches = [
            patch("security.WORKSPACE_DIR", self.workspace),
            patch("security.CACHE_DIR", self.note_cache),
            patch("windows_tools._CURRENT_NOTEPAD_FILE", self.default_note),
            patch("windows_tools._ACTIVE_NOTEPAD_FILE", self.default_note),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary_directory.cleanup()

    def test_rename_keeps_extension_and_updates_active_note(self) -> None:
        self.default_note.write_text("255 spent on the swiggy order", encoding="utf-8")

        result = windows_tools.rename_file(str(self.default_note), "nagpur-expenses")

        renamed = self.notes / "nagpur-expenses.txt"
        self.assertEqual(result["path"], str(renamed))
        self.assertFalse(self.default_note.exists())
        self.assertEqual(renamed.read_text(encoding="utf-8"), "255 spent on the swiggy order")
        self.assertEqual(windows_tools.ensure_current_notepad_file(), renamed)

    def test_copy_to_directory_then_move_to_exact_path(self) -> None:
        source = self.workspace / "expense.txt"
        source.write_text("expense", encoding="utf-8")
        backup = self.workspace / "backup"
        archive = self.workspace / "archive"
        backup.mkdir()
        archive.mkdir()

        copied = windows_tools.copy_file(str(source), str(backup))
        copied_path = backup / source.name
        self.assertEqual(copied["path"], str(copied_path))
        self.assertTrue(source.exists())
        self.assertEqual(copied_path.read_text(encoding="utf-8"), "expense")

        moved_path = archive / "old-expense.txt"
        moved = windows_tools.move_file(str(copied_path), str(moved_path))
        self.assertEqual(moved["path"], str(moved_path))
        self.assertFalse(copied_path.exists())
        self.assertTrue(moved_path.exists())

    def test_existing_destination_is_not_silently_overwritten(self) -> None:
        source = self.workspace / "source.txt"
        destination = self.workspace / "destination.txt"
        source.write_text("new", encoding="utf-8")
        destination.write_text("keep", encoding="utf-8")

        with self.assertRaises(Exception):
            windows_tools.copy_file(str(source), str(destination))

        self.assertEqual(destination.read_text(encoding="utf-8"), "keep")

    def test_delete_file_removes_only_the_requested_file(self) -> None:
        target = self.workspace / "remove.txt"
        keep = self.workspace / "keep.txt"
        target.write_text("remove", encoding="utf-8")
        keep.write_text("keep", encoding="utf-8")

        result = windows_tools.delete_file(str(target))

        self.assertFalse(target.exists())
        self.assertTrue(keep.exists())
        self.assertEqual(result["deleted_bytes"], 6)

    def test_file_changes_outside_managed_roots_are_rejected(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        with self.assertRaises(Exception):
            windows_tools.rename_file(str(outside), "renamed.txt")


if __name__ == "__main__":
    unittest.main()
