import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import app as app_module
from app import app


def _mock_tk_modules(selected="", tk_factory=None):
    tk_module = types.ModuleType("tkinter")
    filedialog_module = types.ModuleType("tkinter.filedialog")

    class FakeRoot:
        def withdraw(self):
            pass

        def attributes(self, *args):
            pass

        def update(self):
            pass

        def destroy(self):
            pass

    tk_module.Tk = tk_factory or FakeRoot
    filedialog_module.askdirectory = mock.Mock(return_value=selected)
    tk_module.filedialog = filedialog_module
    return tk_module, filedialog_module


class NativeFolderPickerTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_successful_native_selection_returns_selected_path(self):
        with tempfile.TemporaryDirectory() as td:
            tk_module, filedialog_module = _mock_tk_modules(selected=td)
            with mock.patch.object(app_module.os, "name", "nt"), mock.patch.dict(
                "sys.modules",
                {"tkinter": tk_module, "tkinter.filedialog": filedialog_module},
            ):
                resp = self.client.post("/api/native-folder-picker", json={"initial_dir": td})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"), msg=data)
        self.assertFalse(data.get("cancelled"))
        self.assertEqual(Path(data.get("path")).resolve(), Path(td).resolve())

    def test_cancel_returns_cancelled_with_empty_path(self):
        tk_module, filedialog_module = _mock_tk_modules(selected="")
        with mock.patch.object(app_module.os, "name", "nt"), mock.patch.dict(
            "sys.modules",
            {"tkinter": tk_module, "tkinter.filedialog": filedialog_module},
        ):
            data = app_module._native_folder_picker()

        self.assertTrue(data.get("ok"), msg=data)
        self.assertTrue(data.get("cancelled"))
        self.assertEqual(data.get("path"), "")

    def test_non_windows_returns_fallback_allowed(self):
        with mock.patch.object(app_module.os, "name", "posix"):
            data = app_module._native_folder_picker()

        self.assertFalse(data.get("ok"))
        self.assertTrue(data.get("fallback_allowed"))

    def test_strict_browse_mode_rejects_selection_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            tk_module, filedialog_module = _mock_tk_modules(selected=outside)
            with mock.patch.object(app_module.os, "name", "nt"), mock.patch.object(
                app_module, "BROWSE_STRICT_MODE", True
            ), mock.patch.object(
                app_module, "BROWSE_ALLOWED_ROOTS", [Path(allowed)]
            ), mock.patch.dict(
                "sys.modules",
                {"tkinter": tk_module, "tkinter.filedialog": filedialog_module},
            ):
                data = app_module._native_folder_picker()

        self.assertFalse(data.get("ok"))
        self.assertIn("outside allowed roots", data.get("error", ""))
        self.assertNotIn("path", data)

    def test_dialog_lock_released_after_exception(self):
        class BrokenRoot:
            def __init__(self):
                raise RuntimeError("boom")

        tk_module, filedialog_module = _mock_tk_modules(tk_factory=BrokenRoot)
        with mock.patch.object(app_module.os, "name", "nt"), mock.patch.dict(
            "sys.modules",
            {"tkinter": tk_module, "tkinter.filedialog": filedialog_module},
        ):
            resp = self.client.post("/api/native-folder-picker", json={})
            data = resp.get_json()

        self.assertFalse(data.get("ok"))
        acquired = app_module._NATIVE_FOLDER_PICKER_LOCK.acquire(blocking=False)
        try:
            self.assertTrue(acquired)
        finally:
            if acquired:
                app_module._NATIVE_FOLDER_PICKER_LOCK.release()


if __name__ == "__main__":
    unittest.main()
