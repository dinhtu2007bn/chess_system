from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
import zipfile
from typing import Any

from chesslab_sdk import BotConfig, SearchContext


ROOT_DIR = Path(__file__).resolve().parents[1]
MANAGED_BOTS_DIR = ROOT_DIR / "managed_bots"
REGISTRY_FILE = MANAGED_BOTS_DIR / "registry.json"


class BotManager:
    """Manages bot uploads, validation, registry metadata, and disk storage."""

    def __init__(self) -> None:
        MANAGED_BOTS_DIR.mkdir(parents=True, exist_ok=True)
        if not REGISTRY_FILE.exists():
            self._save_registry({"bots": []})

    def _load_registry(self) -> dict[str, Any]:
        if not REGISTRY_FILE.exists():
            return {"bots": []}
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"bots": []}

    def _save_registry(self, data: dict[str, Any]) -> None:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def list_bots(self) -> list[dict[str, Any]]:
        registry = self._load_registry()
        bots = registry.get("bots", [])
        enriched = []
        for b in bots:
            dir_name = b.get("dir_name", "")
            bot_dir = MANAGED_BOTS_DIR / dir_name
            # Fallback to case-insensitive match on actual directory names
            if not bot_dir.is_dir() and MANAGED_BOTS_DIR.is_dir():
                for actual_dir in MANAGED_BOTS_DIR.iterdir():
                    if actual_dir.is_dir() and actual_dir.name.lower() == dir_name.lower():
                        bot_dir = actual_dir
                        break

            exists = bot_dir.is_dir()
            entry = dict(b)
            entry["dir_path"] = str(bot_dir) if exists else None
            entry["is_valid"] = exists and (
                (bot_dir / "bot.py").is_file() or (bot_dir / "student" / "bot.py").is_file()
            )
            enriched.append(entry)
        return enriched

    def get_bot(self, bot_id: str) -> dict[str, Any] | None:
        if not bot_id:
            return None
        bots = self.list_bots()
        # 1. Exact match on id
        for b in bots:
            if b.get("id") == bot_id:
                return b

        # 2. Case-insensitive match on id or dir_name
        for b in bots:
            if b.get("id", "").lower() == bot_id.lower() or b.get("dir_name", "").lower() == bot_id.lower():
                return b

        # 3. Match on display name (e.g. "Bot V1" or "Bot v1")
        for b in bots:
            if b.get("name", "").lower() == bot_id.lower():
                return b

        # 4. Alias mapping for backward compatibility
        alias_map = {
            "student-default": "bot_v1",
            "student-fast-timer": "bot_v0",
            "bot_68c5de1d": "bot_v2",
            "bot_fd246a5a": "bot_v3",
        }
        if bot_id.lower() in alias_map:
            target_id = alias_map[bot_id.lower()]
            for b in bots:
                if b.get("id", "").lower() == target_id:
                    return b

        # 5. Fallback for default request
        if bot_id.lower() in ("default", "student"):
            for b in bots:
                if b.get("is_default"):
                    return b

        return None

    def rename_bot(self, bot_id: str, new_name: str) -> dict[str, Any]:
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Tên hiển thị của bot không được để trống")
        registry = self._load_registry()
        bots = registry.get("bots", [])
        target = None
        for b in bots:
            if b.get("id") == bot_id:
                b["name"] = new_name
                target = b
                break
        if target is None:
            raise KeyError(f"Không tìm thấy bot có ID '{bot_id}'")
        self._save_registry(registry)
        return target

    def delete_bot(self, bot_id: str) -> bool:
        registry = self._load_registry()
        bots = registry.get("bots", [])
        target_idx = None
        target_dir_name = None
        for i, b in enumerate(bots):
            if b.get("id") == bot_id:
                target_idx = i
                target_dir_name = b.get("dir_name")
                break
        if target_idx is None:
            raise KeyError(f"Không tìm thấy bot có ID '{bot_id}'")

        bots.pop(target_idx)
        self._save_registry({"bots": bots})

        if target_dir_name:
            target_dir = MANAGED_BOTS_DIR / target_dir_name
            if target_dir.exists() and target_dir.is_dir():
                try:
                    shutil.rmtree(target_dir)
                except Exception:
                    pass
        return True

    def export_bot_zip(self, bot_id: str) -> tuple[io.BytesIO, str]:
        """Packages bot python files into a clean zip archive ready for extraction into student/."""
        bot = self.get_bot(bot_id)
        if not bot or not bot.get("dir_path"):
            raise KeyError(f"Bot '{bot_id}' không tồn tại")
        bot_dir = Path(bot["dir_path"])
        if not bot_dir.is_dir():
            raise FileNotFoundError(f"Thư mục của bot '{bot_id}' không tồn tại")

        # Determine source dir (prefer student/ if it exists and has bot.py, else bot_dir)
        source_dir = bot_dir / "student" if (bot_dir / "student" / "bot.py").is_file() else bot_dir

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob("*.py"):
                if "__pycache__" in file_path.parts:
                    continue
                arcname = file_path.relative_to(source_dir)
                zf.write(file_path, arcname=str(arcname))

        buffer.seek(0)
        bot_slug = re.sub(r"[^\w\-]", "_", bot.get("name", bot_id))
        filename = f"{bot_slug}_student.zip"
        return buffer, filename

    def validate_bot_directory(self, target_dir: Path) -> None:
        """Validates that target_dir has a valid bot.py implementing StudentBot."""
        bot_file = None
        if (target_dir / "student" / "bot.py").is_file():
            bot_file = target_dir / "student" / "bot.py"
        elif (target_dir / "bot.py").is_file():
            bot_file = target_dir / "bot.py"

        if bot_file is None:
            raise ValueError(
                "Thư mục tải lên không hợp lệ: Không tìm thấy file 'bot.py'. "
                "Vui lòng chọn thư mục chứa mã nguồn bot."
            )

        # Syntax check and import validation in a clean subprocess
        target_dir_escaped = str(target_dir.resolve()).replace("\\", "/")
        code = f"""
import sys
import importlib
from pathlib import Path

bot_dir = Path(r'{target_dir_escaped}').resolve()
student_dir = bot_dir / 'student'

# Clear any cached imports
for m in list(sys.modules.keys()):
    if m == 'student' or m.startswith('student.') or m in ['bot', 'search', 'evaluation', 'time_manager']:
        del sys.modules[m]

# Add both paths to sys.path
for p in [str(student_dir), str(bot_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Try importing as student.bot, then direct bot
m = None
try:
    m = importlib.import_module('student.bot')
except Exception as e1:
    try:
        m = importlib.import_module('bot')
    except Exception as e2:
        raise RuntimeError(f'Lỗi nạp module bot:\\n- student.bot: {{e1}}\\n- bot: {{e2}}')

assert hasattr(m, 'StudentBot'), 'Class StudentBot không tồn tại trong bot.py'
import inspect
sig = inspect.signature(m.StudentBot.__init__)
assert 'config' in sig.parameters, 'StudentBot.__init__ phải nhận tham số config'
assert hasattr(m.StudentBot, 'choose_move'), 'StudentBot phải có hàm choose_move'
"""
        import subprocess

        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            raise ValueError(f"Kiểm tra mã nguồn bot thất bại:\n{err_msg}")

    def save_uploaded_files(
        self,
        name: str,
        files: list[tuple[str, bytes]],
        description: str = "",
    ) -> dict[str, Any]:
        """Saves uploaded files to managed_bots/{bot_id}/ and updates registry."""
        name = name.strip()
        if not name:
            raise ValueError("Tên bot không được để trống")
        if not files:
            raise ValueError("Không có file nào được tải lên")

        # Determine clean bot_id based on name (e.g. "Bot V4" -> "bot_V4")
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        if clean_name:
            if not clean_name.lower().startswith("bot_"):
                base_id = f"bot_{clean_name}"
            else:
                base_id = clean_name
        else:
            base_id = f"bot_{uuid.uuid4().hex[:8]}"

        existing_ids = {b.get("id", "").lower() for b in self._load_registry().get("bots", [])}
        existing_dirs = {p.name.lower() for p in MANAGED_BOTS_DIR.iterdir() if p.is_dir()} if MANAGED_BOTS_DIR.is_dir() else set()
        bot_id = base_id
        counter = 1
        while bot_id.lower() in existing_ids or bot_id.lower() in existing_dirs:
            bot_id = f"{base_id}_{counter}"
            counter += 1

        target_dir = MANAGED_BOTS_DIR / bot_id
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Locate bot.py in uploaded files
            bot_rel_path = None
            for rpath, _ in files:
                norm = rpath.replace("\\", "/").strip("/")
                if norm == "bot.py" or norm.endswith("/bot.py"):
                    bot_rel_path = norm
                    break

            if not bot_rel_path:
                raise ValueError(
                    "Thư mục tải lên không hợp lệ: Không tìm thấy file 'bot.py'. "
                    "Vui lòng chọn thư mục chứa mã nguồn bot."
                )

            # Prefix of the folder containing bot.py (e.g. 'student' or 'chesslab-template/student' or '')
            prefix_parts = bot_rel_path.split("/")[:-1]
            prefix_str = "/".join(prefix_parts)

            for rel_path, content in files:
                norm = rel_path.replace("\\", "/").strip("/")
                # Relative subpath inside the bot package
                if prefix_str and norm.startswith(prefix_str + "/"):
                    subpath = norm[len(prefix_str) + 1:]
                else:
                    subpath = norm

                # 1. Write to target_dir/student/<subpath>
                dest_student = target_dir / "student" / Path(subpath)
                dest_student.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_student, "wb") as f:
                    f.write(content)

                # 2. Write to target_dir/<subpath> (at root of target_dir)
                dest_root = target_dir / Path(subpath)
                dest_root.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_root, "wb") as f:
                    f.write(content)

            # Ensure __init__.py exists in both places
            init_student = target_dir / "student" / "__init__.py"
            if not init_student.exists():
                init_student.touch()
            init_root = target_dir / "__init__.py"
            if not init_root.exists():
                init_root.touch()

            # Validate bot implementation
            self.validate_bot_directory(target_dir)

            # Record in registry
            new_bot = {
                "id": bot_id,
                "name": name,
                "dir_name": bot_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_default": False,
                "description": description or f"Bot tải lên từ thư mục bởi người dùng lúc {datetime.now().strftime('%H:%M %d/%m/%Y')}",
            }
            registry = self._load_registry()
            registry.setdefault("bots", []).append(new_bot)
            self._save_registry(registry)

            new_bot["dir_path"] = str(target_dir)
            new_bot["is_valid"] = True
            return new_bot

        except Exception as exc:
            # Clean up on failure
            if target_dir.exists():
                try:
                    shutil.rmtree(target_dir)
                except Exception:
                    pass
            raise exc

