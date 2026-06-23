from __future__ import annotations

import os
import tarfile
import zipfile

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata
from tools.filesystem import _safe_path, WORKSPACE_DIR

@tool
def create_archive(paths: list[str], archive_name: str) -> str:
    """Create a zip archive containing the specified paths."""
    archive_path = _safe_path(archive_name)
    if not str(archive_path).endswith(".zip"):
        return "Error: archive_name must end with .zip"
        
    targets = []
    for p in paths:
        target = _safe_path(p)
        if not target.exists():
            return f"Error: path not found: {target}"
        targets.append(target)
        
    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for target in targets:
                if target.is_file():
                    arcname = target.relative_to(WORKSPACE_DIR)
                    zipf.write(target, arcname)
                elif target.is_dir():
                    for p in target.rglob("*"):
                        if p.is_file():
                            arcname = p.relative_to(WORKSPACE_DIR)
                            zipf.write(p, arcname)
        return f"Created archive: {archive_path.relative_to(WORKSPACE_DIR)}"
    except Exception as e:
        return f"Error: {e}"

@tool
def extract_archive(archive_path: str, dest: str = ".") -> str:
    """Extract a zip or tar archive to the specified destination."""
    archive = _safe_path(archive_path)
    if not archive.exists():
        return f"Error: archive not found: {archive}"
        
    dest_dir = _safe_path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if str(archive).endswith(".zip"):
            with zipfile.ZipFile(archive, "r") as zipf:
                for member in zipf.infolist():
                    member_path = (dest_dir / member.filename).resolve()
                    if dest_dir not in member_path.parents and dest_dir != member_path:
                        return f"Error: archive contains unsafe path: {member.filename}"
                zipf.extractall(dest_dir)
        elif str(archive).endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
            with tarfile.open(archive, "r:*") as tar:
                for member in tar.getmembers():
                    member_path = (dest_dir / member.name).resolve()
                    if dest_dir not in member_path.parents and dest_dir != member_path:
                        return f"Error: archive contains unsafe path: {member.name}"
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner)
                safe_extract(tar, dest_dir)
        else:
            return "Error: unsupported archive format"
            
        return f"Extracted {archive.relative_to(WORKSPACE_DIR)} to {dest_dir.relative_to(WORKSPACE_DIR)}"
    except Exception as e:
        return f"Error: {e}"

register_lc_tool(create_archive, metadata=ToolMetadata(risk_level="moderate"))
register_lc_tool(extract_archive, metadata=ToolMetadata(risk_level="moderate"))
