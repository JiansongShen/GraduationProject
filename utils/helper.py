def format_time(seconds: float) -> str:
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    return f"{seconds:.1f}s"


def count_files(directory: str, pattern: str = "*.nii.gz") -> int:
    from pathlib import Path
    return len(list(Path(directory).rglob(pattern)))

def find_all_file_paths_recursively(directory: str, pattern: str = "*.nii.gz") -> list[str]:
    from pathlib import Path
    return [str(path) for path in Path(directory).rglob(pattern)]

