"""Step 2: Verify ABC files are valid and remove corrupted ones before merging."""
from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import re
import subprocess


ROOT = Path(__file__).resolve().parent.parent
ABC_DIR = ROOT / "data" / "raw"  # Directory containing converted ABC files

MIN_SIZE: int = 150      # Minimum file size (bytes) - smaller files are likely metadata only
MAX_SIZE: int = 300_000  # Maximum file size (bytes) - larger files may be corrupted
MIN_NOTES: int = 10      # Minimum note count to ensure actual music content


def has_required_headers(content: str) -> bool:
    """Check for required ABC headers."""
    has_x = bool(re.search(r"^X:\s*\d+", content, re.MULTILINE))
    has_k = bool(re.search(r"^K:\s*\w+", content, re.MULTILINE))
    return has_x or has_k


def has_music_content(content: str) -> bool:
    """Check for actual music notation (notes)."""
    # ABC notes: A-G, a-g, with optional accidentals and octave markers
    notes = re.findall(r"[A-Ga-g][,']*", content)
    return len(notes) >= MIN_NOTES


def is_valid_structure(content: str) -> bool:
    """Check ABC has valid structure (not just garbage)."""
    # Should have some bar lines
    has_bars = "|" in content
    # Should not be mostly non-printable
    printable_ratio = sum(1 for c in content if c.isprintable() or c in "\n\t") / max(1, len(content))
    return has_bars and printable_ratio > 0.95


def can_convert_to_midi(abc_path: Path) -> bool:
    """
    Try to convert ABC back to MIDI using abc2midi.
    This is the ultimate validation - if it can round-trip, it's valid.
    """
    try:
        result = subprocess.run(
            ["abc2midi", str(abc_path), "-o", "/dev/null"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_abc_file(abc_path: Path, strict: bool = False) -> tuple[bool, str]:
    """
    Validate a single ABC file.
    
    Args:
        abc_path: Path to ABC file
        strict: If True, also try abc2midi conversion
    
    Returns:
        (is_valid, reason)
    """
    try:
        size = abc_path.stat().st_size
        
        if size < MIN_SIZE:
            return False, "too_small"
        if size > MAX_SIZE:
            return False, "too_large"
        
        content = abc_path.read_text(encoding="utf-8", errors="ignore")
        
        if not has_required_headers(content):
            return False, "missing_headers"
        
        if not has_music_content(content):
            return False, "no_notes"
        
        if not is_valid_structure(content):
            return False, "bad_structure"
        
        if strict and not can_convert_to_midi(abc_path):
            return False, "midi_fail"
        
        return True, "ok"
        
    except Exception as e:
        return False, f"error:{type(e).__name__}"


def main(strict: bool = False, delete_invalid: bool = True) -> dict:
    """
    Verify all ABC files in the raw directory.
    
    Args:
        strict: If True, verify files can convert back to MIDI
        delete_invalid: If True, delete invalid files
    
    Returns:
        Statistics dict
    """
    print("=== ABC Verification ===")
    print(f"Directory: {ABC_DIR}")
    print(f"Strict mode: {strict}")
    print(f"Delete invalid: {delete_invalid}")
    
    if not ABC_DIR.exists():
        raise FileNotFoundError(f"Missing ABC directory: {ABC_DIR}")
    
    abc_files = list(ABC_DIR.glob("*.abc"))
    print(f"Found {len(abc_files)} ABC files")
    
    if not abc_files:
        return {"total": 0, "valid": 0, "invalid": 0}
    
    stats = {
        "total": len(abc_files),
        "valid": 0,
        "invalid": 0,
        "reasons": {},
    }
    
    invalid_files = []
    
    for abc_path in tqdm(abc_files, desc="Verifying"):
        is_valid, reason = validate_abc_file(abc_path, strict=strict)
        
        if is_valid:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            invalid_files.append(abc_path)
    
    # Delete invalid files
    if delete_invalid and invalid_files:
        print(f"\nDeleting {len(invalid_files)} invalid files...")
        for p in invalid_files:
            p.unlink(missing_ok=True)
    
    # Report
    print(f"\n{'='*40}")
    print(f"Total:   {stats['total']:,}")
    print(f"Valid:   {stats['valid']:,} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"Invalid: {stats['invalid']:,} ({100*stats['invalid']/stats['total']:.1f}%)")
    
    if stats["reasons"]:
        print("\nInvalid reasons:")
        for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count:,}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify ABC files")
    parser.add_argument("--strict", action="store_true", help="Also verify abc2midi conversion")
    parser.add_argument("--no-delete", action="store_true", help="Don't delete invalid files")
    args = parser.parse_args()
    
    main(strict=args.strict, delete_invalid=not args.no_delete)
