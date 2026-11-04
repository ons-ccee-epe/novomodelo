//! Legacy-input overlays: for a deck whose committed files were migrated, the
//! directory holding the files the legacy deck had, plus `absent.json`, a JSON
//! array of the relative paths it did not have.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const OVERLAY_ROOT: &str = "tests/gate_fixtures/legacy_inputs";
const ABSENT_FILE: &str = "absent.json";

/// The overlay directory name of a deck key: each `/` becomes `__`.
#[must_use]
pub fn flatten_key(deck_key: &str) -> String {
    deck_key.replace('/', "__")
}

/// The absolute overlay directory of a deck key.
#[must_use]
pub fn overlay_dir(deck_key: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(OVERLAY_ROOT)
        .join(flatten_key(deck_key))
}

/// Writes each overlay file over `deck_copy`, then deletes each path listed in the
/// overlay's `absent.json`; a listed path the copy does not hold is skipped.
///
/// # Errors
///
/// Returns an error when the overlay or its `absent.json` is missing, when
/// `absent.json` is not a JSON array of strings (nothing is written then), or on
/// any I/O failure.
pub fn materialize(overlay: &Path, deck_copy: &Path) -> io::Result<()> {
    let absent: Vec<String> = serde_json::from_slice(&fs::read(overlay.join(ABSENT_FILE))?)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut files = Vec::new();
    overlay_files(overlay, Path::new(""), &mut files)?;
    files.sort();
    for relative in files
        .iter()
        .filter(|relative| relative.as_path() != Path::new(ABSENT_FILE))
    {
        let target = deck_copy.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(overlay.join(relative), target)?;
    }

    for relative in absent {
        match fs::remove_file(deck_copy.join(relative)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    Ok(())
}

fn overlay_files(root: &Path, relative: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root.join(relative))? {
        let entry = entry?;
        let path = relative.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            overlay_files(root, &path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::ErrorKind;
    use std::path::Path;

    use tempfile::TempDir;

    use super::materialize;

    fn write(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    fn read(root: &Path, relative: &str) -> String {
        fs::read_to_string(root.join(relative)).unwrap()
    }

    #[test]
    fn materialize_overwrites_and_adds_overlay_files_but_not_the_absent_list() {
        let overlay = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(overlay.path(), "absent.json", "[]");
        write(overlay.path(), "a.json", "legacy");
        write(overlay.path(), "sub/new.json", "legacy new");
        write(copy.path(), "a.json", "migrated");
        write(copy.path(), "sub/kept.json", "migrated");

        materialize(overlay.path(), copy.path()).unwrap();

        assert_eq!(read(copy.path(), "a.json"), "legacy");
        assert_eq!(read(copy.path(), "sub/new.json"), "legacy new");
        assert_eq!(read(copy.path(), "sub/kept.json"), "migrated");
        assert!(!copy.path().join("absent.json").exists());
    }

    #[test]
    fn materialize_deletes_each_listed_path() {
        let overlay = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(
            overlay.path(),
            "absent.json",
            r#"["post_study_stages.json", "source/post_study_stages.json"]"#,
        );
        write(copy.path(), "post_study_stages.json", "migrated");
        write(copy.path(), "source/post_study_stages.json", "migrated");
        write(copy.path(), "source/config.json", "kept");

        materialize(overlay.path(), copy.path()).unwrap();

        assert!(!copy.path().join("post_study_stages.json").exists());
        assert!(!copy.path().join("source/post_study_stages.json").exists());
        assert_eq!(read(copy.path(), "source/config.json"), "kept");
    }

    #[test]
    fn materialize_skips_a_listed_path_the_copy_does_not_hold() {
        let overlay = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(overlay.path(), "absent.json", r#"["missing.json"]"#);
        write(copy.path(), "kept.json", "kept");

        materialize(overlay.path(), copy.path()).unwrap();

        assert_eq!(read(copy.path(), "kept.json"), "kept");
    }

    #[test]
    fn materialize_rejects_a_malformed_list_and_a_missing_overlay() {
        let overlay = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(overlay.path(), "absent.json", r#"{"not": "an array"}"#);
        write(overlay.path(), "a.json", "legacy");
        write(copy.path(), "a.json", "migrated");

        let malformed = materialize(overlay.path(), copy.path()).unwrap_err();
        assert_eq!(malformed.kind(), ErrorKind::InvalidData);
        assert_eq!(read(copy.path(), "a.json"), "migrated");

        let missing = materialize(&overlay.path().join("absent"), copy.path());
        assert!(missing.is_err());
    }
}
