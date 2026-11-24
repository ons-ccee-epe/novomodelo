//! Unmigrated-input overlays: for a deck whose committed files were migrated, the
//! directory holding the files the deck had before migration, plus `absent.json`,
//! a JSON array of the relative paths it did not have.
//!
//! Overlays are grouped by [`Generation`], one per migration, under
//! `unmigrated_inputs/<generation>/<flat deck key>/`. An overlay patches the deck as
//! the next-newer generation left it, so materializing generation `g` applies
//! every newer generation first, newest to oldest.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const OVERLAY_ROOT: &str = "tests/unswept_fixtures/unmigrated_inputs";
const ABSENT_FILE: &str = "absent.json";
const LEGACY_MARKER: &str = "initial_conditions.json";
const MIGRATED_MARKER: &str = "boundary_conditions.json";

/// A migration whose pre-migration files an overlay restores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generation {
    /// The `stages` key of `post_study_stages.json`, or no such file.
    PostStudyStagesKey,
    /// `initial_conditions.json`, `stages.json` and `post_study_stages.json`
    /// before they moved to `boundary_conditions.json`.
    InitialConditions,
}

impl Generation {
    const NEWEST_FIRST: [Self; 2] = [Self::InitialConditions, Self::PostStudyStagesKey];

    fn dir_name(self) -> &'static str {
        match self {
            Self::PostStudyStagesKey => "post_study_stages_key",
            Self::InitialConditions => "initial_conditions",
        }
    }
}

fn flatten_key(deck_key: &str) -> String {
    deck_key.replace('/', "__")
}

fn default_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(OVERLAY_ROOT)
}

/// Patches `deck_copy` into the legacy twin of `deck_key` at `generation`: applies,
/// newest first, each newer generation that has an overlay for the deck, then the
/// overlay of `generation`, then checks the result with [`check_legacy_twin`].
///
/// # Errors
///
/// Returns `ErrorKind::NotFound` when `generation` has no overlay for the deck
/// (nothing is written then), `ErrorKind::InvalidData` when an `absent.json` is
/// not a JSON array of strings or the result is not a legacy twin, or any I/O
/// failure.
pub fn materialize_twin(
    generation: Generation,
    deck_key: &str,
    deck_copy: &Path,
) -> io::Result<()> {
    materialize_twin_in(&default_root(), generation, deck_key, deck_copy)
}

fn materialize_twin_in(
    root: &Path,
    generation: Generation,
    deck_key: &str,
    deck_copy: &Path,
) -> io::Result<()> {
    let flat_key = flatten_key(deck_key);
    let overlay = |g: Generation| root.join(g.dir_name()).join(&flat_key);
    let own = overlay(generation);
    if !own.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no {} overlay for {deck_key}", generation.dir_name()),
        ));
    }
    let newer = Generation::NEWEST_FIRST
        .into_iter()
        .take_while(|newer| *newer != generation)
        .map(overlay)
        .filter(|dir| dir.is_dir());
    for dir in newer.chain([own]) {
        materialize(&dir, deck_copy)?;
    }
    check_legacy_twin(deck_copy)
}

/// Succeeds when `deck` holds `initial_conditions.json` and no
/// `boundary_conditions.json`.
///
/// # Errors
///
/// Returns `ErrorKind::InvalidData` naming the offending file.
pub fn check_legacy_twin(deck: &Path) -> io::Result<()> {
    let not_legacy = |reason: String| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not a legacy twin: {reason}", deck.display()),
        )
    };
    if !deck.join(LEGACY_MARKER).exists() {
        return Err(not_legacy(format!("{LEGACY_MARKER} is missing")));
    }
    if deck.join(MIGRATED_MARKER).exists() {
        return Err(not_legacy(format!("{MIGRATED_MARKER} is present")));
    }
    Ok(())
}

/// The sorted deck keys that have an overlay in `generation`.
///
/// # Panics
///
/// Panics when the generation's directory exists and cannot be read.
#[must_use]
pub fn deck_keys(generation: Generation) -> Vec<String> {
    deck_keys_in(&default_root(), generation)
}

#[expect(
    clippy::panic,
    reason = "an unreadable overlay directory is a broken checkout, not a runtime condition to recover from"
)]
fn deck_keys_in(root: &Path, generation: Generation) -> Vec<String> {
    let dir = root.join(generation.dir_name());
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut keys: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|entry| entry.unwrap_or_else(|e| panic!("dir entry under {}: {e}", dir.display())))
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().replace("__", "/"))
        .collect();
    keys.sort();
    keys
}

/// Writes each overlay file over `deck_copy`, then deletes each path listed in the
/// overlay's `absent.json`; a listed path the copy does not hold is skipped.
///
/// # Errors
///
/// Returns an error when the overlay or its `absent.json` is missing, when
/// `absent.json` is not a JSON array of strings (nothing is written then), or on
/// any I/O failure.
fn materialize(overlay: &Path, deck_copy: &Path) -> io::Result<()> {
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

    use super::{
        Generation, check_legacy_twin, deck_keys_in, flatten_key, materialize, materialize_twin_in,
    };

    fn write(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    fn read(root: &Path, relative: &str) -> String {
        fs::read_to_string(root.join(relative)).unwrap()
    }

    const DECK: &str = "examples/deck";

    fn write_overlay(root: &Path, generation: Generation, deck: &str, files: &[(&str, &str)]) {
        let dir = format!("{}/{}", generation.dir_name(), flatten_key(deck));
        for (relative, content) in files {
            write(root, &format!("{dir}/{relative}"), content);
        }
    }

    fn migrated_copy() -> TempDir {
        let copy = TempDir::new().unwrap();
        write(copy.path(), "boundary_conditions.json", "migrated");
        write(copy.path(), "post_study_stages.json", "migrated");
        copy
    }

    #[test]
    fn materialize_overwrites_and_adds_overlay_files_but_not_the_absent_list() {
        let overlay = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(overlay.path(), "absent.json", "[]");
        write(overlay.path(), "a.json", "overlay");
        write(overlay.path(), "sub/new.json", "overlay new");
        write(copy.path(), "a.json", "migrated");
        write(copy.path(), "sub/kept.json", "migrated");

        materialize(overlay.path(), copy.path()).unwrap();

        assert_eq!(read(copy.path(), "a.json"), "overlay");
        assert_eq!(read(copy.path(), "sub/new.json"), "overlay new");
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
        write(overlay.path(), "a.json", "overlay");
        write(copy.path(), "a.json", "migrated");

        let malformed = materialize(overlay.path(), copy.path()).unwrap_err();
        assert_eq!(malformed.kind(), ErrorKind::InvalidData);
        assert_eq!(read(copy.path(), "a.json"), "migrated");

        let missing = materialize(&overlay.path().join("absent"), copy.path());
        assert!(missing.is_err());
    }

    #[test]
    fn newer_generation_is_applied_before_the_requested_one() {
        let root = TempDir::new().unwrap();
        let copy = migrated_copy();
        write_overlay(
            root.path(),
            Generation::InitialConditions,
            DECK,
            &[
                ("absent.json", r#"["boundary_conditions.json"]"#),
                ("initial_conditions.json", "ic"),
                ("stages.json", "newer stages"),
                ("post_study_stages.json", "newer"),
            ],
        );
        write_overlay(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            &[
                ("absent.json", r#"["stages.json"]"#),
                ("post_study_stages.json", "older"),
            ],
        );

        materialize_twin_in(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            copy.path(),
        )
        .unwrap();

        assert_eq!(read(copy.path(), "post_study_stages.json"), "older");
        assert_eq!(read(copy.path(), "initial_conditions.json"), "ic");
        assert!(!copy.path().join("stages.json").exists());
        assert!(!copy.path().join("boundary_conditions.json").exists());
    }

    #[test]
    fn missing_newer_overlay_is_skipped() {
        let root = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(copy.path(), "initial_conditions.json", "ic");
        write(copy.path(), "post_study_stages.json", "migrated");
        write_overlay(
            root.path(),
            Generation::InitialConditions,
            "examples/other",
            &[("absent.json", "[]"), ("other.json", "other")],
        );
        write_overlay(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            &[("absent.json", "[]"), ("post_study_stages.json", "older")],
        );

        materialize_twin_in(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            copy.path(),
        )
        .unwrap();

        assert_eq!(read(copy.path(), "post_study_stages.json"), "older");
        assert!(!copy.path().join("other.json").exists());
    }

    #[test]
    fn requested_generation_without_overlay_is_not_found() {
        let root = TempDir::new().unwrap();
        let copy = migrated_copy();
        write_overlay(
            root.path(),
            Generation::InitialConditions,
            DECK,
            &[("absent.json", "[]"), ("initial_conditions.json", "ic")],
        );

        let err = materialize_twin_in(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            copy.path(),
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert!(!copy.path().join("initial_conditions.json").exists());
    }

    #[test]
    fn twin_holding_boundary_conditions_is_rejected() {
        let root = TempDir::new().unwrap();
        let copy = migrated_copy();
        write(copy.path(), "initial_conditions.json", "ic");
        write_overlay(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            &[("absent.json", "[]")],
        );

        let err = materialize_twin_in(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            copy.path(),
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("boundary_conditions.json"),
            "{err}"
        );
    }

    #[test]
    fn twin_without_initial_conditions_is_rejected() {
        let root = TempDir::new().unwrap();
        let copy = TempDir::new().unwrap();
        write(copy.path(), "post_study_stages.json", "migrated");
        write_overlay(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            &[("absent.json", "[]")],
        );

        let err = materialize_twin_in(
            root.path(),
            Generation::PostStudyStagesKey,
            DECK,
            copy.path(),
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("initial_conditions.json"), "{err}");
        assert!(check_legacy_twin(copy.path()).is_err());
    }

    #[test]
    fn deck_keys_lists_the_sorted_unflattened_keys_of_one_generation() {
        let root = TempDir::new().unwrap();
        for deck in ["examples/b", "crates/a/tests/c"] {
            write_overlay(
                root.path(),
                Generation::PostStudyStagesKey,
                deck,
                &[("absent.json", "[]")],
            );
        }

        assert_eq!(
            deck_keys_in(root.path(), Generation::PostStudyStagesKey),
            ["crates/a/tests/c", "examples/b"]
        );
        assert!(deck_keys_in(root.path(), Generation::InitialConditions).is_empty());
    }
}
