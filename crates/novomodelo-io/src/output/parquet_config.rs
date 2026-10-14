//! Frozen Parquet writer encoding parameters shared by every output writer.
//!
//! Process-lifetime constants resolved once via [`WRITER_PROPERTIES`]; writers
//! clone the handle instead of rebuilding per file.

use std::sync::LazyLock;

use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

const ROW_GROUP_SIZE: usize = 100_000;
const DICTIONARY_ENCODING: bool = true;

fn compression() -> Compression {
    #[allow(clippy::expect_used)]
    let zstd_level = ZstdLevel::try_new(3).expect("ZstdLevel 3 is always valid");
    Compression::ZSTD(zstd_level)
}

pub(crate) static WRITER_PROPERTIES: LazyLock<WriterProperties> = LazyLock::new(|| {
    WriterProperties::builder()
        .set_compression(compression())
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .set_dictionary_enabled(DICTIONARY_ENCODING)
        .build()
});

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use parquet::schema::types::ColumnPath;

    use super::*;

    #[test]
    fn frozen_constants_match_former_default() {
        assert_eq!(ROW_GROUP_SIZE, 100_000);
        const { assert!(DICTIONARY_ENCODING, "dictionary encoding must stay enabled") };
        assert!(
            matches!(compression(), Compression::ZSTD(_)),
            "default compression must be ZSTD, got {:?}",
            compression()
        );
    }

    #[test]
    fn zstd_level_is_three() {
        let Compression::ZSTD(level) = compression() else {
            panic!("expected Compression::ZSTD, got {:?}", compression());
        };
        // ZstdLevel encodes as "ZSTD(ZstdLevel(<n>))" in Debug
        let debug = format!("{level:?}");
        assert!(debug.contains('3'), "ZSTD level must be 3, got: {debug}");
    }

    #[test]
    fn writer_properties_reflects_frozen_constants() {
        let col = ColumnPath::from("any_column");
        assert_eq!(
            WRITER_PROPERTIES.max_row_group_row_count(),
            Some(ROW_GROUP_SIZE)
        );
        assert_eq!(
            WRITER_PROPERTIES.dictionary_enabled(&col),
            DICTIONARY_ENCODING
        );
        assert!(matches!(
            WRITER_PROPERTIES.compression(&col),
            Compression::ZSTD(_)
        ));
    }
}
