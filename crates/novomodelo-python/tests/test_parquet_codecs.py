"""Compression codecs of the Parquet files novomodelo reads.

Input files of a case and the result files of a run are read whether they are
uncompressed or compressed with zstd, Snappy, gzip or LZ4; Brotli is refused.
pyarrow writes `LZ4_RAW` for `compression="lz4"`; the legacy `LZ4` framing is
covered by the reader's Rust tests, which can write it.

Run with (from the repo root):
    pytest crates/novomodelo-python/tests/test_parquet_codecs.py
"""

from __future__ import annotations

import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[3]
VALID_CASE = _REPO_ROOT / "examples" / "1dtoy"

ACCEPTED_COMPRESSIONS = ["none", "zstd", "snappy", "gzip", "lz4"]
BROTLI_REFUSAL = "Disabled feature at compile time: brotli"


def _rewrite(path: pathlib.Path, compression: str) -> None:
    pq.write_table(pq.read_table(path), path, compression=compression)


def _case_with_inputs_compressed(dest: pathlib.Path, compression: str) -> pathlib.Path:
    shutil.copytree(VALID_CASE, dest)
    for parquet_file in sorted((dest / "scenarios").glob("*.parquet")):
        _rewrite(parquet_file, compression)
    return dest


def _lower_bounds(output_dir: pathlib.Path) -> list[float]:
    import novomodelo.results

    return [
        row["lower_bound"] for row in novomodelo.results.load_convergence(output_dir)
    ]


def test_version_info_codecs_are_the_compressions_these_tests_read() -> None:
    """Every codec `version_info` lists is one this module reads a file in."""
    import novomodelo

    assert set(novomodelo.version_info()["codecs"]) == set(ACCEPTED_COMPRESSIONS) - {
        "none"
    }


@pytest.fixture(scope="module")
def uncompressed_lower_bounds(
    tmp_path_factory: pytest.TempPathFactory,
) -> list[float]:
    import novomodelo.run

    work = tmp_path_factory.mktemp("codec_reference")
    case = _case_with_inputs_compressed(work / "case", "none")
    novomodelo.run.run(str(case), output_dir=str(work / "out"))
    return _lower_bounds(work / "out")


@pytest.mark.parametrize("compression", ["zstd", "snappy", "gzip", "lz4"])
def test_a_compressed_input_gives_the_run_of_the_uncompressed_input(
    compression: str, uncompressed_lower_bounds: list[float], tmp_path: pathlib.Path
) -> None:
    """A case whose input Parquet is compressed validates and trains to the
    lower bounds of the same case stored uncompressed."""
    import novomodelo.io
    import novomodelo.run

    case = _case_with_inputs_compressed(tmp_path / "case", compression)

    assert novomodelo.io.validate(str(case))["valid"] is True
    novomodelo.run.run(str(case), output_dir=str(tmp_path / "out"))
    assert uncompressed_lower_bounds
    assert _lower_bounds(tmp_path / "out") == uncompressed_lower_bounds


def test_a_brotli_input_is_refused_naming_the_file_and_the_codec(
    tmp_path: pathlib.Path,
) -> None:
    """A Brotli-compressed input file fails validation with the codec named."""
    import novomodelo.io

    case = _case_with_inputs_compressed(tmp_path / "case", "brotli")

    result = novomodelo.io.validate(str(case))

    assert result["valid"] is False
    messages = " ".join(error["message"] for error in result["errors"])
    assert "scenarios/inflow_seasonal_stats.parquet" in messages
    assert BROTLI_REFUSAL in messages


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    import novomodelo.run

    output_dir = tmp_path_factory.mktemp("codec_results")
    novomodelo.run.run(str(VALID_CASE), output_dir=str(output_dir))
    return output_dir


def _output_copy_with_results_compressed(
    run_output: pathlib.Path, dest: pathlib.Path, compression: str
) -> pathlib.Path:
    shutil.copytree(run_output, dest)
    _rewrite(dest / "training" / "convergence.parquet", compression)
    for parquet_file in sorted((dest / "simulation" / "costs").rglob("*.parquet")):
        _rewrite(parquet_file, compression)
    return dest


@pytest.mark.parametrize("compression", ACCEPTED_COMPRESSIONS)
def test_compressed_results_load_as_the_results_the_run_wrote(
    compression: str, run_output: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The result loaders return the same rows whichever accepted codec the
    result files are stored in."""
    import novomodelo.results

    recompressed = _output_copy_with_results_compressed(
        run_output, tmp_path / "out", compression
    )

    assert novomodelo.results.load_convergence(
        recompressed
    ) == novomodelo.results.load_convergence(run_output)
    assert pa.table(novomodelo.results.load_convergence_arrow(recompressed)).equals(
        pa.table(novomodelo.results.load_convergence_arrow(run_output))
    )
    assert (
        novomodelo.results.load_simulation(recompressed)["costs"]
        == novomodelo.results.load_simulation(run_output)["costs"]
    )
    assert pa.table(
        novomodelo.results.load_simulation_arrow(recompressed)["costs"]
    ).equals(pa.table(novomodelo.results.load_simulation_arrow(run_output)["costs"]))


def test_brotli_results_are_refused_when_the_first_batch_is_read(
    run_output: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A Brotli-compressed result file opens (the footer is uncompressed) and
    fails on its first batch, with the codec named."""
    import novomodelo.results

    recompressed = _output_copy_with_results_compressed(
        run_output, tmp_path / "out", "brotli"
    )

    with pytest.raises(OSError, match="error reading Parquet batch") as refusal:
        novomodelo.results.load_convergence(recompressed)
    assert BROTLI_REFUSAL in str(refusal.value)
    with pytest.raises(OSError, match=BROTLI_REFUSAL):
        novomodelo.results.load_convergence_arrow(recompressed)
    with pytest.raises(OSError, match=BROTLI_REFUSAL):
        novomodelo.results.load_simulation(recompressed)
