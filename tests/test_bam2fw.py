# test_bam2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig bam2fw`, against `figwig bam2bw` on the same inputs.

`figwig bam2fw` counts what `figwig bam2bw` counts, so each file's tracks in
the figwig file must read as the bigWigs `figwig bam2bw` writes from that
file alone, under the same options.
"""

import os
import sys
import subprocess

import numpy
import pytest

if sys.platform == 'win32':
	pytest.skip("pysam does not build on Windows", allow_module_level=True)

import figwig.bam2bw  # noqa: E402
import figwig.bam2fw  # noqa: E402

from figwig import BigWigReader  # noqa: E402
from figwig import FigWigReader  # noqa: E402

from figwig.__main__ import main  # noqa: E402

from .reads import write_bam  # noqa: E402
from .reads import write_chrom_sizes  # noqa: E402
from .reads import write_intervals  # noqa: E402

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
	MKL_NUM_THREADS="1")

CHROM_SIZES = [("chr1", 1000), ("chr2", 500), ("chr3", 200)]
BAM_CHROM_SIZES = CHROM_SIZES + [("chrUN", 100)]

READS = [
	("chr1", 100, 50, False),
	("chr1", 100, 50, False),
	("chr1", 200, 30, False),
	("chr1", 300, 40, True),
	("chr1", 300, 40, True),
	("chr1", 400, 20, True),
	("chr2", 50, 25, False),
	("chr2", 80, 10, True),
	("chrUN", 10, 10, False),
]
OTHER_READS = [
	("chr1", 120, 50, False),
	("chr1", 980, 30, True),
	("chr2", 5, 25, False),
	("chr2", 490, 10, True),
	("chr3", 0, 10, False),
]
UNMAPPED = [("chr1", 500, 20)]
INTERVALS = [
	("chr1", 100, 150),
	("chr1", 100, 150),
	("chr1", 200, 230),
	("chr2", 50, 75),
	("chrUN", 10, 20),
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
	return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sizes(data_dir):
	return str(write_chrom_sizes(data_dir / "test.chrom.sizes", CHROM_SIZES))


@pytest.fixture(scope="session")
def bam(data_dir):
	return str(write_bam(data_dir / "test.bam", BAM_CHROM_SIZES, READS,
		unmapped=UNMAPPED))


@pytest.fixture(scope="session")
def other_bam(data_dir):
	return str(write_bam(data_dir / "other.bam", BAM_CHROM_SIZES, OTHER_READS))


@pytest.fixture(scope="session")
def tsv_gz(data_dir):
	return str(write_intervals(data_dir / "test.tsv.gz", INTERVALS))


def bam2bw(tmp_path, filename, sizes, *flags):
	"""`figwig bam2bw` on one file, as the bigWigs it wrote."""

	name = str(tmp_path / ("bw_" + os.path.basename(filename)))
	figwig.bam2bw.main([filename, "-s", sizes, "-n", name] + list(flags),
		prog="figwig bam2bw")
	if "-u" in flags:
		return [name + ".bw"]
	return [name + ".+.bw", name + ".-.bw"]


def bam2fw(tmp_path, filenames, sizes, *flags):
	os.makedirs(tmp_path, exist_ok=True)
	name = str(tmp_path / "out")
	main(["bam2fw"] + list(filenames) + ["-s", sizes, "-n", name] + list(flags))
	return name + ".fw"


def bigwig_values(path, chrom, length):
	"""A bigWig's values over a chromosome, as zeros for a bigWig without
	data blocks, which `figwig bam2bw` writes for a strand with no counts
	and `BigWigReader` does not open."""

	reader = BigWigReader(path)
	if reader._data_index == 0:
		return numpy.zeros(length, dtype=numpy.float32)
	return reader.read(chrom, [0], length)[0]


def assert_tracks_match_bigwigs(path, names, bigwigs):
	reader = FigWigReader(path)
	for chrom, length in CHROM_SIZES:
		expected = numpy.stack([bigwig_values(bigwig, chrom, length) for bigwig
			in bigwigs])[None]
		y = reader.read(chrom, [0], length, tracks=names, dtype="float32")
		numpy.testing.assert_array_equal(y, expected)


@pytest.mark.parametrize("flags", [(), ("-u",), ("-3p",), ("-ps", "4", "-ns",
	"-5"), ("-sf", "0.5",), ("-r",), ("-sf", "2", "-r"), ("-f", "-u")])
def test_counts_what_bam2bw_counts(tmp_path, bam, sizes, flags):
	bigwigs = bam2bw(tmp_path, bam, sizes, *flags)
	path = bam2fw(tmp_path, [bam], sizes, *flags)
	names = ["test"] if "-u" in flags else ["test.+", "test.-"]
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == names
	assert reader.chrom_sizes == dict(CHROM_SIZES)
	assert_tracks_match_bigwigs(path, names, bigwigs)

	if "-sf" in flags or "-r" in flags:
		assert reader.tracks[0]["dtype"] == "float32"
	else:
		assert reader.tracks[0]["dtype"] == "uint8"
	if "-u" not in flags:
		assert reader.tracks[0]["strand"] == "+" and reader.tracks[1]["strand"] \
			== "-"
		assert [track["group"] for track in reader.tracks] == [0, 0]


def test_each_file_is_its_own_tracks(tmp_path, bam, other_bam, sizes):
	path = bam2fw(tmp_path, [bam, other_bam], sizes)
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["test.+", "test.-",
		"other.+", "other.-"]
	assert [track["group"] for track in reader.tracks] == [0, 0, 1, 1]
	assert_tracks_match_bigwigs(path, ["test.+", "test.-"], bam2bw(tmp_path, bam,
		sizes))
	assert_tracks_match_bigwigs(path, ["other.+", "other.-"], bam2bw(tmp_path,
		other_bam, sizes))


def test_read_depth_is_per_file(tmp_path, bam, other_bam, sizes):
	path = bam2fw(tmp_path, [bam, other_bam], sizes, "-r")
	assert_tracks_match_bigwigs(path, ["test.+", "test.-"], bam2bw(tmp_path, bam,
		sizes, "-r"))
	assert_tracks_match_bigwigs(path, ["other.+", "other.-"], bam2bw(tmp_path,
		other_bam, sizes, "-r"))


def test_fragments_file(tmp_path, tsv_gz, sizes):
	path = bam2fw(tmp_path, [tsv_gz], sizes, "-f")
	assert_tracks_match_bigwigs(path, ["test.+", "test.-"], bam2bw(tmp_path,
		tsv_gz, sizes, "-f"))


def test_metadata_table(tmp_path, bam, other_bam, sizes):
	table = tmp_path / "meta.tsv"
	table.write_text("file\tname\tassay\tbiosample\n"
		"test.bam\tK562_ATAC\tATAC-seq\tK562\n"
		"{}\t\tDNase-seq\tHepG2\n".format(other_bam))
	path = bam2fw(tmp_path, [bam, other_bam], sizes, "-m", str(table), "-g",
		"hg38", "-t", "256", "--dtype", "uint16", "-l", "0")
	reader = FigWigReader(path)
	assert reader.genome == "hg38" and reader.tile == 256
	assert [track["name"] for track in reader.tracks] == ["K562_ATAC.+",
		"K562_ATAC.-", "other.+", "other.-"]
	assert reader.tracks[0]["assay"] == "ATAC-seq"
	assert reader.tracks[1]["biosample"] == "K562"
	assert reader.tracks[2]["biosample"] == "HepG2"
	assert [track["dtype"] for track in reader.tracks] == ["uint16"] * 4
	assert_tracks_match_bigwigs(path, ["K562_ATAC.+", "K562_ATAC.-"], bam2bw(
		tmp_path, bam, sizes))


def test_discarded_entries_are_reported(tmp_path, bam, sizes, capsys):
	bam2fw(tmp_path, [bam], sizes, "-ps", "700")
	out = capsys.readouterr().out
	assert "fell outside the provided chromosome sizes" in out


def test_errors(tmp_path, bam, sizes):
	with pytest.raises(SystemExit, match="Filenames must end in"):
		bam2fw(tmp_path, [str(tmp_path / "x.txt")], sizes)
	table = tmp_path / "meta.tsv"
	table.write_text("file\tcell\nnope.bam\tK562\n")
	with pytest.raises(SystemExit, match="has no row for"):
		bam2fw(tmp_path, [bam], sizes, "-m", str(table))
	with pytest.raises(SystemExit, match="outside 0 to 255"):
		bam2fw(tmp_path, [bam], sizes, "--dtype", "uint8", "-sf", "200")
	with pytest.raises(SystemExit, match="No such file|not a file|cannot"):
		bam2fw(tmp_path, [bam], str(tmp_path / "none.sizes"))
	path = bam2fw(tmp_path, [bam], sizes)
	with pytest.raises(SystemExit, match="exists; --force writes over it"):
		bam2fw(tmp_path, [bam], sizes)
	assert bam2fw(tmp_path, [bam], sizes, "--force") == path


def test_helpers():
	assert figwig.bam2fw._base_name("/a/b/x.tsv.gz") == "x"
	assert figwig.bam2fw._base_name("x.bam") == "x"
	assert figwig.bam2fw._base_name("x.other") == "x.other"
	tracks, groups = figwig.bam2fw._tracks_for_files(["a.bam", "b.bam"], None,
		True, "uint32")
	assert tracks == [{"name": "a", "dtype": "uint32"}, {"name": "b", "dtype":
		"uint32"}]
	assert groups == [["a"], ["b"]]


def test_command_line(tmp_path, bam, sizes):
	name = str(tmp_path / "cli")
	result = subprocess.run([sys.executable, "-m", "figwig", "bam2fw", bam, "-s",
		sizes, "-n", name, "-p", "2"], env=ENV, capture_output=True, text=True,
		timeout=600)
	assert result.returncode == 0, result.stderr
	assert len(FigWigReader(name + ".fw").tracks) == 2


# --barcode-column: a fragments file as one track per cell.

FRAGMENTS = [
	("chr1", 100, 150, "AAAC-1", 2),
	("chr1", 100, 150, "AAAG-1", 1),
	("chr1", 120, 160, "AAAC-1", 1),
	("chr1", 200, 230, "CCCT-1", 1),
	("chr2", 50, 75, "AAAG-1", 3),
	("chr2", 60, 75, "AAAC-1", 1),
	("chr2", 480, 510, "CCCT-1", 1),
	("chrUN", 10, 20, "AAAC-1", 1),
]
BARCODES = ["AAAC-1", "AAAG-1", "CCCT-1"]


@pytest.fixture(scope="session")
def fragments(data_dir):
	path = data_dir / "cells.tsv"
	with open(path, "w") as handle:
		handle.write("# id=test\n#\n")
		for fields in FRAGMENTS:
			handle.write("\t".join(map(str, fields)) + "\n")
	return str(path)


@pytest.fixture(scope="session")
def fragments_gz(data_dir):
	return str(write_intervals(data_dir / "cells.tsv.gz", FRAGMENTS))


@pytest.fixture(scope="session")
def per_cell_bigwigs(tmp_path_factory, data_dir, sizes):
	"""`figwig bam2bw -f -u` on each barcode's fragments alone, per flag
	set: the oracle for a track of the per-cell file."""

	import functools

	@functools.lru_cache(maxsize=None)
	def bigwigs(flags):
		out = tmp_path_factory.mktemp("percell")
		paths = {}
		for barcode in BARCODES:
			path = write_intervals(out / (barcode + ".tsv"), [fields for fields in
				FRAGMENTS if fields[3] == barcode])
			paths[barcode] = bam2bw(out, str(path), sizes, "-u", *flags)[0]
		return paths

	return bigwigs


@pytest.mark.parametrize("flags", [("-f",), (), ("-3p",), ("-f", "-ps", "4",
	"-ns", "-5"), ("-f", "-sf", "0.5")])
def test_barcode_column_counts_each_cell(tmp_path, fragments, sizes,
		per_cell_bigwigs, flags):
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", *flags)
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == BARCODES
	assert [track["group"] for track in reader.tracks] == [0, 0, 0]
	assert "strand" not in reader.tracks[0]
	bigwigs = per_cell_bigwigs(flags)
	for barcode in BARCODES:
		assert_tracks_match_bigwigs(path, [barcode], [bigwigs[barcode]])


def test_barcode_column_read_depth_is_the_files(tmp_path, fragments, sizes):
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f")
	scaled = bam2fw(tmp_path / "r", [fragments], sizes, "--barcode-column", "4",
		"-f", "-r")
	counts, reader = FigWigReader(path), FigWigReader(scaled)
	depth = counts.stats["sum"].sum()
	for chrom, length in CHROM_SIZES:
		expected = counts.read(chrom, [0], length, dtype="float64") / depth
		numpy.testing.assert_allclose(reader.read(chrom, [0], length,
			dtype="float64"), expected, rtol=1e-6)


def test_cells_table_picks_names_and_attributes(tmp_path, fragments, sizes,
		per_cell_bigwigs, capsys):
	table = tmp_path / "cells.csv"
	table.write_text("barcode,name,cell_type,dtype\n"
		"CCCT-1,,B,\nAAAC-1,mono_1,CD14 Mono,uint16\nGGGG-1,,T,\n")
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f",
		"--cells", str(table), "-v")
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["CCCT-1", "mono_1",
		"GGGG-1"]
	assert [track["cell_type"] for track in reader.tracks] == ["B", "CD14 Mono",
		"T"]
	assert [track["dtype"] for track in reader.tracks] == ["uint8", "uint16",
		"uint8"]
	assert reader.stats["sum"][2] == 0
	bigwigs = per_cell_bigwigs(("-f",))
	assert_tracks_match_bigwigs(path, ["CCCT-1"], [bigwigs["CCCT-1"]])
	assert_tracks_match_bigwigs(path, ["mono_1"], [bigwigs["AAAC-1"]])
	out = capsys.readouterr().out
	assert "2 fragments of {} have barcodes not in --cells".format(fragments) in out
	assert "chrUN encountered in input but not in FASTA/chrom sizes." in out


def test_groups_and_several_files(tmp_path, fragments, fragments_gz, sizes):
	table = tmp_path / "files.tsv"
	table.write_text("file\tname\tsample\ncells.tsv\tA\tpbmc\ncells.tsv.gz\t\t"
		"pbmc2\n")
	path = bam2fw(tmp_path, [fragments, fragments_gz], sizes, "--barcode-column",
		"4", "-f", "--groups", "2", "-m", str(table))
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["A." + b for b in
		BARCODES] + ["cells." + b for b in BARCODES]
	assert [track["group"] for track in reader.tracks] == [0, 0, 1, 2, 2, 3]
	assert [track["sample"] for track in reader.tracks] == ["pbmc"] * 3 + [
		"pbmc2"] * 3
	y = reader.read("chr1", [0], 1000, dtype="float32")
	numpy.testing.assert_array_equal(y[0, :3], y[0, 3:])


def test_threads_and_units_give_the_same_file(tmp_path, fragments, fragments_gz,
		sizes, monkeypatch):
	one = bam2fw(tmp_path / "one", [fragments], sizes, "--barcode-column", "4",
		"-f", "-p", "1")
	# Units of 64 bytes of text, so that barcodes are found across units
	# and numbered afterwards.
	monkeypatch.setattr(figwig.bam2bw, "_IV_UNIT_BYTES", 64)
	units = bam2fw(tmp_path / "units", [fragments], sizes, "--barcode-column",
		"4", "-f", "-p", "3")
	# A plain-gzip file is read as a stream whatever -p is.
	stream = bam2fw(tmp_path / "gz", [fragments_gz], sizes, "--barcode-column",
		"4", "-f", "-p", "3")
	with open(one, "rb") as a, open(units, "rb") as b:
		assert a.read() == b.read()
	assert [t["name"] for t in FigWigReader(stream).tracks] == BARCODES
	reader, other = FigWigReader(one), FigWigReader(stream)
	numpy.testing.assert_array_equal(reader.stats["sum"], other.stats["sum"])


def test_bgzf_units(tmp_path, fragments, sizes):
	import pysam

	bgzf = str(tmp_path / "cells.bed.gz")
	pysam.tabix_compress(fragments, bgzf)
	one = bam2fw(tmp_path / "one", [fragments], sizes, "--barcode-column", "4",
		"-f")
	path = bam2fw(tmp_path / "bgzf", [bgzf], sizes, "--barcode-column", "4", "-f",
		"-p", "2")
	reader, other = FigWigReader(one), FigWigReader(path)
	numpy.testing.assert_array_equal(reader.stats["sum"], other.stats["sum"])
	assert [t["name"] for t in other.tracks] == BARCODES


def test_barcode_column_errors(tmp_path, fragments, bam, sizes):
	with pytest.raises(SystemExit, match="column number from 4"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "2")
	with pytest.raises(SystemExit, match="reads BED/tsv files of fragments"):
		bam2fw(tmp_path, [bam], sizes, "--barcode-column", "4")
	with pytest.raises(SystemExit, match="does not hold a chromosome, a start, "
			"an end and a barcode in column 6"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "6")
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\nAAAC-1\nAAAC-1\n")
	with pytest.raises(SystemExit, match="lists a barcode more than once"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "--cells",
			str(table))
	elsewhere = tmp_path / "elsewhere.tsv"
	elsewhere.write_text("chrZ\t1\t5\tAAAC-1\n")
	with pytest.raises(SystemExit, match="no fragment of .* is on a chromosome "
			"of .*; the file's chromosomes include 'chrZ'"):
		bam2fw(tmp_path, [str(elsewhere)], sizes, "--barcode-column", "4")
	table.write_text("barcode\nTTTT-1\n")
	path = bam2fw(tmp_path / "listed", [fragments], sizes, "--barcode-column",
		"4", "--cells", str(table))
	assert FigWigReader(path).stats["sum"].tolist() == [0]
	bad = tmp_path / "bad.tsv"
	bad.write_text("chr1\t100\tx\tAAAC-1\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		bam2fw(tmp_path, [str(bad)], sizes, "--barcode-column", "4")
	bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-ps", "700",
		"-f")


def test_barcode_column_discards_and_reports(tmp_path, fragments, sizes, capsys):
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f")
	out = capsys.readouterr().out
	assert "1 entries fell outside the provided chromosome sizes" in out
	assert "chr2: 1" in out
	reader = FigWigReader(path)
	assert reader.read("chr2", [480], 20, tracks="CCCT-1")[0, 0] == 1


def test_key_packing_with_many_chromosomes(tmp_path):
	"""The key holds the chromosome above the cell above the position; 3,000
	chromosomes take 12 bits of it, and the counts must still land on the
	right chromosome, cell and base."""

	chrom_sizes = [("c{}".format(k), 50_000) for k in range(3000)]
	sizes = str(write_chrom_sizes(tmp_path / "many.chrom.sizes", chrom_sizes))
	rng = numpy.random.default_rng(0)
	rows = []
	for _ in range(2000):
		chrom = "c{}".format(rng.integers(3000))
		start = int(rng.integers(49_000))
		rows.append((chrom, start, start + 100, "B{}".format(rng.integers(50))))
	path = write_intervals(tmp_path / "many.tsv", rows)
	out = bam2fw(tmp_path, [str(path)], sizes, "--barcode-column", "4", "-f")
	reader = FigWigReader(out)
	names = [track["name"] for track in reader.tracks]
	assert names == list(dict.fromkeys(barcode for *_, barcode in rows))
	expected = numpy.zeros((3000, 50, 50_000), dtype=numpy.int64)
	for chrom, start, end, barcode in rows:
		expected[int(chrom[1:]), names.index(barcode), [start, end - 1]] += 1
	for k in rng.choice(3000, 20, replace=False):
		y = reader.read("c{}".format(k), [0], 50_000, dtype="int64")[0]
		numpy.testing.assert_array_equal(y, expected[k])
	numpy.testing.assert_array_equal(reader.stats["sum"], expected.sum(axis=(0, 2)))


def test_too_many_cells_for_the_key(tmp_path, fragments, sizes, monkeypatch):
	"""With 3 chromosomes the key keeps 2 bits for them; 37 key bits leave 3
	for the cell, so 8 cells."""

	import figwig._cells

	monkeypatch.setattr(figwig._cells, "_KEY_BITS", 37)
	rows = [("chr1", 10 * k, 10 * k + 5, "B{}".format(k)) for k in range(9)]
	path = str(write_intervals(tmp_path / "nine.tsv", rows))
	with pytest.raises(SystemExit, match="holds more than 8 barcodes"):
		bam2fw(tmp_path, [path], sizes, "--barcode-column", "4")
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\n" + "".join("B{}\n".format(k) for k in range(9)))
	with pytest.raises(SystemExit, match="--cells lists 9 barcodes, more than "
			"the 8"):
		bam2fw(tmp_path, [path], sizes, "--barcode-column", "4", "--cells",
			str(table))
	out = bam2fw(tmp_path, [str(write_intervals(tmp_path / "eight.tsv",
		rows[:8]))], sizes, "--barcode-column", "4")
	assert len(FigWigReader(out).tracks) == 8


def test_malformed_fragments_files(tmp_path, sizes):
	import pysam

	def convert(name, text, *flags, binary=False):
		path = tmp_path / name
		if binary:
			path.write_bytes(text)
		else:
			path.write_text(text)
		return bam2fw(tmp_path / name.replace(".", "_"), [str(path)], sizes,
			"--barcode-column", "4", "-f", *flags)

	with pytest.raises(SystemExit, match="no fragment of .* was counted: the "
			"file holds none"):
		convert("empty.tsv", "")
	with pytest.raises(SystemExit, match="the file holds none"):
		convert("comments.tsv", "# id=x\n#\n\n")
	out = convert("crlf.tsv", "chr1\t100\t150\tAAAC-1\t1\r\nchr1\t200\t250\t"
		"AAAG-1\t2\r\n")
	assert FigWigReader(out).stats["sum"].tolist() == [2, 2]
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("utf8.tsv", b"chr1\t100\t150\tAAAC\xc3\xa9-1\t1\n", binary=True)
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("huge.tsv", "chr1\t1000000000000000000\t150\tAAAC-1\t1\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("float.tsv", "chr1\t100.0\t150\tAAAC-1\t1\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("short.tsv", "chr1\t100\t150\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("bare_cr.tsv", "chr1\t100\t150\tAAAC-1\t1\rchr1\t200\t250\t"
			"AAAC-1\t1\n")
	out = convert("negative.tsv", "chr1\t-5\t3\tAAAC-1\t1\nchr1\t5\t3\tAAAC-1\t1\n")
	assert FigWigReader(out).stats["sum"].tolist() == [3]

	# A gzip file cut inside a member, plain or BGZF, and a BGZF file cut at
	# a block boundary, which lacks the empty block that ends one.
	lines = "".join("chr1\t{}\t{}\tB{}\t1\n".format(k, k + 50, k % 7) for k in
		range(20_000))
	(tmp_path / "whole.tsv").write_text(lines)
	bgzf = str(tmp_path / "whole.tsv.gz")
	pysam.tabix_compress(str(tmp_path / "whole.tsv"), bgzf)
	with open(bgzf, "rb") as handle:
		data = handle.read()
	assert len(data) > 2 * 28
	# Without its end block, a BGZF file is refused before it is read; the
	# cut inside a member is what the gzip stream finds otherwise.
	with pytest.raises(SystemExit, match="ends without the empty block that "
			"ends a BGZF file, so it was cut short"):
		convert("half.tsv.gz", data[:len(data) // 2], binary=True)
	with pytest.raises(SystemExit, match="ends without the empty block"):
		convert("blocks.tsv.gz", data[:-28], "-p", "2", binary=True)
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("half_marked.tsv.gz", data[:len(data) // 2] + data[-28:],
			binary=True)
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("half_marked.tsv.gz", data[:len(data) // 2] + data[-28:], "-p",
			"2", binary=True)
	import gzip
	plain = gzip.compress(lines.encode())
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("plain.tsv.gz", plain[:len(plain) // 2], binary=True)
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("garbage.tsv.gz", b"\x1f\x8b\x08\x00" + bytes(100), binary=True)


def test_bgzf_blocks_as_units(tmp_path, sizes, monkeypatch):
	"""A BGZF file of several blocks, scanned a block per unit, with lines
	cut by block ends and barcodes found in every unit."""

	import pysam

	rng = numpy.random.default_rng(1)
	rows = [("chr1", int(k), int(k) + 10, "B{}".format(rng.integers(40))) for k
		in sorted(rng.integers(0, 990, 30_000))]
	path = write_intervals(tmp_path / "blocks.tsv", rows)
	bgzf = str(tmp_path / "blocks.tsv.gz")
	pysam.tabix_compress(str(path), bgzf)
	assert os.path.getsize(bgzf) > 3 * 65_536 // 4
	one = bam2fw(tmp_path / "one", [str(path)], sizes, "--barcode-column", "4",
		"-f")
	monkeypatch.setattr(figwig.bam2bw, "_IV_UNIT_BLOCKS", 1)
	units = bam2fw(tmp_path / "units", [bgzf], sizes, "--barcode-column", "4",
		"-f", "-p", "3")
	with open(one, "rb") as a, open(units, "rb") as b:
		assert a.read() == b.read()
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\n" + "".join("B{}\n".format(k) for k in
		range(0, 40, 2)))
	listed = bam2fw(tmp_path / "listed", [bgzf], sizes, "--barcode-column", "4",
		"-f", "-p", "3", "--cells", str(table))
	reader, other = FigWigReader(one), FigWigReader(listed)
	names = [track["name"] for track in reader.tracks]
	numpy.testing.assert_array_equal(other.stats["sum"], reader.stats["sum"][[
		names.index("B{}".format(k)) for k in range(0, 40, 2)]])


def test_barcode_column_other_options(tmp_path, fragments, sizes):
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\tname\nAAAC-1\tx\nAAAG-1\tx\n")
	with pytest.raises(SystemExit, match="x"):
		bam2fw(tmp_path / "dup", [fragments], sizes, "--barcode-column", "4",
			"--cells", str(table))
	table.write_text("barcode\nCCCT-1\nAAAC-1\n")
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f",
		"-u", "-r", "--cells", str(table), "-z", "2", "-t", "256", "-l", "0",
		"--dtype", "float32", "-g", "hg38")
	reader = FigWigReader(path)
	assert reader.tile == 256 and reader.genome == "hg38" and reader.zooms == (
		4, 16)
	assert [track["dtype"] for track in reader.tracks] == ["float32"] * 2
	# -r divides by the depth of the cells kept: CCCT-1 and AAAC-1 together.
	plain = FigWigReader(bam2fw(tmp_path / "plain", [fragments], sizes,
		"--barcode-column", "4", "-f", "--cells", str(table)))
	depth = plain.stats["sum"].sum()
	numpy.testing.assert_allclose(reader.stats["sum"], plain.stats["sum"] / depth,
		rtol=1e-6)
	with pytest.raises(SystemExit, match="--mate_pairs only supports BAM/SAM"):
		bam2fw(tmp_path / "mp", [fragments], sizes, "--barcode-column", "4", "-mp")


def test_barcode_column_command_line(tmp_path, fragments, sizes):
	name = str(tmp_path / "cli")
	result = subprocess.run([sys.executable, "-m", "figwig", "bam2fw", fragments,
		"-s", sizes, "-n", name, "-p", "2", "--barcode-column", "4", "-f", "-v"],
		env=ENV, capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert [track["name"] for track in FigWigReader(name + ".fw").tracks] == \
		BARCODES
	assert "chrUN encountered in input but not in FASTA/chrom sizes." in \
		result.stdout
