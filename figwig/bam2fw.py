# bam2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig bam2fw` command: reads or fragments into one figwig file of
per-base counts.

It takes what `figwig bam2bw` takes, SAM/BAM files of reads or BED/tsv
files of fragments and the same options for what to count, and reads them
the same way. Where `figwig bam2bw` pools every file into one pair of
bigWigs, this writes each file as its own pair of tracks, `<name>.+` and
`<name>.-`, or one track with `-u`, in one group, so that a figwig file
holds a whole set of experiments. A metadata table gives the tracks their
names and attributes. With `--barcode-column`, a fragments file is written
as one track per cell instead, named by its barcode, the cells listed by
`--cells` with their attributes or every barcode in the file.
"""

import argparse
import collections
import importlib.util
import os

import numpy

from ._tables import _exit
from ._tables import _match_rows
from ._tables import _read_table
from ._tables import _refuse_existing
from .figwig import _DEFAULT_TILE
from .figwig import FigWigWriter

_FRAGMENT_EXTENSIONS = ('.tsv', '.tsv.gz', '.bed', '.bed.gz')

# The values a chunk of the figwig file holds at most, tracks times tile,
# which FigWigWriter checks; a file's cells are grouped within it by default.
_CHUNK_VALUES = 1 << 29

_EXTENSIONS = ('.bam', '.sam', '.tsv.gz', '.tsv', '.bed.gz', '.bed')


def _parser(prog):
	from .bam2bw import _input_arguments

	parser = argparse.ArgumentParser(prog=prog, description="Convert SAM/BAM "
		"files of reads, or BED/tsv files of fragments, into one figwig file of "
		"per-base counts, with a track per strand of each file.")
	_input_arguments(parser)
	parser.add_argument('-p', '--parallel', default=1, type=int, help="The "
		"number of cores to use, or a negative number to count back from the "
		"number of CPUs: -1 is all of them.")
	parser.add_argument('-n', '--name', required=True, help="The figwig file "
		"to write, without its .fw extension.")
	parser.add_argument('-m', '--metadata', default=None, help="A tab- or "
		"comma-separated table with a header and one row per input file, "
		"matched on its first column by the path given here or its base name. "
		"A 'name' column names the file's tracks, and every other column is an "
		"attribute of them.")
	parser.add_argument('-g', '--genome', default=None, help="The reference "
		"genome, as a name or a URL, recorded in the file.")
	parser.add_argument('--barcode-column', default=None, type=int, help="The "
		"column of a BED/tsv file of fragments that holds the cell barcode, "
		"counted from 1 (4 in a Cell Ranger fragments file). Each file is then "
		"written as one track per cell, named by its barcode, in one group. "
		"Default is to pool a file's fragments into one track per strand.")
	parser.add_argument('--cells', default=None, help="With --barcode-column, "
		"a tab- or comma-separated table with a header whose first column is "
		"the barcode: only these cells are written, in this order, a 'name' "
		"column names each cell's track, a 'dtype' column sets its dtype, and "
		"every other column is an attribute of it. Default is every barcode in "
		"the file, in order of first appearance.")
	parser.add_argument('--groups', default=None, type=int, help="With "
		"--barcode-column, the consecutive cells whose tiles are stored "
		"together. Default is every cell of a file in one group, up to the "
		"2**29 values a chunk holds, 32,768 cells at the default tile.")
	parser.add_argument('-t', '--tile', default=_DEFAULT_TILE, type=int,
		help="The bases per tile, a power of two from 16 to 65536. Default is "
		"{}.".format(_DEFAULT_TILE))
	parser.add_argument('--dtype', default=None, help="The dtype of every "
		"track, such as uint16. Default is the narrowest type that holds each "
		"track's values.")
	parser.add_argument('-z', '--zooms', default=0, type=int, help="The number "
		"of zoom levels to keep beside the tile sums, each 4 times coarser than "
		"the last, for FigWigReader.tile_sums(level=k). Default is 0.")
	parser.add_argument('-l', '--level', default=1, type=int, help="The "
		"compression level, or 0 to store the chunks uncompressed. Default is "
		"1.")
	parser.add_argument('--force', action='store_true', help="Write over a "
		"figwig file at <-n>.fw, which is refused without it.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def _base_name(filename):
	"""A file's name without its directory and its extension."""

	name = os.path.basename(filename)
	for extension in _EXTENSIONS:
		if name.endswith(extension):
			return name[:-len(extension)]

	return name


def _tracks_for_files(filenames, metadata, unstranded, dtype=None):
	"""The track records of every file, as (tracks, groups): a track per
	strand of each file, or one with `unstranded`, and the files' rows of the
	metadata table as their attributes."""

	rows = [None] * len(filenames)
	if metadata is not None:
		header, records = _read_table(metadata)
		rows = _match_rows(metadata, records, header, filenames)

	tracks, groups = [], []
	for filename, row in zip(filenames, rows):
		name = _base_name(filename)
		attributes = {}
		if row is not None:
			first = list(row)[0]
			for column, value in row.items():
				if column == 'name' and len(value) > 0:
					name = value
				elif column not in ('name', 'dtype', first):
					attributes[column] = value

		if unstranded:
			names = [name]
			strands = [None]
		else:
			names = [name + '.+', name + '.-']
			strands = ['+', '-']

		group = []
		for track_name, strand in zip(names, strands):
			track = {'name': track_name, 'dtype': dtype}
			if strand is not None:
				track['strand'] = strand
			track.update(attributes)
			tracks.append(track)
			group.append(track_name)
		groups.append(group)

	return tracks, groups


def _trim_counts(reads, chrom_sizes, unstranded):
	"""Each file's counts as sorted arrays within their chromosomes, with
	the entries that fall outside a chromosome dropped and counted."""

	from .bam2bw import as_arrays

	trimmed, discarded = [], collections.defaultdict(int)
	for pos_reads, neg_reads in reads:
		strands = {}
		for key, counts in (('+', pos_reads), ('-', neg_reads)):
			if unstranded and key == '-':
				continue
			strands[key] = {}
			for chrom, size in chrom_sizes:
				positions, values = as_arrays(counts.pop(chrom))
				lo, hi = numpy.searchsorted(positions, [0, size])
				if lo > 0 or hi < len(positions):
					discarded[chrom] += int(values[:lo].sum()) + int(values[hi:]
						.sum())
				strands[key][chrom] = positions[lo:hi], values[lo:hi]
		trimmed.append(strands)

	return trimmed, discarded


def _write_counts(writer, names, strands, chrom_sizes, scale_factor, read_depth):
	"""Write one file's counts, strands['+'] and strands['-'] or strands['+']
	alone, as the tracks `names`, chromosome by chromosome, each count times
	`scale_factor` and divided by `read_depth` unless it is None. The
	counts are nonzero bases, so they go in as entries, which the writer
	encodes on its threads."""

	for chrom, _ in chrom_sizes:
		parts = [strands[key][chrom] for key in ('+', '-') if key in strands]
		positions = numpy.concatenate([p for p, _ in parts])
		if len(positions) == 0:
			continue
		tracks = numpy.concatenate([numpy.full(len(p), k, dtype=numpy.int64)
			for k, (p, _) in enumerate(parts)])
		values = numpy.concatenate([counts for _, counts in parts]).astype(
			numpy.float64)
		values *= scale_factor
		if read_depth is not None:
			values /= read_depth

		writer.write_sparse(chrom, positions, writer._select(names)[0][tracks],
			values)


def _check_cell_arguments(args):
	"""Raise for arguments --barcode-column does not take."""

	if args.barcode_column < 4:
		raise ValueError("--barcode-column is a column number from 4, since "
			"columns 1 to 3 hold the chromosome, start and end; a Cell Ranger "
			"fragments file's barcode is column 4.")
	for filename in args.filename:
		if not filename.endswith(_FRAGMENT_EXTENSIONS):
			raise ValueError("--barcode-column reads BED/tsv files of fragments, "
				"not {}.".format(filename))
	if args.groups is not None and args.groups < 1:
		raise ValueError("--groups must be at least 1.")


def _cell_table(path):
	"""The cells of a --cells table, as (barcodes, names, dtypes,
	attributes), each a list over the rows."""

	header, records = _read_table(path)
	barcodes = [record[header[0]] for record in records]
	if len(set(barcodes)) != len(barcodes):
		raise ValueError("{} lists a barcode more than once in its first column, "
			"{!r}.".format(path, header[0]))

	names, dtypes, attributes = [], [], []
	for record in records:
		names.append(record.get('name') or record[header[0]])
		dtypes.append(record.get('dtype'))
		attributes.append({column: value for column, value in record.items()
			if column not in (header[0], 'name', 'dtype')})

	return barcodes, names, dtypes, attributes


def _tracks_for_cells(filenames, metadata, barcodes_per_file, cells, dtype,
		groups, tile=_DEFAULT_TILE):
	"""The track records of every file's cells, as (tracks, groups): one
	track per barcode kept, named by the --cells table's name or the
	barcode, with the file's name in front when there are several files,
	carrying the cell's attributes and the file's. Without `groups`, a
	file's cells are one group, or as few as a chunk's values allow."""

	if groups is None:
		groups = max(_CHUNK_VALUES // tile, 1)

	rows = [None] * len(filenames)
	if metadata is not None:
		header, records = _read_table(metadata)
		rows = _match_rows(metadata, records, header, filenames)

	tracks, track_groups = [], []
	for filename, row, barcodes in zip(filenames, rows, barcodes_per_file):
		prefix = _base_name(filename)
		file_attributes = {}
		if row is not None:
			first = list(row)[0]
			for column, value in row.items():
				if column == 'name' and len(value) > 0:
					prefix = value
				elif column not in ('name', 'dtype', first):
					file_attributes[column] = value

		names = []
		for k, barcode in enumerate(barcodes):
			name, cell_dtype, attributes = barcode, None, {}
			if cells is not None:
				name, cell_dtype, attributes = cells[1][k], cells[2][k], cells[3][k]
			if len(filenames) > 1:
				name = prefix + '.' + name

			track = {'name': name, 'dtype': cell_dtype or dtype}
			track.update(file_attributes)
			track.update(attributes)
			tracks.append(track)
			names.append(name)

		for k in range(0, len(names), groups):
			track_groups.append(names[k:k + groups])

	return tracks, track_groups


def _cells_main(args, prog, cores, chrom_sizes):
	"""`figwig bam2fw --barcode-column`: each fragments file as one track per
	cell."""

	from ._cells import _read_cells

	try:
		_check_cell_arguments(args)
		cells = None
		if args.cells is not None:
			cells = _cell_table(args.cells)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)

	listed = None if cells is None else cells[0]
	field = args.barcode_column - 1
	files, barcodes_per_file, dropped = [], [], collections.defaultdict(int)
	for idx, filename in enumerate(args.filename):
		try:
			counts, barcodes, missing, file_dropped, skipped = _read_cells(filename,
				args, chrom_sizes, field, listed, cores, idx, args.tile)
			if len(barcodes) == 0 and len(missing) > 0:
				raise ValueError(("no fragment of {} is on a chromosome of {}; the "
					"file's chromosomes include {}.").format(filename, args.sizes,
					", ".join(repr(name) for name in missing[:3])))
		except (ValueError, TypeError, OSError) as e:
			raise _exit(prog, e)

		if args.verbose:
			for chrom in missing:
				print("{} encountered in input but not in FASTA/chrom sizes.".format(
					chrom))
			if skipped > 0:
				print("{} fragments of {} have barcodes not in --cells, and were "
					"left out.".format(skipped, filename))
		for chrom, count in file_dropped.items():
			dropped[chrom] += count
		files.append(counts)
		barcodes_per_file.append(barcodes)

	if args.verbose:
		print("\n" * len(args.filename))
	if len(dropped) > 0:
		print("{} entries fell outside the provided chromosome sizes and were "
			"discarded ({}).".format(sum(dropped.values()), ", ".join(
			"{}: {}".format(chrom, count) for chrom, count in dropped.items())))

	try:
		tracks, groups = _tracks_for_cells(args.filename, args.metadata,
			barcodes_per_file, cells, args.dtype, args.groups, args.tile)
		if len(tracks) == 0:
			raise ValueError(("no fragment of {} was counted: {}.").format(
				", ".join(args.filename), "none has a barcode in --cells" if cells
				is not None else "the file holds none"))

		with FigWigWriter(args.name + '.fw', chrom_sizes, tracks,
				genome=args.genome, tile=args.tile, groups=groups, level=args.level,
				n_jobs=cores, zooms=args.zooms) as writer:
			writer._copies = False
			offset = 0
			for counts, barcodes in zip(files, barcodes_per_file):
				# The entries come sorted by tile, cell and offset, which is
				# the writer's own order when the file's cells are one group.
				writer._sorted_entries = len(barcodes) <= (max(_CHUNK_VALUES //
					args.tile, 1) if args.groups is None else args.groups)
				read_depth = None
				if args.read_depth:
					read_depth = sum(int(values.sum()) for _, _, values in
						counts.values())
					if args.verbose:
						print("Dividing {} through by a read depth of {}.".format(
							tracks[offset]['name'] if len(barcodes) > 0 else '',
							read_depth))

				for chrom, _ in chrom_sizes:
					if chrom not in counts:
						continue
					positions, cell_ids, values = counts[chrom]
					values = values.astype(numpy.float64)
					values *= args.scale_factor
					if read_depth is not None:
						values /= read_depth
					writer.write_sparse(chrom, positions, cell_ids.astype(numpy.int64) + offset,
						values)
				offset += len(barcodes)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)


def main(argv=None, prog='figwig bam2fw'):
	from .bam2bw import _check_filenames
	from .bam2bw import _cores
	from .bam2bw import _REQUIRED
	from .bam2bw import _tqdm
	from .bam2bw import read_chrom_sizes
	from .bam2bw import read_files

	args = _parser(prog).parse_args(argv)

	missing = [name for name in _REQUIRED if importlib.util.find_spec(name) is
		None]
	if len(missing) > 0:
		raise SystemExit(("{} needs {}, from figwig's bam2bw extra: pip install "
			"'figwig[bam2bw]'").format(prog, ', '.join(missing)))

	try:
		if not args.force:
			_refuse_existing(args.name + '.fw', '--force')
		_check_filenames(args)
		chrom_sizes = read_chrom_sizes(args.sizes)
		cores = _cores(args.parallel)
		if args.barcode_column is None:
			tracks, groups = _tracks_for_files(args.filename, args.metadata,
				args.unstranded, args.dtype)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)

	if args.barcode_column is not None:
		return _cells_main(args, prog, cores, chrom_sizes)

	tqdm = _tqdm(args.verbose)
	if args.verbose:
		import multiprocessing
		tqdm.set_lock(multiprocessing.RLock())

	reads = read_files(args, chrom_sizes, cores)
	if args.verbose:
		print("\n" * len(args.filename))

	files, discarded = _trim_counts(reads, chrom_sizes, args.unstranded)
	del reads
	if len(discarded) > 0:
		print("{} entries fell outside the provided chromosome sizes and were "
			"discarded ({}).".format(sum(discarded.values()), ", ".join(
			"{}: {}".format(chrom, count) for chrom, count in discarded.items())))

	try:
		with FigWigWriter(args.name + '.fw', chrom_sizes, tracks,
				genome=args.genome, tile=args.tile, groups=groups, level=args.level,
				n_jobs=cores, zooms=args.zooms) as writer:
			# The arrays `_write_counts` makes are the writer's to keep.
			writer._copies = False
			for strands, group in zip(files, groups):
				read_depth = None
				if args.read_depth:
					read_depth = sum(int(values.sum()) for counts in strands.values()
						for _, values in counts.values())
					if args.verbose:
						print("Dividing {} through by a read depth of {}.".format(
							group[0], read_depth))

				_write_counts(writer, group, strands, chrom_sizes, args.scale_factor,
					read_depth)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)
