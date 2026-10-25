# _cells.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Per-cell counts from a fragments file, for `figwig bam2fw
--barcode-column`.

A fragments file, as Cell Ranger ATAC writes it, holds one fragment per
line: chromosome, start, end, the droplet barcode and a count. `figwig
bam2bw` reads it as a pooled track; this module counts each barcode's
fragment ends apart, so that a figwig file holds one track per cell. The
scanner is a numba kernel like bam2bw's `_scan_intervals`, over the same
BGZF units and the same chromosome-name table, with a second table of the
same shape for barcodes. Every recorded end becomes one int64 key,
`((chromosome << cell_bits) | cell) << 32 | position`, so that one sort of
the keys orders the entries as `FigWigWriter.write_sparse` takes them, a
chromosome at a time, and the runs of equal keys are the counts. The
position must lie inside its chromosome, which bounds it to 32 bits; an
end outside is counted as dropped, as `figwig bam2fw` reports it.

Barcodes are either listed up front (`--cells`), in which case every
thread shares one table and an unlisted barcode is skipped, or discovered
as they appear, in which case each unit of the file is scanned against a
copy of the table and the barcodes it finds are numbered in file order
afterwards, so that the tracks come out in order of first appearance
whatever the number of threads.
"""

import collections

import numpy

from .bam2bw import _FNV_OFFSET
from .bam2bw import _FNV_PRIME
from .bam2bw import _IV_BGZF_EXTRA
from .bam2bw import _IV_BGZF_HEADER
from .bam2bw import _NameTable
from .bam2bw import _NotBlockGzip
from .bam2bw import _bgzf_units
from .bam2bw import _inflate_unit
from .bam2bw import _jit
from .bam2bw import _text_units
from .bam2bw import _tqdm

_POSITION_BITS = 32
_POSITION_MASK = (1 << _POSITION_BITS) - 1
_STREAM_BLOCK = 1 << 24


def _scan_cells(buf, n, i, n_lines, keys, n_keys, table_hash, table_name,
	name_data, name_offsets, name_slots, cell_hash, cell_name, cell_data,
	cell_offsets, cell_slots, field, cell_bits, sizes, dropped, skipped, mode,
	pos_shift, neg_shift):
	"""Scan the lines of buf[i:n], which ends at a newline or at the end of
	the file, appending one key per recorded end of each fragment whose
	barcode, in column `field` (from 0), has a cell number in the barcode
	table. Compiled with numba by _cell_scanner.

	Returns (status, i, n_lines, n_keys, a, b). Status 0: every line was
	scanned. Status 1: the chromosome name buf[a:b] is not in the name
	table; status 4: the barcode buf[a:b] is not in the barcode table; status
	2: keys is full. For those three, i is the start of the line to resume
	from once the caller has added the name or grown keys. Status 3: the line
	buf[a:b] cannot be read.

	mode is 0 to record the start, 1 to record end - 1, and 2 to record both.
	An end that falls outside its chromosome is counted in dropped[slot]; a
	fragment of a barcode whose cell number is -1 is counted in skipped[0].
	"""

	mask = table_hash.shape[0] - 1
	cell_mask = cell_hash.shape[0] - 1
	capacity = keys.shape[0]
	prev_a = 0
	prev_b = -1
	prev_slot = -1
	cell_a = 0
	cell_b = -1
	cell_prev = -1

	while i < n:
		line_start = i
		if n_keys + 2 > capacity:
			return 2, line_start, n_lines, n_keys, 0, 0

		# Split the line on spaces and tabs, keeping the bounds of the first
		# three fields and of the barcode field. Every byte must be printable
		# ASCII, a space, a tab, or a carriage return that ends the line.
		n_fields = 0
		a0 = 0
		b0 = 0
		a1 = 0
		b1 = 0
		a2 = 0
		b2 = 0
		a3 = 0
		b3 = 0
		bad = False
		while True:
			while i < n:
				c = buf[i]
				if c == 9 or c == 32:
					i += 1
				elif c == 13:
					if i + 1 < n and buf[i + 1] == 10:
						i += 1
					else:
						bad = True
						break
				elif c == 10 or (c >= 33 and c <= 126):
					break
				else:
					bad = True
					break

			if bad or i == n or buf[i] == 10:
				break

			a = i
			while i < n and buf[i] >= 33 and buf[i] <= 126:
				i += 1

			if n_fields == 0:
				a0 = a
				b0 = i
			elif n_fields == 1:
				a1 = a
				b1 = i
			elif n_fields == 2:
				a2 = a
				b2 = i
			if n_fields == field:
				a3 = a
				b3 = i
			n_fields += 1

		if bad:
			while i < n and buf[i] != 10:
				i += 1
			return 3, line_start, n_lines, n_keys, line_start, i

		next_i = i + 1 if i < n else n

		# Blank lines and # comments are skipped.
		if n_fields == 0 or buf[a0] == 35:
			i = next_i
			n_lines += 1
			continue

		if n_fields < 3 or n_fields <= field:
			return 3, line_start, n_lines, n_keys, line_start, i

		# Look the chromosome up, reusing the previous line's answer when the
		# name is the same, which it nearly always is in a sorted file.
		length = b0 - a0
		same = length == prev_b - prev_a
		if same:
			for k in range(length):
				if buf[a0 + k] != buf[prev_a + k]:
					same = False
					break

		if same:
			slot = prev_slot
		else:
			h = _FNV_OFFSET
			for k in range(a0, b0):
				h = (h ^ buf[k]) * _FNV_PRIME

			j = h & mask
			found = -1
			while True:
				name = table_name[j]
				if name < 0:
					break

				if table_hash[j] == h:
					s = name_offsets[name]
					if name_offsets[name + 1] - s == length:
						match = True
						for k in range(length):
							if name_data[s + k] != buf[a0 + k]:
								match = False
								break

						if match:
							found = name
							break

				j = (j + 1) & mask

			if found < 0:
				return 1, line_start, n_lines, n_keys, a0, b0

			slot = name_slots[found]
			prev_a = a0
			prev_b = b0
			prev_slot = slot

		if slot < 0:
			i = next_i
			n_lines += 1
			continue

		# The barcode, the same way; consecutive lines often share it too.
		length = b3 - a3
		same = length == cell_b - cell_a
		if same:
			for k in range(length):
				if buf[a3 + k] != buf[cell_a + k]:
					same = False
					break

		if same:
			cell = cell_prev
		else:
			h = _FNV_OFFSET
			for k in range(a3, b3):
				h = (h ^ buf[k]) * _FNV_PRIME

			j = h & cell_mask
			found = -1
			while True:
				name = cell_name[j]
				if name < 0:
					break

				if cell_hash[j] == h:
					s = cell_offsets[name]
					if cell_offsets[name + 1] - s == length:
						match = True
						for k in range(length):
							if cell_data[s + k] != buf[a3 + k]:
								match = False
								break

						if match:
							found = name
							break

				j = (j + 1) & cell_mask

			if found < 0:
				return 4, line_start, n_lines, n_keys, a3, b3

			cell = cell_slots[found]
			cell_a = a3
			cell_b = b3
			cell_prev = cell

		if cell < 0:
			skipped[0] += 1
			i = next_i
			n_lines += 1
			continue

		k = a1
		negative = False
		if buf[k] == 45:
			negative = True
			k += 1
		elif buf[k] == 43:
			k += 1

		if k == b1 or b1 - k > 15:
			return 3, line_start, n_lines, n_keys, line_start, i

		start = 0
		while k < b1:
			c = buf[k]
			if c < 48 or c > 57:
				return 3, line_start, n_lines, n_keys, line_start, i

			start = start * 10 + (c - 48)
			k += 1

		if negative:
			start = -start

		k = a2
		negative = False
		if buf[k] == 45:
			negative = True
			k += 1
		elif buf[k] == 43:
			k += 1

		if k == b2 or b2 - k > 15:
			return 3, line_start, n_lines, n_keys, line_start, i

		end = 0
		while k < b2:
			c = buf[k]
			if c < 48 or c > 57:
				return 3, line_start, n_lines, n_keys, line_start, i

			end = end * 10 + (c - 48)
			k += 1

		if negative:
			end = -end

		start += pos_shift
		end += neg_shift
		size = sizes[slot]
		high = ((slot << cell_bits) | cell) << _POSITION_BITS

		if mode != 1:
			if start < 0 or start >= size:
				dropped[slot] += 1
			else:
				keys[n_keys] = high | start
				n_keys += 1

		if mode != 0:
			if end - 1 < 0 or end - 1 >= size:
				dropped[slot] += 1
			else:
				keys[n_keys] = high | (end - 1)
				n_keys += 1

		i = next_i
		n_lines += 1

	return 0, i, n_lines, n_keys, 0, 0


_cell_scanner_cache = []

def _cell_scanner():
	"""_scan_cells compiled with numba."""

	if not _cell_scanner_cache:
		_cell_scanner_cache.append(_jit(_scan_cells))

	return _cell_scanner_cache[0]


class _Barcodes(_NameTable):
	"""A name table of barcodes, each with its cell number or -1 for a
	barcode that is not kept. `n_kept` is the next cell number. A copy made
	for a scanning thread keeps the arrays and hashes but not the dict of
	names, which only the base table needs."""

	def __init__(self, n):
		super().__init__(n)
		self.n_kept = 0

	def add(self, data, slot):
		count = len(self.hashes)
		if self.names is None:
			self.names = {}
			super().add(data, slot)
			self.names = None
		else:
			super().add(data, slot)
		if slot >= 0:
			self.n_kept += 1
		return count

	def copy(self):
		other = _Barcodes.__new__(_Barcodes)
		for attr in ('table_hash', 'table_name', 'name_data', 'name_offsets',
			'name_slots'):
			setattr(other, attr, getattr(self, attr).copy())

		other.names = None
		other.hashes = list(self.hashes)
		other.n_kept = self.n_kept
		return other


class _CellScan:
	"""The state one scan of a fragments file shares between its units: the
	chromosome table, the barcode table, the sizes, and the counters."""

	def __init__(self, chroms, sizes, barcodes, field, cell_bits, mode, args,
			filename):
		self.table = _NameTable(len(chroms))
		for slot, chrom in enumerate(chroms):
			data = chrom.encode('utf-8')
			if len(data) > 0 and min(data) >= 33 and max(data) <= 126:
				self.table.add(data, slot)

		self.keep_all = barcodes is None
		self.cells = _Barcodes(1024 if barcodes is None else len(barcodes))
		for barcode in barcodes or ():
			self.cells.add(barcode.encode('utf-8'), self.cells.n_kept)

		self.sizes = numpy.asarray(sizes, dtype=numpy.int64)
		self.field = field
		self.cell_bits = cell_bits
		self.mode = mode
		self.args = args
		self.filename = filename
		self.dropped = numpy.zeros(len(chroms), dtype=numpy.int64)
		self.skipped = numpy.zeros(1, dtype=numpy.int64)
		self.missing = []
		self.scan = _cell_scanner()

	def scan_lines(self, buf, i, end, keys, n_keys, table, cells, missing,
			new_cells, dropped, skipped):
		"""Run the scanner over buf[i:end] with the given tables and
		counters, adding each chromosome name it does not know to `table`
		(as not in the sizes) and `missing`, each barcode it does not know
		to `cells` (numbered, and listed in `new_cells`, when every barcode
		is kept; as not kept otherwise), and growing `keys` when it fills.
		Returns (keys, n_keys, n_lines)."""

		n_lines = 0
		args = self.args
		while True:
			status, i, n_lines, n_keys, a, b = self.scan(buf, end, i, n_lines,
				keys, n_keys, table.table_hash, table.table_name, table.name_data,
				table.name_offsets, table.name_slots, cells.table_hash,
				cells.table_name, cells.name_data, cells.name_offsets,
				cells.name_slots, self.field, self.cell_bits, self.sizes, dropped,
				skipped, self.mode, args.pos_shift, args.neg_shift)

			if status == 0:
				return keys, n_keys, n_lines
			elif status == 1:
				data = bytes(buf[a:b])
				table.add(data, -1)
				missing.append(data.decode('ascii'))
			elif status == 2:
				grown = numpy.empty(2 * len(keys), dtype=numpy.int64)
				grown[:n_keys] = keys[:n_keys]
				keys = grown
			elif status == 4:
				data = bytes(buf[a:b])
				if self.keep_all:
					if cells.n_kept >= 1 << self.cell_bits:
						raise ValueError(("{} holds more than {:,} barcodes, which is "
							"the most a file with {:,} chromosomes can keep apart; "
							"give --cells the barcodes to keep.").format(self.filename,
							1 << self.cell_bits, len(self.sizes)))
					cells.add(data, cells.n_kept)
					new_cells.append(data)
				else:
					cells.add(data, -1)
			else:
				line = bytes(buf[a:b]).decode('ascii', errors='replace')
				raise ValueError(("{}: the line {!r} does not hold a chromosome, "
					"a start, an end and a barcode in column {}.").format(
					self.filename, line[:120], self.field + 1))


def _renumber(keys, first_new, new_cells, base, cell_bits):
	"""Give the keys of a unit scanned against a copy of `base` the cell
	numbers `base` assigns, adding the unit's new barcodes to it in order."""

	if len(new_cells) == 0:
		return keys

	remap = numpy.arange(first_new + len(new_cells), dtype=numpy.int64)
	for j, data in enumerate(new_cells):
		count = base.names.get(data)
		if count is None:
			count = base.add(data, base.n_kept)
		remap[first_new + j] = base.name_slots[count]

	cell = (keys >> _POSITION_BITS) & ((1 << cell_bits) - 1)
	keys += (remap[cell] - cell) << _POSITION_BITS
	return keys


def _scan_file_units(state, compressed, bar, threads):
	"""Scan a BGZF or uncompressed file in units on `threads` threads, as
	bam2bw's _scan_units does, returning the keys. Raises _NotBlockGzip
	when a compressed file is not a series of BGZF blocks."""

	from concurrent.futures import ThreadPoolExecutor

	import threading

	local = threading.local()

	def work(unit, first, cells):
		text = _inflate_unit(*unit) if compressed else unit

		start = 0
		if not first:
			start = text.find(b'\n') + 1
			if start == 0:
				return False, bytes(text), b'', None, 0, None, None, None

		end = text.rfind(b'\n') + 1

		table = getattr(local, 'table', None)
		if table is None:
			table = local.table = state.table.copy()
		if cells is None:
			cells = getattr(local, 'cells', None)
			if cells is None:
				cells = local.cells = state.cells.copy()

		names, new_cells = [], []
		first_new = cells.n_kept
		dropped = numpy.zeros(len(state.dropped), dtype=numpy.int64)
		skipped = numpy.zeros(1, dtype=numpy.int64)
		keys = numpy.empty((end - start) // 12 + 16, dtype=numpy.int64)
		keys, n_keys, n_lines = state.scan_lines(numpy.frombuffer(text,
			dtype=numpy.uint8), start, end, keys, 0, table, cells, names,
			new_cells, dropped, skipped)

		return (True, bytes(text[:start]), bytes(text[end:]), keys[:n_keys],
			n_lines, names, (first_new, new_cells), (dropped, skipped))

	parts = []
	carry = bytearray()
	carry_table = state.table.copy()

	def scan_carry():
		keys = numpy.empty(16, dtype=numpy.int64)
		keys, n_keys, n_lines = state.scan_lines(numpy.frombuffer(carry,
			dtype=numpy.uint8), 0, len(carry), keys, 0, carry_table, state.cells,
			state.missing, [], state.dropped, state.skipped)
		parts.append(keys[:n_keys])
		bar.update(n_lines)

	def collect(result):
		nonlocal carry

		split, head, tail, keys, n_lines, names, found, counters = result
		carry += head
		if split:
			if len(carry) > 0:
				scan_carry()

			carry = bytearray(tail)
			if state.keep_all:
				keys = _renumber(keys, found[0], found[1], state.cells,
					state.cell_bits)
			parts.append(keys)
			state.missing.extend(names)
			state.dropped += counters[0]
			state.skipped += counters[1]
			bar.update(n_lines)

	pool = ThreadPoolExecutor(threads)
	try:
		with open(state.filename, "rb", buffering=0) as handle:
			units = _bgzf_units(handle) if compressed else _text_units(handle)

			pending = collections.deque()
			for k, unit in enumerate(units):
				# When barcodes are discovered as they come, each unit scans
				# a copy of the table as it stands when the unit is handed
				# out, taken here on the thread that also adds to it.
				cells = state.cells.copy() if state.keep_all else None
				pending.append(pool.submit(work, unit, k == 0, cells))
				if len(pending) > 2 * threads:
					collect(pending.popleft().result())

			while pending:
				collect(pending.popleft().result())

		if len(carry) > 0:
			scan_carry()
	finally:
		pool.shutdown(wait=True, cancel_futures=True)

	state.missing = list(dict.fromkeys(state.missing))
	if parts:
		return numpy.concatenate(parts)
	return numpy.empty(0, dtype=numpy.int64)


def _scan_file_stream(state, compressed, bar):
	"""Scan a file as a stream on the calling thread, which any gzip file
	allows, returning the keys."""

	if compressed:
		from isal import igzip
		handle = igzip.open(state.filename, "rb")
	else:
		handle = open(state.filename, "rb")

	keys = numpy.empty(1 << 22, dtype=numpy.int64)
	n_keys = 0
	with handle:
		pending = bytearray()
		while True:
			block = handle.read(_STREAM_BLOCK)
			pending += block

			if len(block) > 0:
				end = pending.rfind(b'\n') + 1
				if end == 0:
					continue
			else:
				end = len(pending)

			buf = numpy.frombuffer(pending, dtype=numpy.uint8)
			keys, n_keys, n_lines = state.scan_lines(buf, 0, end, keys, n_keys,
				state.table, state.cells, state.missing, [], state.dropped,
				state.skipped)

			bar.update(n_lines)
			del buf
			del pending[:end]

			if len(block) == 0:
				break

	return keys[:n_keys]


def _is_bgzf(filename):
	with open(filename, 'rb') as handle:
		header = handle.read(16)
	return header[:4] == _IV_BGZF_HEADER and header[10:16] == _IV_BGZF_EXTRA


def _read_cells(filename, args, chrom_sizes, field, barcodes, threads, idx=0):
	"""Count each barcode's fragment ends in a fragments file.

	Parameters
	----------
	filename: str
		A BED/tsv file, BGZF, gzip or uncompressed.

	args: argparse.Namespace
		`figwig bam2fw`'s arguments: fragments, three_prime, pos_shift,
		neg_shift and verbose decide what is recorded.

	chrom_sizes: list of (str, int)
		The chromosomes to keep, in order.

	field: int
		The column that holds the barcode, from 0.

	barcodes: list of str or None
		The barcodes to keep, in the order their tracks take, or None to
		keep every barcode in order of first appearance.

	threads: int
		Threads to inflate and scan on; a gzip file that is not BGZF is
		read on one.

	idx: int
		The input's index, for the progress bar's row.

	Returns
	-------
	counts: dict
		Chromosome name -> (positions, cells, counts), the sorted distinct
		(position, cell) pairs with fragment ends and how many each has.

	barcodes: list of str
		The barcodes kept, cell number i being barcodes[i].

	missing: list of str
		Chromosome names not in `chrom_sizes`, in order of first appearance.

	dropped: dict
		Chromosome name -> ends that fell outside it after the shifts.

	skipped: int
		Fragments of barcodes not in `barcodes`.
	"""

	chroms = [chrom for chrom, _ in chrom_sizes]
	sizes = [size for _, size in chrom_sizes]
	if len(chroms) >= 1 << 22:
		raise ValueError("--barcode-column takes at most {:,} chromosomes.".format(
			1 << 22))
	if max(abs(args.pos_shift), abs(args.neg_shift)) >= 1 << 38:
		raise ValueError("--barcode-column takes shifts below 2**38.")
	if max(sizes, default=0) > _POSITION_MASK:
		raise ValueError("--barcode-column takes chromosomes of at most {:,} "
			"bases.".format(_POSITION_MASK))

	slot_bits = max(1, int(max(len(chroms), 1) - 1).bit_length())
	cell_bits = 63 - _POSITION_BITS - slot_bits
	if barcodes is not None and len(barcodes) > 1 << cell_bits:
		raise ValueError(("--cells lists {:,} barcodes, more than the {:,} a file "
			"with {:,} chromosomes can keep apart.").format(len(barcodes),
			1 << cell_bits, len(chroms)))

	mode = 2 if args.fragments else (1 if args.three_prime else 0)
	state = _CellScan(chroms, sizes, barcodes, field, cell_bits, mode, args,
		filename)
	compressed = not filename.endswith(('.tsv', '.bed'))

	bar = _tqdm(args.verbose)(disable=not args.verbose, position=idx,
		desc=filename.split('/')[-1])
	try:
		keys = None
		if threads > 1 and (not compressed or _is_bgzf(filename)):
			try:
				keys = _scan_file_units(state, compressed, bar, threads)
			except _NotBlockGzip:
				bar.reset()
		if keys is None:
			keys = _scan_file_stream(state, compressed, bar)
	finally:
		bar.close()

	keys.sort()
	counts = {}
	n_keys = len(keys)
	if n_keys > 0:
		first = numpy.empty(n_keys, dtype=bool)
		first[0] = True
		numpy.not_equal(keys[1:], keys[:-1], out=first[1:])
		starts = numpy.flatnonzero(first)
		del first

		distinct = keys[starts]
		del keys
		run_lengths = numpy.diff(starts, append=n_keys)
		del starts

		positions = distinct & _POSITION_MASK
		cells = (distinct >> _POSITION_BITS) & ((1 << cell_bits) - 1)
		slots = distinct >> (_POSITION_BITS + cell_bits)
		del distinct
		bounds = numpy.searchsorted(slots, numpy.arange(len(chroms) + 1))

		for slot in numpy.flatnonzero(bounds[1:] > bounds[:-1]):
			lo, hi = int(bounds[slot]), int(bounds[slot + 1])
			counts[chroms[slot]] = (positions[lo:hi], cells[lo:hi],
				run_lengths[lo:hi])

	if barcodes is None:
		table = state.cells
		barcodes = [bytes(table.name_data[table.name_offsets[k]:table
			.name_offsets[k + 1]]).decode('utf-8') for k in range(len(
			table.hashes))]

	dropped = {chroms[slot]: int(state.dropped[slot]) for slot in
		numpy.flatnonzero(state.dropped)}
	return counts, list(barcodes), state.missing, dropped, int(state.skipped[0])
