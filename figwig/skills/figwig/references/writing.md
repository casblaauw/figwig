# Writing bigWigs

`write_bigwig(paths, chrom_sizes, chroms, starts, values, ends=None, missing=0.0, zooms=10, level=6, engine='auto', n_jobs=8)`
writes everything in one call; `BigWigWriter(path, chrom_sizes, zooms=10, level=6, engine='auto', n_jobs=8)`
with `write(chroms, starts, values, ends=None, missing=0.0)` writes a batch
at a time and finishes the file at `close()` (`with` calls it).

| Items | `ends` | `values`, one path | `values`, a list of k paths |
|---|---|---|---|
| windows | `None` | `(n, width)` | `(n, k, width)`, channel `i` to `paths[i]` |
| intervals | `(n,)` | `(n,)` | `(n, k)` |
| single bases (per-base counts) | `None` | `(n, 1)`, windows of width 1 | `(n, k, 1)` |

A list of one path still needs the channel axis. `chrom_sizes` is a dict or
a list of pairs; `BigWigReader(path).chrom_sizes` copies a file's.

```python
import numpy
from figwig import read_bigwig
from figwig import write_bigwig

chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
chroms = numpy.array(["chr2", "chr1", "chr1"])
starts = numpy.array([300_000, 2_500_000, 1_000_000])
y_hat = numpy.random.default_rng(0).random((3, 2, 1000), dtype=numpy.float32)

write_bigwig(["plus.bw", "minus.bw"], chrom_sizes, chroms, starts, y_hat)
y = read_bigwig(["plus.bw", "minus.bw"], chroms, starts, width=1000)
print(numpy.array_equal(y, y_hat))
# True
```

## Rules

Each raises `ValueError`; a call that raises writes nothing and the writer
stays usable.

- Every chromosome written is in `chrom_sizes`.
- Within one call items may come in any order but must not overlap. A
  window's full width counts, even where its values are `missing` or NaN:
  overlapping predictions must be averaged first and written as single
  bases or intervals.
- Each call's items come after the last call's: a later chromosome in
  `chrom_sizes`, or the same one at or after the last end. An ENCODE file's
  `chrom_sizes` is in name order, so writing chr1, chr2, … one call each
  raises at chr10. Loop over `writer.chrom_sizes`, or build `chrom_sizes`
  in writing order.
- A window may run past its chromosome's end only where its values are
  NaN: `y[..., length - start:] = numpy.nan` before writing the last window.
- Values are written as float32 and must be finite or NaN.

## What is left out

Bases and intervals whose value is `missing` (0.0 by default) or NaN are not
written, and -0.0 counts as 0.0, so reading with the same `missing` gives
the values back. To store zeros as data, write and read with
`missing=numpy.nan`.

## A batch at a time

```python
import numpy
from figwig import BigWigReader
from figwig import BigWigWriter

chrom_sizes = {"chr1": 100_000, "chr2": 50_000}
rng = numpy.random.default_rng(0)
with BigWigWriter("track.bw", chrom_sizes, zooms=0) as writer:
	for chrom, length in writer.chrom_sizes.items():
		for start in range(0, length, 10_000):
			batch = numpy.arange(start, min(start + 10_000, length), 1000)
			y = rng.random((len(batch), 1000), dtype=numpy.float32)
			writer.write(chrom, batch, y)

print(BigWigReader("track.bw").read("chr2", [49_000], width=1000).shape)
# (1, 1000)
```

- The file is not a bigWig until `close`: a read before it, or after an
  exception inside the `with` block, says "is not a bigWig file". Write it
  again.
- `write_bigwig` with several paths writes them one after another; if one
  fails it is left without a header and the later ones are not written.
- Working memory does not grow with the size of a call.

## Options

| Option | Default | When to change it |
|---|---|---|
| `zooms` | 10 | `0` unless a genome browser will draw the file: building zoom levels is most of a write's time, and on sparse count tracks they can multiply the file size. |
| `engine` | `'auto'`: libdeflate if the `deflate` package loads, else zlib | `'zlib'` for bytes identical to pyBigWig's (with `level=6`, `zooms=0`, `missing=numpy.nan`, intervals or single bases). `'isal'` is fastest at levels 1 to 3 but its bytes differ run to run at 1 and 2. |
| `level` | 6 | 1 to write faster for a slightly larger file. |
| `n_jobs` | 8 | threads compressing blocks beside the calling thread; `references/threads-and-memory.md`. |

The bytes do not depend on `n_jobs` or on how values are split into calls
(except `'isal'`). Section types (fixedStep, varStep, bedGraph) are chosen
per window by size; there is nothing to set. Single bases come out as
varStep, which is what `figwig bam2bw` writes (`references/bam2bw.md`).
