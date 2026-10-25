# Recipes: whole jobs, step by step

Each step names the file with its rules. Commands run with `-p 8` where the
machine allows (`references/threads-and-memory.md`).

## A training set of many tracks, read every epoch

1. Make the tracks. From BAMs: `figwig bam2fw -s sizes -n data -m files.tsv -p 8 *.bam`
   (a stranded pair per file, `-ps 4 -ns -5` for ATAC-seq). From bigWigs:
   `figwig bw2fw -o data.fw -m tracks.tsv -g hg38 -p 8 *.bw`, with `-t 1024
   --groups 1` if they are dense float signal, and `--missing nan` if a
   measured 0 matters (`references/converters.md`).
2. Put the labels in the metadata table (assay, biosample, strand, fold,
   whatever the model selects by); `fw.select(assay="ATAC-seq")` returns
   the indices to pass as `tracks=`.
3. Read: `fw.read(chroms, starts, width, tracks=..., dtype="float32")` for
   dense targets, `bins=width` for per-window counts, `labels=` to sum
   replicates or cells into one channel, `device="cuda"` to land on the
   GPU (`references/figwig-reading.md`). If every window fits in memory,
   read them all once and index the array; otherwise a Dataset whose item
   is a batch, `n_jobs=1` in workers (`references/reading.md`).
4. Normalize with `fw.stats["sum"]` or `tile_sums` rather than a pass over
   the file.

## Genome-wide predictions to bigWigs

1. `chrom_sizes = BigWigReader(input).chrom_sizes` (name order) or a dict
   in the order you will write.
2. `write_bigwig` makes one file per call, so open one `BigWigWriter` per
   output channel once (`zooms=10` only if a browser will draw them) and
   loop over `writer.chrom_sizes`, tiling each chromosome with
   non-overlapping windows; average overlapping predictions first, and set
   the values past a chromosome's end to NaN.
3. `writer.write(chrom, starts, y_hat[:, i])` per channel, batches of
   thousands of windows; the file is finished at `close` (`references/writing.md`).
4. For many prediction tracks read in windows later, write a figwig file
   instead: `FigWigWriter(path, chrom_sizes, tracks)` with `write(chroms,
   starts, y_hat)`, all channels at once (`references/figwig-writing.md`).

## A single-cell file from a fragments file

1. Count cut sites per (chromosome, position, cell): both fragment ends,
   then sum duplicates (`numpy.unique` on the composite key, `bincount`).
2. One track per cell with `dtype="uint8"` and the cell type as an
   attribute; `FigWigWriter(path, chrom_sizes, tracks, groups=len(tracks),
   level=0)`; `write_sparse` a chromosome at a time.
3. Pseudobulks: `labels=` from the cell-type attribute. Per-cell windows:
   `sparse=True`, never dense. Peaks × cells: `sums`. Per-cell genome-wide
   slices: `figwig reorder --order tracks` and `read_entries`.
4. Or skip step 1 and 2: `figwig bam2fw -f -u` gives one track per fragments
   file, not per cell.

## Summaries over peaks, genes or cCREs

`fw.sums(chroms, starts, ends, tracks=...)` gives `(regions, tracks)` totals
in one call, whatever the lengths. Means over the region are
`totals / (ends - starts)[:, None]`; means over covered bases, as pyBigWig
`stats` and deepTools give them, need a file written with `--missing nan`
and `sums(nan="skip") / sums(nan="count")`. A genome-wide profile at
coarse resolution is `tile_sums` (reshape for larger bins), or `tile_sums(level=k)`
when the file was written with `-z`; finer than a tile is `read(bins=)`
over the chromosome (`references/figwig-reading.md`).

## Signal tracks through a figwig file and back, losslessly

`figwig bw2fw --missing nan` then `figwig fw2bw --missing nan` (or no flag:
`fw2bw` follows the file's recorded convention). Without `--missing nan`
the explicit 0.0 intervals of fold-change and p-value tracks are dropped
and uncovered bases read as 0. Check with `read_bigwig(..., keep_nan=True)`
on both sides.

## Reads as tracks (Fiber-seq, per-read channels)

1. Sort reads by start. One track per read and channel, named `r17.m6a`,
   with attributes `read`, `channel`, `strand` and the span.
2. `groups=` lists of about 1,000 consecutive reads per channel, position
   order; write each read's calls with `write_sparse` or `write`.
3. `read(..., channels="channel", items="read", tracks=fw.select(...))`
   gives `(windows, reads, channels, width)`; `sums(..., paired=True)` each
   read over its own span.

## Count tracks from many BAMs as bigWigs

One `figwig bam2bw` per sample, several at once at a low `-p`, with
`OPENBLAS_NUM_THREADS=1`; `-z 0` unless browsed; compare outputs by decoded
values, not bytes (`references/bam2bw.md`).
