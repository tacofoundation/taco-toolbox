from importlib.resources import as_file, files

import polars as pl


def _chunks(seq, size: int):
    """Yield consecutive chunks from sequence of specified size."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_admin_layer(
    level: int | str, admin_layers_pkg: str = "tacotoolbox"
) -> pl.DataFrame:
    """Load admin lookup table (code->name) for given level (0=countries, 1=states, 2=districts)."""
    lvl = str(level)
    base = files(admin_layers_pkg).joinpath("tortilla/data/admin/")
    traversable = base / f"admin{lvl}.parquet"

    with as_file(traversable) as path:
        return pl.read_parquet(path)


def morton_key(lon: float, lat: float, bits: int = 24) -> int:
    """Generate Morton (Z-order) key for spatial coordinates to improve EE cache locality."""
    x = (lon + 180.0) / 360.0
    y = (lat + 90.0) / 180.0
    maxv = (1 << bits) - 1
    xi = max(0, min(maxv, int(x * maxv)))
    yi = max(0, min(maxv, int(y * maxv)))

    def _part1by1(v: int) -> int:
        """Interleave bits by inserting zeros between them."""
        v &= 0x00000000FFFFFFFF
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v

    return (_part1by1(xi) << 1) | _part1by1(yi)
