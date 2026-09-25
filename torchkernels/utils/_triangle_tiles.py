"""Paired symmetric blocks on the first GPU and distributed rectangles."""

from fractions import Fraction


def _diagonal_blocks(size, block_size):
    # Equal-size adjacent pairs, plus one scalar for odd n. No padding.
    pairs = (size // 2 + block_size - 1) // block_size
    blocks = []
    start = 0
    if pairs:
        width, extra = divmod(size // 2, pairs)
        for index in range(pairs):
            count = width + (index < extra)
            blocks.extend([(start, start + count), (start + count, start + 2 * count)])
            start += 2 * count
    if size % 2:
        blocks.append((start, start + 1))
    return blocks


def _weighted_quotas(total, capacities, weights, initial):
    """Water-fill additional element counts under hard memory caps.

    Minimize the largest (initial[i] + quota[i]) / weight[i], subject to
    quota[i] <= capacity[i]. Initial loads account for mandatory diagonal work.
    """
    if sum(capacities) < total:
        raise MemoryError("Insufficient GPU capacity for rectangular tiles")
    if not total:
        return [0] * len(capacities)
    # Exact arithmetic keeps memory caps and rounding reliable even when
    # ratings have a large dynamic range. Each device starts filling at
    # initial/weight and stops filling at (initial+capacity)/weight.
    rates = [Fraction(weight) for weight in weights]
    bases = [Fraction(base) for base in initial]
    events = []
    for capacity, rate, base in zip(capacities, rates, bases):
        if capacity and rate:
            events.extend([(base / rate, rate), ((base + capacity) / rate, -rate)])
    level = Fraction(0)
    placed = Fraction(0)
    slope = Fraction(0)
    for boundary, change in sorted(events):
        next_placed = placed + slope * (boundary - level)
        if slope and next_placed >= total:
            level += (total - placed) / slope
            break
        placed = next_placed
        level = boundary
        slope += change
    ideal = [min(capacity, max(0, level * rate - base))
             for capacity, rate, base in zip(capacities, rates, bases)]
    quotas = [int(value) for value in ideal]
    left = total - sum(quotas)
    order = sorted(range(len(quotas)), key=lambda i: ideal[i] - quotas[i], reverse=True)
    for index in order[:left]:
        quotas[index] += 1
    return quotas


def _split_rectangle(tile):
    row, stop, column, end = tile
    if stop - row >= end - column:
        middle = (row + stop) // 2
        return [(row, middle, column, end), (middle, stop, column, end)]
    middle = (column + end) // 2
    return [(row, stop, column, middle), (row, stop, middle, end)]


def plan_tiles(size, block_size, capacities, load_balance="memory", tflops=None):
    """Plan paired diagonals on device 0, then balance all rectangles.

    Capacity is measured in matrix elements after reserving runtime buffers.
    Shrink diagonal blocks to fit the first GPU and its fair work/storage share.
    """
    required = size * (size + 1) // 2
    if sum(capacities) < required:
        raise MemoryError(f"Packed matrix needs {required} elements; GPUs can hold "
                          f"{sum(capacities)} after reserving buffers and headroom")
    if capacities[0] < size:
        raise MemoryError("First selected GPU cannot hold even the principal diagonal")
    weights = list(capacities) if load_balance == "memory" else list(tflops)
    # Estimate a fair first-GPU budget using the smallest possible diagonal
    # blocks. Choose the largest paired blocks that stay within that budget,
    # so e.g. three equal GPUs are not forced into a half-matrix diagonal load.
    minimum_remaining = list(capacities)
    minimum_remaining[0] -= size
    minimum_initial = [0.0] * len(capacities)
    minimum_initial[0] = size if load_balance == "memory" else size / 2
    targets = _weighted_quotas(required - size, minimum_remaining, weights, minimum_initial)
    first_target = minimum_initial[0] + targets[0]
    low, high = 1, min(block_size, max(1, size // 2))
    while low < high:
        width = (low + high + 1) // 2
        blocks = _diagonal_blocks(size, width)
        storage = sum((end - start) * (end - start + 1) // 2 for start, end in blocks)
        work = storage if load_balance == "memory" else sum((b - a)**2 for a, b in blocks) / 2
        if storage <= capacities[0] and work <= first_target:
            low = width
        else:
            high = width - 1
    blocks = _diagonal_blocks(size, low)
    diagonal = sum((end - start) * (end - start + 1) // 2 for start, end in blocks)
    remaining = list(capacities)
    remaining[0] -= diagonal
    initial = [0.0] * len(capacities)
    # Rectangles cost ~4 flops/value; each symmetric b x b block costs ~2b².
    initial[0] = (diagonal if load_balance == "memory" else
                  sum((end - start)**2 for start, end in blocks) / 2)
    quotas = _weighted_quotas(required - diagonal, remaining, weights, initial)
    assignments = [[] for _ in capacities]
    available = quotas.copy()
    for index, (row, stop) in enumerate(blocks):
        for column, end in blocks[index + 1:]:
            pending = [(row, stop, column, end)]
            while pending:
                tile = pending.pop()
                count = (tile[1] - tile[0]) * (tile[3] - tile[2])
                device = max(range(len(available)), key=available.__getitem__)
                if count <= available[device]:
                    assignments[device].append(tile)
                    available[device] -= count
                else:
                    pending.extend(_split_rectangle(tile))
    counts = quotas.copy()
    counts[0] += diagonal
    return blocks, assignments, counts


def _copy_matrix_values(destination, source):
    # Stage cross-device source tiles through CPU. This avoids depending on
    # peer-copy support/correctness between heterogeneous CUDA devices.
    if source.is_cuda and source.device != destination.device:
        source = source.cpu()
    destination.copy_(source)


def pack_diagonal_pairs(matrix, blocks, packed):
    """Pack paired triangles as column-major (b+1) x b rectangles.

    Return (global start, size, element offset, lda, upper) SYMV descriptors.
    B's lower triangle is copied from its upper triangle using symmetry.
    """
    specs = []
    offset = 0
    for index in range(0, len(blocks) - 1, 2):
        start, stop = blocks[index]
        second, _ = blocks[index + 1]
        size = stop - start
        count = size * (size + 1)
        pair = packed[offset:offset + count].view(size, size + 1).T
        for column in range(size):
            _copy_matrix_values(pair[:column + 1, column],
                                matrix[start:start + column + 1, start + column])
            _copy_matrix_values(pair[column + 1:, column],
                                matrix[second + column, second + column:second + size])
        specs.extend([(start, size, offset, size + 1, True),
                      (second, size, offset + 1, size + 1, False)])
        offset += count
    if len(blocks) % 2:
        start, _ = blocks[-1]
        _copy_matrix_values(packed[offset], matrix[start, start])
        specs.append((start, 1, offset, 1, True))
        offset += 1
    return specs, offset


def pack_rectangles(matrix, tiles, packed, offset=0):
    specs = []
    for row, stop, column, end in tiles:
        count = (stop - row) * (end - column)
        _copy_matrix_values(packed[offset:offset + count].view(stop - row, end - column),
                            matrix[row:stop, column:end])
        specs.append((row, stop, column, end, offset))
        offset += count
    return specs


def accumulate_rectangles(packed, specs, vector, output):
    for row, stop, column, end, offset in specs:
        count = (stop - row) * (end - column)
        block = packed[offset:offset + count].view(stop - row, end - column)
        output[row:stop].addmv_(block, vector[column:end])
        output[column:end].addmv_(block.T, vector[row:stop])
