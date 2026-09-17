"""Milan ızgarası ile hücre kimliği arasındaki dönüşümler.

Izgara 100x100'dür. square_id 1'den 10000'e kadar numaralanır ve
ızgarayı satır satır gezer. Satır ve sütun indeksleri 0'dan başlar.
"""

from src.config import GRID_SIZE


def cell_position(cell_id):
    """square_id -> (satır, sütun). Izgara sol üstten 0,0 ile başlar."""
    row = (cell_id - 1) // GRID_SIZE
    column = (cell_id - 1) % GRID_SIZE
    return row, column


def position_to_cell(row, column):
    """(satır, sütun) -> square_id. cell_position'ın tersidir."""
    return row * GRID_SIZE + column + 1


def get_window(center_id, radius):
    """Merkez hücre çevresindeki kare pencerenin hücre kimliklerini döndürür.

    radius=10 -> 20x20 = 400 hücre (merkez dahil, üst sınır hariç).
    Izgara dışına taşan konumlar atlanır, bu yüzden kenara yakın
    merkezlerde pencere beklenenden küçük olabilir.
    """
    center_row, center_col = cell_position(center_id)
    cell_ids = []

    for row in range(center_row - radius, center_row + radius):
        if not 0 <= row < GRID_SIZE:
            continue
        for col in range(center_col - radius, center_col + radius):
            if not 0 <= col < GRID_SIZE:
                continue
            cell_ids.append(position_to_cell(row, col))

    return cell_ids