from chesslab_sdk import SearchContext


def allocate_move_time(context: SearchContext) -> float:
    """Tính số giây bot nên dùng để suy nghĩ cho nước hiện tại."""
    remaining = max(0.0, context.time_remaining_ms / 1000)
    if remaining <= 2.0:
        return 0

    # Ước lượng số lượt bot còn phải đi (không vượt trần 300 ply)
    until_limit = max(1, (300 - context.ply + 1) // 2)
    moves_left = min(max(15, 50 - context.ply // 5), until_limit)

    # Dự trữ an toàn sàn 0.5s nhưng không vượt quá 40% quỹ thời gian còn lại
    reserved = min(moves_left * 0.55, remaining * 0.40)
    usable = max(0.0, remaining - reserved)

    phase_factor = 0.90 if context.ply < 24 else 1.25 if context.ply < 120 else 1.00
    budget = (usable / moves_left) * phase_factor

    # Trần mỗi nước: tối đa 6.0s và không quá 15% tổng thời gian còn lại
    return max(0.05, min(budget, min(6.0, remaining * 0.15)))
