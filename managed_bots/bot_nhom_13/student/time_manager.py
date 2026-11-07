MAX_PLIES = 300
MOVE_OVERHEAD = 0.85
FREE_THINK = 0.45

 # Trả về soft limit và hard limit đơn vị giây
def allocate(time_remaining_ms: int, ply: int):

    # Nếu thời gian còn lại dưới 2 giây thì chỉ dùng 20-50 ms cho nước đi tiếp theo
    t = max(time_remaining_ms, 0) / 1000.0
    if t < 2.0:
        return 0.02, 0.05

    # Số lượt còn lại của mình
    own_moves_left = max(1, (MAX_PLIES - ply + 1) // 2)
    
    # Ván thường kết thúc sớm hơn 300 ply nên không cần tính quỹ giờ cho toàn bộ 300 ply. Giả sử ván kết thúc trong 70 lượt
    horizon = min(own_moves_left, 70)

    # Quỹ giờ còn lại có thể dùng để tính nước đi tiếp theo (trừ đi overhead và 1.5 giây dự phòng)
    spendable = t - MOVE_OVERHEAD * horizon - 1.5

    # Chia quỹ giờ còn lại cho khoảng 20-40 lượt để tính soft limit. Nếu spendable < 0 thì soft limit = 0
    divisor = max(20, min(40, horizon))
    soft = spendable / divisor if spendable > 0 else 0.0


    soft = max(soft, FREE_THINK)
    soft = min(soft, t * 0.10)           # Không dùng quá 10% quỹ giờ
    hard = min(soft * 2.2, t * 0.15)     # Không dùng quá 15% quỹ giờ, không quá 2.2 lần soft limit
    return soft, max(hard, soft)