import chess 
import numpy as np



def board_to_tensor(board:chess.Board):
    tensor= np.zeros((19, 8, 8), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type - 1    #0= pawn, 1= knight, ..., 5= king
            colour= piece.color
            tensor[piece_type + (6 if colour else 0), square // 8, square % 8] = 1 #colour 0= white, 1= black
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1        
    if board.ep_square is not None:
        ep_square = board.ep_square
        tensor[16, ep_square // 8, ep_square % 8] = 1
    tensor[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    # --- 1 whose turn plane ---
    tensor[18, :, :] = 1 if board.turn == chess.WHITE else 0
    return tensor


