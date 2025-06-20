import pickle
import chess

# encodes best move or any move as an index for the output layer 
def encode(boardFEN,boardUCI):
    with open("dictionary.pkl", "rb") as f:
        dictionary = pickle.load(f)
    indexV=-1
    Castling = False
    board = chess.Board(boardFEN)
    if boardUCI in {"e1g1", "e1c1", "e8g8", "e8c8"}:
        kingSquare=boardUCI[0:2]
        piece = board.piece_at(chess.SQUARE_NAMES.index(kingSquare))
        if piece is not None and piece.symbol().lower() == 'k':
            Castling = True
        
    if Castling:
        last_4_items = list(dictionary.items())[-4:]
        for index, move in last_4_items:
            if move == boardUCI:
                indexV = index
                break
    else:
        for index, move in dictionary.items():
            if move == boardUCI:
                indexV = index
                break
    if indexV == -1:
        raise ValueError(f"Move {boardUCI} not found in dictionary.")
    return indexV
def decode(index):
    with open("dictionary.pkl", "rb") as f:
        dictionary = pickle.load(f)
    if index in dictionary:
        return dictionary[index]
    else:
        raise ValueError(f"Index {index} not found in dictionary.")
