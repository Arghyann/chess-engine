import chess
import pickle

def generate_move_list():
    move_list = []

    # All normal moves
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue
            move_uci = chess.square_name(from_square) + chess.square_name(to_square)
            move_list.append(move_uci)

    promotion_pieces = ['q', 'r', 'b', 'n']

    # White promotions
    for from_square in chess.SQUARES:
        rank = chess.square_rank(from_square)
        file = chess.square_file(from_square)
        if rank == 6:  # White pawn on rank 7
            to_rank = 7

            # Forward
            to_square = chess.square(file, to_rank)
            for promo_piece in promotion_pieces:
                move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                move_list.append(move_uci)

            # Capture left
            if file > 0:
                to_square = chess.square(file - 1, to_rank)
                for promo_piece in promotion_pieces:
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                    move_list.append(move_uci)

            # Capture right
            if file < 7:
                to_square = chess.square(file + 1, to_rank)
                for promo_piece in promotion_pieces:
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                    move_list.append(move_uci)

    # Black promotions
    for from_square in chess.SQUARES:
        rank = chess.square_rank(from_square)
        file = chess.square_file(from_square)
        if rank == 1:  # Black pawn on rank 2
            to_rank = 0

            # Forward
            to_square = chess.square(file, to_rank)
            for promo_piece in promotion_pieces:
                move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                move_list.append(move_uci)

            # Capture left
            if file > 0:
                to_square = chess.square(file - 1, to_rank)
                for promo_piece in promotion_pieces:
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                    move_list.append(move_uci)

            # Capture right
            if file < 7:
                to_square = chess.square(file + 1, to_rank)
                for promo_piece in promotion_pieces:
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                    move_list.append(move_uci)

    # Castling (last 4 moves)
    move_list.append("e1g1")  # White O-O
    move_list.append("e1c1")  # White O-O-O
    move_list.append("e8g8")  # Black O-O
    move_list.append("e8c8")  # Black O-O-O

    return move_list


list_of_moves = generate_move_list()
print(list_of_moves[4032]) 

def todict(x):
    dct = {i: x[i] for i in range(len(x))}
    with open("dictionary.pkl", "wb") as f:
        pickle.dump(dct, f)
        print("dumped")
#todict(list_of_moves)
with open("dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)
    print(dictionary[257])  # test lookup

    for index, move in dictionary.items():
        if move == "c2b1q":
            print(f"Found move {move} at index {index}")
            break
        else:
            print(f"Move {move} at index {index} does not match 'c2b1q'")