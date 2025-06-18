import chess
import pickle

def generate_move_list():
    move_list = []

    
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue
            move_uci = chess.square_name(from_square) + chess.square_name(to_square)
            move_list.append(move_uci)

    
    promotion_pieces = ['q', 'r', 'b', 'n']

    for from_square in chess.SQUARES:
        rank = chess.square_rank(from_square)
        file = chess.square_file(from_square)

        
        if rank == 6:
            to_square = chess.square(file, 7)
            for promo_piece in promotion_pieces:
                move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                move_list.append(move_uci)

        
        if rank == 1:
            to_square = chess.square(file, 0)
            for promo_piece in promotion_pieces:
                move_uci = chess.square_name(from_square) + chess.square_name(to_square) + promo_piece
                move_list.append(move_uci)

    
    move_list.append("e1g1")  # White O-O
    move_list.append("e1c1")  # White O-O-O
    move_list.append("e8g8")  # Black O-O
    move_list.append("e8c8")  # Black O-O-O

    return move_list


list_of_moves = generate_move_list()
print(list_of_moves[4032]) 

def todict(x):
    dict = {i: x[i] for i in range(len(x))}
    with open("dictionary.pkl", "wb") as f:
        pickle.dump(dict, f)
        print("dumped")

todict(list_of_moves)