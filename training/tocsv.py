import chess.pgn
import chess.engine
import csv

# Paths
pgn_path = r"chess-engine\lichess db\lichess_db_standard_rated_2017-03.pgn"
engine_path = r"chess-engine\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"


engine = chess.engine.SimpleEngine.popen_uci(engine_path)


pgn = open(pgn_path, encoding="utf-8")


csvfile = open("training_data.csv", "w", newline='', encoding="utf-8")
writer = csv.writer(csvfile)
writer.writerow(["fen", "best_move"])  # header

max_positions = 100000
counter = 0

while counter < max_positions:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break

    board = game.board()

    for move in game.mainline_moves():
        fen = board.fen()

       
        info = engine.analyse(board, chess.engine.Limit(depth=10))
        best_move = info["pv"][0]

       
        writer.writerow([fen, best_move.uci()])

        
        board.push(move)
        counter += 1

        if counter % 1000 == 0:
            print(f"Processed {counter} positions")

        if counter >= max_positions:
            break

engine.quit()
csvfile.close()
print("Done!")
