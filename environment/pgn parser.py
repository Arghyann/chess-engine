import chess.pgn

# Open PGN file
pgn = open(r"D:\projects\python\chess engine\chess-engine\lichess db\lichess_db_standard_rated_2017-03.pgn", encoding="utf-8")

# How many games to process
max_games = 5
game_counter = 0

while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    
    print(f"\n=== Game {game_counter + 1} ===")
    print(game.headers.get("White", "?"), "vs", game.headers.get("Black", "?"))

    # Play through the game and print moves
    board = game.board()
    move_counter = 1
    for move in game.mainline_moves():
        move_san = board.san(move)  # <--- must be before push()
        board.push(move)
        print(f"{move_counter}. {move_san}", end="  ")
        move_counter += 1
    
    game_counter += 1
    if game_counter >= max_games:
        break
