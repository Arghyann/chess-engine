import chess
import chess.svg

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")

svg_code = chess.svg.board(
    board,
    fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
    arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
    squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
    size=350,
)

# Save to file
with open("board.svg", "w") as f:
    f.write(svg_code)

print("SVG saved to board.svg. Open it in a browser to view the board.")