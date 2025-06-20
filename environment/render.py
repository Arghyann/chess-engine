import chess
import chess.svg
import argparse
import os

def generate_svg_from_fen(fen, output_path="board.svg", size=350):
    """
    Generate an SVG file for a chess position given in FEN notation.
    
    Args:
        fen (str): FEN string representing the chess position
        output_path (str): Path to save the SVG file
        size (int): Size of the board in pixels
    """
    try:
        # Create a board from the FEN string
        board = chess.Board(fen)
        
        # Generate SVG
        svg_code = chess.svg.board(
            board,
            size=size,
        )
        
        # Save to file
        with open(output_path, "w") as f:
            f.write(svg_code)
            
        return True, f"SVG saved to {output_path}"
    except Exception as e:
        return False, f"Error generating SVG: {str(e)}"

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate SVG image from chess FEN string")
    parser.add_argument("--fen", type=str, default="3r1b1R/pp2kpP1/2n5/4P1P1/3pb3/PP2nNKB/1B6/8 w - - 1 29",
                       help="FEN string representing the chess position")
    parser.add_argument("--output", type=str, default="board.svg",
                       help="Output file path for the SVG")
    parser.add_argument("--size", type=int, default=350,
                       help="Size of the board in pixels")
    
    args = parser.parse_args()
    
    # Generate the SVG
    success, message = generate_svg_from_fen(args.fen, args.output, args.size)
    
    print(message)
    
    if success:
        print(f"Open it in a browser to view the board.")