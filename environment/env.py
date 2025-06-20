import chess
import random
import numpy as np
from boardToTensor import board_to_tensor

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_obs()

    def get_obs(self):
        # FEN or custom encoding for the agent
        return self.board.fen()

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            return self.get_obs(), -1, True, {"illegal": True}

        self.board.push(move)

        done = self.board.is_game_over()
        reward = 0
        if done:
            result = self.board.result()
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0

        return self.get_obs(), reward, done, {}

    def render(self):
        print(self.board, "\n")


BoardEnv = ChessEnv()
BoardEnv.render()
BoardEnv.step("e2e4")
BoardEnv.step("e7e5")
BoardEnv.step("e1e2")

print(BoardEnv.get_obs())
BoardEnv.render()
print(np.shape(board_to_tensor(BoardEnv.board)))
print(board_to_tensor(BoardEnv.board)[20, :, :])
  # Print whose turn it is