import sys
import os

# Insert project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from environment.boardToTensor import board_to_tensor
from dataprep.encodeDecode import encode
import pickle

class ChessDataset(Dataset):
    
    def __init__(self, fen_list, move_list):
        self.fen_list = fen_list
        self.move_list = move_list

    def __len__(self):
        return len(self.fen_list)

    def __getitem__(self, idx):
        fen = self.fen_list[idx]
        move_uci = self.move_list[idx]

        board_tensor = board_to_tensor(fen)
        move_index = encode(fen, move_uci)

        return board_tensor, move_index
