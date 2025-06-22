import sys
import os

# Insert project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from environment.boardToTensor import board_to_tensor
from dataprep.encodeDecode import encode
import pickle

from torch.utils.data import Dataset
import torch
import pickle
from tqdm import tqdm

class ChessDataset(Dataset):
    def __init__(self, fen_list, move_list):
        print(f"Preprocessing {len(fen_list)} chess positions...")
        
      
        self.board_tensors = []
        self.move_indices = []
        
       
        for i, (fen, move_uci) in enumerate(tqdm(zip(fen_list, move_list), 
                                                  total=len(fen_list), 
                                                  desc="Converting positions")):
            board_tensor = board_to_tensor(fen)
            move_index = encode(fen, move_uci)
            
            self.board_tensors.append(board_tensor)
            self.move_indices.append(move_index)
        
        # Convert to tensors once
        self.board_tensors = torch.stack(self.board_tensors)
        self.move_indices = torch.tensor(self.move_indices, dtype=torch.long)
        
        print("Preprocessing complete!")

    def __len__(self):
        return len(self.board_tensors)

    def __getitem__(self, idx):
        # Now this is instant - just tensor indexing
        return self.board_tensors[idx], self.move_indices[idx]

    def save_cache(self, filepath):
        """Save preprocessed data to avoid recomputing"""
        print(f"Saving preprocessed data to {filepath}")
        torch.save({
            'board_tensors': self.board_tensors,
            'move_indices': self.move_indices
        }, filepath)

    @classmethod
    def load_from_cache(cls, filepath):
        """Load preprocessed data from cache"""
        print(f"Loading preprocessed data from {filepath}")
        data = torch.load(filepath)
        
        # Create empty instance
        instance = cls.__new__(cls)
        instance.board_tensors = data['board_tensors']
        instance.move_indices = data['move_indices']
        
        return instance

