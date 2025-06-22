import torch

cache_file = "train_cache.pt"

print(f"Loading {cache_file} ...")
data = torch.load(cache_file)

# Case 1 — If it's dict of tensors (correct)
if isinstance(data, dict):
    board_tensors = data.get('board_tensors', None)
    move_indices = data.get('move_indices', None)

    if board_tensors is not None and move_indices is not None:
        print(f"✅ Cache format looks correct (dict of tensors)")
        print(f"  board_tensors: {type(board_tensors)}, shape: {board_tensors.shape}")
        print(f"  move_indices: {type(move_indices)}, shape: {move_indices.shape}")

        print("\nEstimated file size:")
        size_MB = (board_tensors.numel() * 4 + move_indices.numel() * 8) / 1024**2
        print(f"  ~{size_MB:.1f} MB (expected)")

    else:
        print("⚠️ Dict format found but keys missing!")

# Case 2 — If it's whole dataset object (wrong)
elif hasattr(data, 'board_tensors') and hasattr(data, 'move_indices'):
    print("⚠️ Found ChessDataset object!")
    print(f"  board_tensors type: {type(data.board_tensors)}")
    print(f"  move_indices type: {type(data.move_indices)}")

    if isinstance(data.board_tensors, list):
        print("❌ board_tensors is a LIST — this will cause huge file size")
    else:
        print("✅ board_tensors is a Tensor")

# Unknown format
else:
    print("❓ Unknown file format!")
