import torch

# Example idx_map: each row is [old_idx, new_idx]
# (e.g., old indices: 0, 2, 4; new indices: 0, 1, 2)
idx_map = torch.tensor([
    [0, 0],
    [2, 1],
    [4, 2]
])
# If not already sorted by the old indices, sort it:
# idx_map, _ = torch.sort(idx_map, dim=0)  # But here we assume it’s sorted.

# The original pair tensor.
# Each row is [src, dst, some_other_info]
pairs = torch.tensor([
    [0, 1, 99],
    [0, 2, 88],
    [1, 2, 77],
    [2, 3, 66]
])

# Step 1. Filtering: Only keep rows where both endpoints are in idx_map.
# The allowed (old) indices are the first column of idx_map.
allowed_old = idx_map[:, 0]

# Create a boolean mask. (torch.isin is available in recent PyTorch versions.)
mask = torch.isin(pairs[:, 0], allowed_old) & torch.isin(pairs[:, 1], allowed_old)
filtered_pairs = pairs[mask]

# At this point, filtered_pairs will be:
# tensor([[0, 2, 88]])
# because only the edge [0,2,88] has both endpoints in allowed_old.

# Step 2. (Optional) Remap the endpoints to new indices.
# Since allowed_old is sorted, we can use torch.searchsorted to perform the remapping.
# Get the new indices from idx_map.
new_idx = idx_map[:, 1]

# Use torch.searchsorted to map the old endpoints.
# (searchsorted returns the index where each value would be inserted in allowed_old;
#  since all our values are known to be in allowed_old, this works for our case.)
mapped_src = new_idx[torch.searchsorted(allowed_old, filtered_pairs[:, 0])]
mapped_dst = new_idx[torch.searchsorted(allowed_old, filtered_pairs[:, 1])]

# Create a new tensor for the remapped pairs.
remapped_pairs = filtered_pairs.clone()  # Clone to preserve other columns.
remapped_pairs[:, 0] = mapped_src
remapped_pairs[:, 1] = mapped_dst

print("Filtered pairs (with old indices):")
print(filtered_pairs)
print("\nRemapped pairs (with new indices):")
print(remapped_pairs)
