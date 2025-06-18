import random
import torch
from BioMol.utils.hierarchy import BioMolStructure
from BioMol.utils.error import NoInterfaceError


def get_chain_crop_indices(
    residue_chain_break: dict[str, tuple[int, int]], crop_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    chain_crop = {}
    for chain in residue_chain_break.keys():
        chain_start, chain_end = residue_chain_break[chain]
        minus_start = crop_indices - chain_start
        minus_end = crop_indices - chain_end
        crop_chain = (minus_start >= 0) & (minus_end <= 0)
        if crop_chain.sum() == 0:
            continue
        chain_crop[chain] = crop_indices[crop_chain] - chain_start
    return chain_crop


def crop_contiguous(
    biomolstructure: BioMolStructure, crop_length: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a contiguous region
    """
    residue_chain_break = biomolstructure.residue_chain_break
    n_added = 0
    chain_id = list(residue_chain_break.keys())
    n_remaining = residue_chain_break[chain_id[-1]][1] + 1
    shuffled_chain_id = random.sample(chain_id, len(chain_id))

    crop_indices = []

    for chain in shuffled_chain_id:
        chain_start = residue_chain_break[chain][0]
        chain_end = residue_chain_break[chain][1]
        n_k = chain_end - chain_start + 1
        n_remaining -= n_k
        crop_max = min(crop_length - n_added, n_k)
        crop_min = min(n_k, max(0, crop_length - n_remaining - n_added))
        crop = random.randint(crop_min, crop_max)
        n_added += crop
        crop_start = random.randint(0, n_k - crop) + chain_start
        crop_indices.extend(list(range(crop_start, crop_start + crop)))
    crop_indices = sorted(crop_indices)
    crop_indices = torch.tensor(crop_indices)

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)

    return crop_indices, chain_crop


def crop_spatial(
    chain_bias: str | None, biomolstructure: BioMolStructure, crop_length: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a spatial region
    """
    residue_chain_break = biomolstructure.residue_chain_break
    residue_tensor = biomolstructure.residue_tensor
    (valid_residue_indices,) = torch.where(residue_tensor[:, 4] == 1)
    valid_residue_num = valid_residue_indices.size(0)
    if valid_residue_num < crop_length:
        chain_crop = get_chain_crop_indices(residue_chain_break, valid_residue_indices)

        return valid_residue_indices, chain_crop

    chain_list = list(residue_chain_break.keys())
    if chain_bias is not None:
        assert chain_bias in chain_list, (
            f"Invalid chain: {chain_bias} \
            chain_list = {chain_list}"
        )
        pivot_chain = chain_bias
    else:
        pivot_chain = random.choice(chain_list)
    pivot_chain_residue_idx = list(
        range(
            residue_chain_break[pivot_chain][0], residue_chain_break[pivot_chain][1] + 1
        )
    )
    pivot_residue = random.choice(pivot_chain_residue_idx)

    residue_xyz = residue_tensor[:, 5:8]
    residue_mask = residue_tensor[:, 4] == 1
    missing_indices = torch.where(~residue_mask)[0]
    pivot_residue_tensor = residue_tensor[pivot_residue]
    pivot_residue_xyz = pivot_residue_tensor[5:8]

    distance_map = torch.norm(residue_xyz - pivot_residue_xyz, dim=1)
    distance_map[~residue_mask] = float("inf")

    crop_indices = torch.topk(distance_map, crop_length, largest=False).indices

    # remove missing residues
    crop_indices = crop_indices[~torch.isin(crop_indices, missing_indices)]
    crop_indices = torch.sort(crop_indices).values

    crop_chain = []
    for chain in chain_list:
        chain_residue_idx = torch.arange(
            residue_chain_break[chain][0], residue_chain_break[chain][1] + 1
        )
        if torch.any(torch.isin(crop_indices, chain_residue_idx)):
            crop_chain.append(chain)

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)

    return crop_indices, chain_crop


def crop_spatial_interface(
    interface_bias: tuple[str, str] | None,
    biomolstructure: BioMolStructure,
    crop_length: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a spatial region
    """
    interface_distance_cutoff = 15.0

    residue_chain_break = biomolstructure.residue_chain_break
    residue_tensor = biomolstructure.residue_tensor
    (valid_residue_indices,) = torch.where(residue_tensor[:, 4] == 1)
    valid_residue_num = valid_residue_indices.size(0)
    if valid_residue_num < crop_length:
        chain_crop = get_chain_crop_indices(residue_chain_break, valid_residue_indices)
        return valid_residue_indices, chain_crop

    chain_list = list(residue_chain_break.keys())

    # 20250601 psk
    # 6yfs A,B valid residues are filtered out by signalp
    # so in this case, we need to define valid_chain_list
    valid_chain_list = []
    residue_mask = residue_tensor[:, 4] == 1
    for chain in chain_list:
        chain_start, chain_end = residue_chain_break[chain]
        if torch.any(residue_mask[chain_start : chain_end + 1]):
            valid_chain_list.append(chain)
    if len(valid_chain_list) == 0:
        raise ValueError("No valid chains found in the biomolstructure.")

    if interface_bias is not None and (
        interface_bias[0] not in valid_chain_list
        or interface_bias[1] not in valid_chain_list
    ):
        raise NoInterfaceError(f"No interface found for {interface_bias}")
    if interface_bias is not None:
        assert interface_bias[0] in chain_list, (
            f"Invalid chain: {interface_bias[0]} \
            chain_list = {chain_list}"
        )
        assert interface_bias[1] in chain_list, (
            f"Invalid chain: {interface_bias[1]} \
            chain_list = {chain_list}"
        )
        pivot_chain = interface_bias[0]
    else:
        pivot_chain = random.choice(valid_chain_list)
    pivot_chain_id = chain_list.index(pivot_chain)

    if interface_bias is not None:
        crop_chains = [interface_bias[1]]
    else:
        contact_graph = biomolstructure.contact_graph
        contact_nodes = contact_graph.get_contact_node(None, pivot_chain_id)
        crop_nodes = contact_nodes
        crop_chains = [chain_list[i] for i in crop_nodes]
        if len(crop_chains) == 0:
            raise NoInterfaceError(f"No interface found for chain {pivot_chain}")
    crop_chains_residue_idx = []
    for chain in crop_chains:
        crop_chains_residue_idx.extend(
            list(range(residue_chain_break[chain][0], residue_chain_break[chain][1] + 1))
        )
    crop_chains_residue_idx = sorted(crop_chains_residue_idx)
    crop_chains_residue_tensor = residue_tensor[crop_chains_residue_idx]
    crop_chains_residue_xyz = crop_chains_residue_tensor[:, 5:8]
    crop_chain_residue_mask = crop_chains_residue_tensor[:, 4] == 1

    pivot_chain_residues_idx = list(
        range(
            residue_chain_break[pivot_chain][0], residue_chain_break[pivot_chain][1] + 1
        )
    )
    pivot_chain_residue_tensor = residue_tensor[pivot_chain_residues_idx]
    pivot_chain_residue_xyz = pivot_chain_residue_tensor[:, 5:8]
    pivot_chain_residue_mask = pivot_chain_residue_tensor[:, 4] == 1

    distance_map = pivot_chain_residue_xyz[:, None] - crop_chains_residue_xyz[None]
    distance_map = torch.norm(distance_map, dim=2)
    distance_map[~pivot_chain_residue_mask, :] = float("inf")
    distance_map[:, ~crop_chain_residue_mask] = float("inf")
    interface_residues = distance_map.min(dim=1).values < interface_distance_cutoff
    interface_residue_indices = torch.where(interface_residues)[0]
    if len(interface_residue_indices) == 0:
        raise NoInterfaceError(
            f"No interface residues found for chain {pivot_chain} with cutoff {interface_distance_cutoff}"
        )  # TODO bug
    pivot_residue_idx = random.choice(interface_residue_indices)
    pivot_residue_idx += residue_chain_break[pivot_chain][0]

    # pivot_residue = random.choice(pivot_chain_residue_idx)

    residue_xyz = residue_tensor[:, 5:8]
    residue_mask = residue_tensor[:, 4] == 1
    missing_indices = torch.where(~residue_mask)[0]
    pivot_residue_tensor = residue_tensor[pivot_residue_idx]
    pivot_residue_xyz = pivot_residue_tensor[5:8]

    distance_map = torch.norm(residue_xyz - pivot_residue_xyz, dim=1)
    distance_map[~residue_mask] = float("inf")

    crop_indices = torch.topk(distance_map, crop_length, largest=False).indices

    # remove missing residues
    crop_indices = crop_indices[~torch.isin(crop_indices, missing_indices)]
    crop_indices = torch.sort(crop_indices).values

    crop_chain = []
    for chain in chain_list:
        chain_residue_idx = torch.arange(
            residue_chain_break[chain][0], residue_chain_break[chain][1] + 1
        )
        if torch.any(torch.isin(crop_indices, chain_residue_idx)):
            crop_chain.append(chain)

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)

    return crop_indices, chain_crop


# def get_crop_indices(
#     biomolstructure: BioMolStructure,
#     seq_hash_to_crop_indices: dict[str, torch.Tensor],
# ) -> list[torch.Tensor]:
#     contact_graph = biomolstructure.contact_graph
#     chain_id_to_seq_hash = biomolstructure.sequence_hash
#     seq_hash_to_chain_id = {value: key for key, value in chain_id_to_seq_hash.items()}

#     biomolstructure_hash = set(biomolstructure.sequence_hash.values())
#     cropped_hash = set(seq_hash_to_crop_indices.keys())

#     assert cropped_hash.issubset(biomolstructure_hash), (
#         f"cropped_hash: {cropped_hash} \
#         biomolstructure_hash: {biomolstructure_hash}"
#     )

#     seq_hash_to_num = {}
#     for seq_hash in cropped_hash:
#         assert len(seq_hash_to_crop_indices[seq_hash]) <= len(
#             seq_hash_to_chain_id[seq_hash]
#         ), (
#             f"seq_hash: {seq_hash} \
#             len(seq_hash_to_crop_indices[seq_hash]): {len(seq_hash_to_crop_indices[seq_hash])} \
#             len(seq_hash_to_chain_id[seq_hash]): {len(seq_hash_to_chain_id[seq_hash])}"
#         )
#         seq_hash_to_num[seq_hash] = len(seq_hash_to_crop_indices[seq_hash])

#     # find all combinations of cropped_hash
#     seq_hash_to_combinations = {}
#     for seq_hash, num in seq_hash_to_num.items():
#         chain_ids = seq_hash_to_chain_id.get(seq_hash, [])
#         seq_hash_to_combinations[seq_hash] = list(itertools.combinations(chain_ids, num))

#     contact_nodes = contact_graph.get_contact_node(None, pivot_chain_id)
