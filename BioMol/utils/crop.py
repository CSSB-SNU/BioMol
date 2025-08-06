import random
import torch
from typing import Literal
from BioMol.utils.hierarchy import BioMolStructure
from BioMol.utils.error import NoInterfaceError, NoValidChainsError


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
    chain_bias: str | None,
    biomolstructure: BioMolStructure,
    crop_length: int,
    level: Literal["atom", "residue"] = "atom",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a spatial region
    """
    residue_chain_break = biomolstructure.residue_chain_break
    residue_tensor = biomolstructure.residue_tensor
    valid_residue_indices = torch.where(residue_tensor[:, 4] == 1)[0]
    if level == "atom":
        atom_chain_break = biomolstructure.atom_chain_break
        atom_tensor = biomolstructure.atom_tensor
        _, atom_to_residue_idx_map = torch.unique(
            atom_tensor[:, 2], sorted=True, return_inverse=True
        )
        atom_mask = atom_tensor[:, 4] == 1

    if level == "residue":
        valid_indices = valid_residue_indices
    else:  # level == "atom"
        valid_indices = torch.where(atom_mask)[0]

    valid_num = valid_indices.size(0)

    if valid_num < crop_length:
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

    if level == "residue":
        pivot_chain_idx = list(
            range(
                residue_chain_break[pivot_chain][0],
                residue_chain_break[pivot_chain][1] + 1,
            )
        )
        pivot_idx = random.choice(pivot_chain_idx)

        xyz = residue_tensor[:, 5:8]
        mask = residue_tensor[:, 4] == 1
    else:  # level == "atom"
        pivot_chain_idx = list(
            range(atom_chain_break[pivot_chain][0], atom_chain_break[pivot_chain][1] + 1)
        )
        pivot_idx = random.choice(pivot_chain_idx)
        xyz = atom_tensor[:, 5:8]
        mask = atom_tensor[:, 4] == 1

    missing_indices = torch.where(~mask)[0]
    pivot_xyz = xyz[pivot_idx]

    distance_map = torch.norm(xyz - pivot_xyz, dim=1)
    distance_map[~mask] = float("inf")

    crop_indices = torch.topk(distance_map, crop_length, largest=False).indices

    # remove missing residues
    crop_indices = crop_indices[~torch.isin(crop_indices, missing_indices)]
    crop_indices = torch.sort(crop_indices).values

    if level == "atom":
        crop_indices = atom_to_residue_idx(
            crop_indices, atom_to_residue_idx_map, valid_residue_indices, atom_mask
        )

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)

    return crop_indices, chain_crop


def crop_spatial_interface(
    interface_bias: tuple[str, str] | None,
    biomolstructure: BioMolStructure,
    crop_length: int,
    level: Literal["atom", "residue"] = "atom",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a spatial region
    """
    interface_distance_cutoff = 15.0

    residue_chain_break = biomolstructure.residue_chain_break
    residue_tensor = biomolstructure.residue_tensor
    valid_residue_indices = torch.where(residue_tensor[:, 4] == 1)[0]
    if level == "atom":
        atom_chain_break = biomolstructure.atom_chain_break
        atom_tensor = biomolstructure.atom_tensor
        _, atom_to_residue_idx_map = torch.unique(
            atom_tensor[:, 2], sorted=True, return_inverse=True
        )
        atom_mask = atom_tensor[:, 4] == 1

    if level == "residue":
        valid_indices = valid_residue_indices
    else:  # level == "atom"
        valid_indices = torch.where(atom_tensor[:, 4] == 1)[0]

    valid_num = valid_indices.size(0)
    if valid_num < crop_length:
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

    if level == "residue":
        pivot_idx = pivot_residue_idx
        xyz = residue_tensor[:, 5:8]
        mask = residue_tensor[:, 4] == 1
    else:  # level == "atom"
        pivot_idxs = torch.where(atom_to_residue_idx_map == pivot_residue_idx)[0]
        if len(pivot_idxs) == 0:
            raise NoInterfaceError(
                f"No interface residues found for chain {pivot_chain} with cutoff {interface_distance_cutoff}"
            )  # TODO bug
        pivot_idx = random.choice(pivot_idxs)
        xyz = atom_tensor[:, 5:8]
        mask = atom_tensor[:, 4] == 1

    missing_indices = torch.where(~mask)[0]
    pivot_xyz = xyz[pivot_idx]

    distance_map = torch.norm(xyz - pivot_xyz, dim=1)
    distance_map[~mask] = float("inf")

    crop_indices = torch.topk(distance_map, crop_length, largest=False).indices

    # remove missing residues
    crop_indices = crop_indices[~torch.isin(crop_indices, missing_indices)]
    crop_indices = torch.sort(crop_indices).values

    if level == "atom":
        crop_indices = atom_to_residue_idx(
            crop_indices, atom_to_residue_idx_map, valid_residue_indices, atom_mask
        )

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)

    return crop_indices, chain_crop


def crop_contiguous_monomer(
    chain_bias: str | None,
    biomolstructure: BioMolStructure,
    crop_length: int,
    level: Literal["atom", "residue"] = "atom",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a contiguous region
    """
    residue_chain_break = biomolstructure.residue_chain_break
    residue_tensor = biomolstructure.residue_tensor
    valid_residue_indices = torch.where(residue_tensor[:, 4] == 1)[0]
    if level == "atom":
        atom_chain_break = biomolstructure.atom_chain_break
        atom_tensor = biomolstructure.atom_tensor
        _, atom_to_residue_idx_map = torch.unique(
            atom_tensor[:, 2], sorted=True, return_inverse=True
        )
        atom_mask = atom_tensor[:, 4] == 1

    def is_valid_chain(chain_id: str, min_residue_length=32) -> bool:
        chain_start, chain_end = residue_chain_break[chain_id]
        valid_residue_indices = torch.where(
            residue_tensor[chain_start : chain_end + 1, 4] == 1
        )[0]
        return valid_residue_indices.size(0) > min_residue_length

    chain_id = None
    if chain_bias is not None:
        chain_id = chain_bias if is_valid_chain(chain_bias) else None

    if chain_id is None:
        chain_id_list = list(residue_chain_break.keys())
        find_valid_chain = False
        while len(chain_id_list) > 0 and not find_valid_chain:
            chain_id = random.sample(chain_id_list, 1)[0]
            chain_id_list.remove(chain_id)
            if is_valid_chain(chain_id):
                find_valid_chain = True
        if not find_valid_chain:
            raise NoValidChainsError(
                f"No valid chains found in the {biomolstructure.ID}"
            )

    if level == "residue":
        chain_start, chain_end = residue_chain_break[chain_id]
        valid_indices = torch.where(residue_tensor[chain_start : chain_end + 1, 4] == 1)[
            0
        ]
    else:  # level == "atom"
        chain_start, chain_end = atom_chain_break[chain_id]
        valid_indices = torch.where(atom_tensor[chain_start : chain_end + 1, 4] == 1)[0]
    if valid_indices.size(0) < crop_length:
        crop_indices = chain_start + valid_indices
    else:
        # uniformly sample a contiguous region
        n_k = valid_indices.size(0)
        n_start = random.randint(0, n_k - crop_length)
        crop_indices = chain_start + valid_indices[n_start : n_start + crop_length]

    if level == "atom":
        crop_indices = atom_to_residue_idx(
            crop_indices, atom_to_residue_idx_map, valid_residue_indices, atom_mask
        )

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)
    return crop_indices, chain_crop


def crop_spatial_monomer(
    pivot_chain: str,
    biomolstructure: BioMolStructure,
    crop_length: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Crop the structure and sequence into a spatial region
    """
    assert pivot_chain is not None, "pivot_chain must be provided for monomer cropping"
    residue_chain_break = biomolstructure.residue_chain_break
    residue_tensor = biomolstructure.residue_tensor
    idx_start, idx_end = residue_chain_break[pivot_chain]
    residue_tensor = residue_tensor[idx_start : idx_end + 1]
    valid_residue_indices = torch.where(residue_tensor[:, 4] == 1)[0]

    valid_indices = valid_residue_indices
    valid_num = valid_indices.size(0)

    if valid_num < crop_length:
        chain_crop = get_chain_crop_indices(residue_chain_break, valid_residue_indices)
        return valid_residue_indices, chain_crop

    pivot_chain_idx = list(range(0, idx_end - idx_start + 1))
    pivot_idx = random.choice(pivot_chain_idx)

    xyz = residue_tensor[:, 5:8]
    mask = residue_tensor[:, 4] == 1

    missing_indices = torch.where(~mask)[0]
    pivot_xyz = xyz[pivot_idx]

    distance_map = torch.norm(xyz - pivot_xyz, dim=1)
    distance_map[~mask] = float("inf")

    crop_indices = torch.topk(distance_map, crop_length, largest=False).indices

    # remove missing residues
    crop_indices = crop_indices[~torch.isin(crop_indices, missing_indices)]
    crop_indices = torch.sort(crop_indices).values

    chain_crop = get_chain_crop_indices(residue_chain_break, crop_indices)
    return crop_indices, chain_crop


def atom_to_residue_idx(
    crop_indices_atom: torch.Tensor,
    atom_to_residue_idx_map: torch.Tensor,
    valid_residue_indices: torch.Tensor,
    atom_mask: torch.Tensor,
):
    """
    Convert atom indices to residue indices.
    """
    # residue_indices = atom_to_residue_idx_map[crop_indices_atom]
    # # Ensure residue indices are unique
    # residue_indices = torch.unique(residue_indices)
    # # remove invalid residues
    # valid_mask = torch.isin(residue_indices, valid_residue_indices)
    # return residue_indices[valid_mask]

    crop_atom = torch.zeros_like(atom_to_residue_idx_map)
    crop_atom[crop_indices_atom] = 1

    ones = torch.ones_like(crop_atom)

    max_res_id = int(atom_to_residue_idx_map.max().item())
    present_counts = torch.zeros(max_res_id + 1, dtype=torch.long)
    total_counts = torch.zeros_like(present_counts)

    present_counts.scatter_add_(0, atom_to_residue_idx_map, crop_atom)
    total_counts.scatter_add_(0, atom_to_residue_idx_map, atom_mask.long())

    full_res_mask = present_counts == total_counts

    valid_full = full_res_mask[valid_residue_indices]
    residue_idx = valid_residue_indices[valid_full]
    residue_idx = torch.unique(residue_idx)
    return residue_idx
