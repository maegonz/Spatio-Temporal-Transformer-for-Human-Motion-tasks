import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn_motion(batch, pad_token_id: int = 0):
    """
    Custom collate function to pad motion to the max length in the batch and create an attention mask.

    Params
    -------
    batch : list[dict]
        List of motions in the batch.
    
    Returns
    -------
    padded_batch : tensor
        Padded batch of motions.
    attn_mask : tensor
        Attention mask indicating the valid positions in the sequences.
    """

    # Compute the maximum length in the batch
    lengths = [item['motion'].shape[0] if item['motion'].ndim==3 else 1 for item in batch]
    assert len(lengths) > 0, "lenghts empty"
    batch_size, max_length = len(batch), max(lengths)

    assert batch[0]['motion'].ndim == 3 or batch[0]['motion'].shape == torch.Size([22, 3]), f"Expected motion to have shape (T, num_joints, coords) instead got shape {batch[0]['motion'].shape} and batch size {len(batch)}"
    if batch[0]['motion'].ndim == 3:
        _, num_joints, coords = batch[0]['motion'].shape
    else:
        num_joints, coords = batch[0]['motion'].shape

    padded_motions = torch.zeros(batch_size, max_length, num_joints, coords)  # Assuming motion has shape (T, num_joints, coords)
    motion_masks = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, (item, l) in enumerate(zip(batch, lengths)):
        padded_motions[i, :l] = item['motion']
        motion_masks[i, :l] = True

    if 'captions' not in batch[0]:
        return {
            "motion": padded_motions,
            "attn_mask": motion_masks
        }

    # Padded inputs ids
    ids = pad_sequence(
        [torch.tensor(item["input_ids"]) for item in batch],
        batch_first=True, padding_value=pad_token_id
    )

    return {
        "motion": padded_motions,
        "attn_mask": motion_masks,
        "input_ids": ids,
        # "captions": [item['captions'] for item in batch],
        "references": [item["references"] for item in batch],
        "names": [item["name"] for item in batch]
    }