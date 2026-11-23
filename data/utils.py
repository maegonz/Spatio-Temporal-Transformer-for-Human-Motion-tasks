import torch

def collate_fn_motion(batch, tokenizer):
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
    max_length = max(lengths)

    assert batch[0]['motion'].ndim == 3 or batch[0]['motion'].shape == torch.Size([22, 3]), f"Expected motion to have shape (T, num_joints, coords) instead got shape {batch[0]['motion'].shape} and batch size {len(batch)}"
    if batch[0]['motion'].ndim == 3:
        _, num_joints, coords = batch[0]['motion'].shape
    else:
        num_joints, coords = batch[0]['motion'].shape

    padded_motions = torch.zeros(len(batch), max_length, num_joints, coords)  # Assuming motion has shape (T, num_joints, coords)
    motion_masks = torch.zeros(len(batch), max_length, dtype=torch.bool)

    for i, item in enumerate(batch):
        l = lengths[i]
        padded_motions[i, :l] = item['motion']
        motion_masks[i, :l] = True

    if 'captions' not in batch[0]:
        return {
            "motion": padded_motions,
            "attn_mask": motion_masks
        }

    # Concatenate captions texts, tokens and t5 attention masks
    caption_texts = [item['captions'] for item in batch]
    list_input_ids = [item['input_ids'] for item in batch]

    # Pad the input_ids and attention masks for T5
    text_features = tokenizer.pad(
        {
            "input_ids": list_input_ids,
        },
        padding=True,
        return_tensors="pt",
    )

    return {
        "motion": padded_motions,
        "attn_mask": motion_masks,
        "input_ids": text_features["input_ids"],
        "captions": caption_texts
    }