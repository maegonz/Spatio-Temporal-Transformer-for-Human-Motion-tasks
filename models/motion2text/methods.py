import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from ..metrics import contrastive_loss, cross_entropy_loss
from torch.profiler import profile, record_function, ProfilerActivity


def training(model: nn.Module,
            train_loader: DataLoader,
            optimizer: Optimizer,
            device: torch.device,
            epochs: int,
            prefix_ids: torch.Tensor,
            val_loader: DataLoader=None,
            alpha: float=0.1,
            use_amp: bool=True):
    """
    Train a PyTorch model with optional Automatic Mixed Precision.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        DataLoader providing the training dataset.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    prefix_ids : torch.Tensor
        Tensor containing the IDs of the prefix tokens.
    val_loader : DataLoader, optional
        DataLoader providing the validation dataset.
        If None, no validation is performed. Default is None.
    alpha : float, optional
        Weighting factor for the contrastive loss. Default is 0.1.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    train_losses : list of float
        Average training loss for each epoch.
    train_accuracies : list of float
        Training accuracy (percentage) for each epoch.
    val_losses : list of float
        Validation loss for each epoch.
        Empty if val_loader is None.
    val_accuracies : list of float
        Validation accuracy (percentage) for each epoch.
        Empty if val_loader is None.
    """

    model.to(device)
    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    train_losses = []
    val_metrics = []

    pad_token_id = train_loader.dataset.tokenizer.pad_token_id

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()
        running_loss = torch.tensor(0.0, device=device)

        for k, item in enumerate(train_loader):

            motion = item['motion'].to(device, non_blocking=True)
            captions_tokens = item['input_ids'].squeeze(1).to(device, non_blocking=True)
            encoder_attn_mask = item['attn_mask'].squeeze(1).to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(motion, captions_tokens, encoder_attn_mask=encoder_attn_mask)

                motion_features, text_features = outputs["motion_embeddings"], outputs["text_embeddings"]

                ce_loss = outputs["loss"] if outputs["loss"] is not None else cross_entropy_loss(outputs["decoder_output"], captions_tokens, pad_token_id)
                cl_loss = contrastive_loss(motion_features, text_features)

                # Combine them (lambda is a hyperparameter, start with 0.1)
                loss = ce_loss + (alpha * cl_loss)

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach() * motion.size(0)

        epoch_loss = running_loss.item() / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if val_loader is not None:
            val_scores = validation(
                model, val_loader, device, use_amp
            )
            val_metrics.append(val_scores)

            epoch_tqdm.set_postfix(
                train_loss=epoch_loss
            )
        else:
            epoch_tqdm.set_postfix(train_loss=epoch_loss)

    return train_losses, val_metrics


from ..metrics import Evaluator
from .transformers.blocks import TextTokenizer

def validation(model: nn.Module, 
               val_loader: DataLoader,
               device: torch.device,
               tokenizer: TextTokenizer,
               prefix_ids: torch.Tensor,
               use_amp: bool = True):
    """
    Evaluate a PyTorch model on a dataset with optional AMP.

    Parameters
    ----------
    model : nn.Module
        The trained model to be evaluated.
    val_loader : DataLoader
        DataLoader providing the validation dataset.
    device : torch.device
        Device on which validation is performed ('cpu' or 'cuda').
    tokenizer : TextTokenizer
        Tokenizer for processing text data.
    use_amp : bool, optional
        Whether to use Automatic Mixed Precision (AMP).
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    avg_loss : float
        Average loss over the entire dataset.
    avg_accuracy : float
        Average accuracy (percentage) over the entire dataset.
    """

    tokenizer.to(device)
    model.to(device)

    model.eval()
    evaluator = Evaluator()
    metrics_scores = []

    with torch.no_grad():
        for item in val_loader:
            all_references = []
            all_generated = []
            motion = item['motion'].to(device, non_blocking=True)
            captions_tokens = item['input_ids'].squeeze(1).to(device, non_blocking=True)
            encoder_attn_mask = item['attn_mask'].squeeze(1).to(device, non_blocking=True)
            captions = item['captions']

            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                outputs_ids = model(motion, captions_tokens, encoder_attn_mask=encoder_attn_mask, generation=True, prefix_ids=prefix_ids)
                outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)

            all_references.extend(captions)
            all_generated.extend(outputs)
            scores = evaluator.compute_metrics(all_references, all_generated)
            metrics_scores.append(scores)

    return metrics_scores