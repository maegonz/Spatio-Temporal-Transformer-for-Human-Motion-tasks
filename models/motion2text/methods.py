import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast
from tqdm import tqdm
from ..metrics import contrastive_loss, cross_entropy_loss, Evaluator
# from torch.profiler import profile, record_function, ProfilerActivity


def training(model: nn.Module,
            train_loader: DataLoader,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            device: torch.device,
            epochs: int,
            label_smoothing: float=0.1,
            num_beams: int = 1,
            val_loader: DataLoader=None,
            val_step: int=25,
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
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning-rate scheduler stepped after each epoch.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    label_smoothing : float, optional
        Label smoothing factor for the cross-entropy loss. Default is 0.1.
    val_loader : DataLoader, optional
        DataLoader providing the validation dataset.
        If None, no validation is performed. Default is None.
    val_step : int, optional
        Run validation every this many epochs. Default is 25.
    alpha : float, optional
        Weighting factor for the contrastive loss. Default is 0.1.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    train_losses : list of float
        Average training loss for each epoch.
    val_metrics : list
        Validation metric dictionaries, populated at each validation step.
    val_losses : list of float
        Average validation loss at each validation step. Empty if val_loader is None.
    """

    model.to(device)
    use_amp = use_amp and device.type == "cuda"

    train_losses = []
    val_metrics, val_losses = [], []

    pad_token_id = train_loader.dataset.tokenizer.pad_token_id

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()
        running_loss = torch.tensor(0.0, device=device)

        for item in train_loader:
            optimizer.zero_grad(set_to_none=True)

            motion = item['motion'].to(device, non_blocking=True)
            captions_tokens = item['input_ids'].to(device, non_blocking=True)
            encoder_attn_mask = item['attn_mask'].to(device, non_blocking=True)

            # decoder input = start token and labels shifted
            labels = captions_tokens
            start = torch.zeros_like(labels[:, :1])
            decoder_input = torch.cat([start, labels[:, :-1]], dim=1)

            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(motion, decoder_input, encoder_attn_mask=encoder_attn_mask)

            motion_features, text_features, logit_scale = outputs["motion_embeddings"], outputs["text_embeddings"], outputs["logit_scale"]

            ce_loss = cross_entropy_loss(outputs["decoder_output"].float(), labels, label_smoothing, pad_token_id)
            cl_loss = contrastive_loss(motion_features.float(), text_features.float(), scale=logit_scale.float())

            # Combine them (lambda is a hyperparameter, start with 0.1)
            loss = ce_loss + (alpha * cl_loss)
        
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.detach() * motion.size(0)

        scheduler.step()

        epoch_loss = running_loss.item() / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if (val_loader is not None) and ((epoch+1) % val_step == 0):
            val_scores, val_loss = validation(
                model, val_loader, device, num_beams=num_beams, use_amp=use_amp
            )
            val_metrics.append(val_scores)
            val_losses.append(val_loss)

        epoch_tqdm.set_postfix(train_loss=epoch_loss)

    return train_losses, val_metrics, val_losses



@torch.no_grad()
def validation(model: nn.Module, 
               val_loader: DataLoader,
               device: torch.device,
               label_smoothing: float = 0.1,
               num_beams: int = 1,
               use_amp: bool = True):
    """
    Generate captions for the validation set and compute text metrics.

    Parameters
    ----------
    model : nn.Module
        The trained model to be evaluated.
    val_loader : DataLoader
        DataLoader providing the validation dataset.
    device : torch.device
        Device on which validation is performed ('cpu' or 'cuda').
    label_smoothing : float, optional
        Label smoothing factor for the cross-entropy loss. Default is 0.1.
    use_amp : bool, optional
        Whether to use Automatic Mixed Precision (AMP).
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    metrics_scores : dict
        Text evaluation metrics computed over the generated captions.
    val_loss : float
        Average cross-entropy loss over the entire validation dataset.
    """

    tokenizer = val_loader.dataset.tokenizer
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    
    model.to(device)
    model.eval()

    evaluator = Evaluator()
    all_references, all_generated = [], []
    running_loss = torch.tensor(0.0, device=device)

    for item in val_loader:
        motion = item['motion'].to(device, non_blocking=True)
        encoder_attn_mask = item['attn_mask'].to(device, non_blocking=True)
        captions = item['captions']
        labels = item['input_ids'].to(device, non_blocking=True)
        decoder_input = labels_shift(labels, pad_id)
        B = motion.size(0)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            # teacher forcing pass
            tf_output = model(motion, decoder_input, encoder_attn_mask=encoder_attn_mask)
            tf_ids = tf_output["decoder_output"]

            # free generation pass
            gen_output = model(motion, None, encoder_attn_mask=encoder_attn_mask, eos_token_id=eos_id, pad_token_id=pad_id, num_beams=num_beams)
            gen_ids = gen_output["decoder_output"]

        # loss
        loss = cross_entropy_loss(tf_ids.float(), labels, label_smoothing, pad_id)
        running_loss += loss.detach() * B

        # metrics
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_references.extend(captions)
        all_generated.extend(texts)

    metrics_scores = evaluator.compute_metrics(all_references, all_generated)
    val_loss = running_loss.item() / len(val_loader.dataset)

    return metrics_scores, val_loss

def labels_shift(labels: torch.Tensor, start_token: int = 0):
    start = torch.full_like(labels[:, :1], start_token)
    decoder_input = torch.cat([start, labels[:, :-1]], dim=1)
    return decoder_input 