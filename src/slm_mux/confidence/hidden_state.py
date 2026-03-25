"""
Hidden-state probe confidence (SAPLMA method).

Based on "The Internal State of an LLM Knows When It's Lying"
(Azaria & Mitchell, Findings of EMNLP 2023, arXiv:2304.13734).

Method:
1. Run the LLM on a question + answer
2. Extract the hidden state at the last token from an intermediate layer
3. Use a pre-trained MLP probe to predict P(answer is correct)
4. Use this probability as the confidence score

This requires:
- Access to model internals (HuggingFace transformers, not API-based)
- A trained probe checkpoint (or the ability to train one)

The probe can be trained with `HiddenStateProbe.train_probe()` on a
validation set of (hidden_state, is_correct) pairs.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from .base import ConfidenceEvaluator, ConfidenceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP Probe (pure PyTorch)
# ---------------------------------------------------------------------------

class MLPProbe:
    """
    3-layer MLP probe: input_dim -> 256 -> 128 -> 64 -> 1 (sigmoid).

    Follows the SAPLMA paper architecture exactly.
    Can be trained and saved/loaded as a checkpoint.
    """

    def __init__(self, input_dim: int = 4096):
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.device = torch.device("cpu")

    def to(self, device):
        import torch
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def predict(self, hidden_states) -> List[float]:
        """
        Predict P(correct) for each hidden state vector.

        Args:
            hidden_states: Tensor of shape [N, input_dim] or list of lists.

        Returns:
            List of probabilities.
        """
        import torch

        self.model.eval()
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
        hidden_states = hidden_states.to(self.device)

        with torch.no_grad():
            probs = self.model(hidden_states).squeeze(-1)
        return probs.cpu().tolist()

    def train_probe(
        self,
        hidden_states,
        labels,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the probe on (hidden_state, is_correct) pairs.

        Args:
            hidden_states: Tensor [N, input_dim] or list of lists.
            labels: List/Tensor of 0/1 labels (1 = correct).
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            verbose: Print training progress.

        Returns:
            Dict with training stats.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        hidden_states = hidden_states.to(self.device)
        labels = labels.to(self.device)

        dataset = TensorDataset(hidden_states, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x).squeeze(-1)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                predicted = (pred > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            avg_loss = total_loss / total if total > 0 else 0.0
            acc = correct / total if total > 0 else 0.0
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})

            if verbose:
                logger.info(f"Probe training epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.4f}")

        return {"history": history}

    def save(self, path: str):
        """Save probe checkpoint."""
        import torch
        state = {
            "input_dim": self.input_dim,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Probe saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MLPProbe":
        """Load probe from checkpoint."""
        import torch
        # Resolve "auto" to actual device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(path, map_location=device, weights_only=True)
        probe = cls(input_dim=state["input_dim"])
        probe.model.load_state_dict(state["model_state_dict"])
        probe.to(device)
        probe.model.eval()
        logger.info(f"Probe loaded from {path} (input_dim={state['input_dim']})")
        return probe


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model_name: str,
    texts: List[str],
    layer: Optional[int] = None,
    device: str = "auto",
    max_length: int = 2048,
) -> Tuple[List[List[float]], int]:
    """
    Extract hidden states at the last token from a specified layer.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct").
        texts: List of input texts (question + answer concatenated).
        layer: Which layer to extract from. None = middle layer.
        device: Torch device.
        max_length: Max token length.

    Returns:
        (hidden_states as list of lists, layer_used)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    logger.info(f"Loading model {model_name} for hidden state extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
        output_hidden_states=True,
    )
    model.to(dev)
    model.eval()

    # Determine layer
    num_layers = model.config.num_hidden_layers
    if layer is None:
        layer = num_layers // 2  # middle layer (SAPLMA finding)
    layer = min(layer, num_layers)

    logger.info(f"Extracting from layer {layer}/{num_layers} on {dev}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_states_all = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(dev)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is the embedding layer, index 1..num_layers are transformer layers
        layer_hidden = outputs.hidden_states[layer]  # [1, seq_len, hidden_dim]

        # Take the last token's hidden state
        last_token_hidden = layer_hidden[0, -1, :].cpu().tolist()
        hidden_states_all.append(last_token_hidden)

    return hidden_states_all, layer


def extract_all_layers(
    model_name: str,
    texts: List[str],
    device: str = "auto",
    max_length: int = 2048,
    batch_size: int = 4,
    target_layers: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[List[float]]], int]:
    """
    Extract last-token hidden states from selected (or all) layers.

    Args:
        target_layers: If provided, only extract these layer indices.
            If None, extract ALL layers (warning: memory-intensive for large datasets).

    Returns:
        (layer_to_hidden_states, num_layers)
        layer_to_hidden_states: {layer_idx: [[hidden_dim floats], ...]}
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    logger.info(f"Loading {model_name} for all-layer extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
        output_hidden_states=True,
    )
    model.to(dev)
    model.eval()

    num_layers = model.config.num_hidden_layers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine which layers to extract
    if target_layers is not None:
        layers_to_extract = [l for l in target_layers if 0 <= l <= num_layers]
    else:
        layers_to_extract = list(range(num_layers + 1))

    all_layers: Dict[int, List[List[float]]] = {i: [] for i in layers_to_extract}

    logger.info(f"Extracting {len(layers_to_extract)} layers from {len(texts)} texts")
    for text in tqdm(texts, desc="Extracting hidden states"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False,
        ).to(dev)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in layers_to_extract:
            last_token = outputs.hidden_states[layer_idx][0, -1, :].cpu().tolist()
            all_layers[layer_idx].append(last_token)

    first_key = layers_to_extract[0] if layers_to_extract else 0
    dim = len(all_layers[first_key][0]) if all_layers[first_key] else 0
    logger.info(f"Extracted: {len(layers_to_extract)} layers, {len(texts)} texts, dim={dim}")
    return all_layers, num_layers


# ---------------------------------------------------------------------------
# Confidence evaluator
# ---------------------------------------------------------------------------

class HiddenStateConfidence(ConfidenceEvaluator):
    """
    Hidden-state probe confidence (SAPLMA, EMNLP 2023).

    Uses a pre-trained MLP probe on the model's intermediate hidden states
    to predict P(answer is correct) as the confidence score.

    Usage:
        # With a pre-trained probe:
        conf = HiddenStateConfidence(
            probe_path="path/to/probe.pt",
            llm_model_name="Qwen/Qwen2.5-7B-Instruct",
            layer=16,
        )

        # Or train a new probe:
        conf = HiddenStateConfidence(llm_model_name="Qwen/Qwen2.5-7B-Instruct")
        conf.train_from_data(questions, answers, is_correct_labels)
    """

    def __init__(
        self,
        probe_path: Optional[str] = None,
        llm_model_name: str = "",
        layer: Optional[int] = None,
        device: str = "auto",
        max_length: int = 2048,
    ):
        """
        Args:
            probe_path: Path to a saved MLPProbe checkpoint. If None, probe
                must be trained before use.
            llm_model_name: HuggingFace model name for hidden state extraction.
            layer: Which transformer layer to extract from. None = middle layer.
            device: Torch device for both model and probe.
            max_length: Max token length for the LLM.
        """
        self._llm_model_name = llm_model_name
        self._layer = layer
        self._device = device
        self._max_length = max_length
        self._probe: Optional[MLPProbe] = None

        if probe_path and os.path.exists(probe_path):
            self._probe = MLPProbe.load(probe_path, device=device)

    @property
    def name(self) -> str:
        return "hidden_state"

    @property
    def requires_external_model(self) -> bool:
        return True

    @property
    def requires_logprobs(self) -> bool:
        return False

    def train_from_data(
        self,
        questions: List[str],
        answers: List[str],
        is_correct: List[bool],
        epochs: int = 5,
        lr: float = 1e-3,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the probe from (question, answer, is_correct) triples.

        1. Concatenate each question + answer
        2. Extract hidden states from the LLM
        3. Train the MLP probe to predict is_correct

        Args:
            questions: List of questions.
            answers: List of model answers.
            is_correct: List of correctness labels.
            epochs: Training epochs.
            lr: Learning rate.
            save_path: Optional path to save the trained probe.

        Returns:
            Training stats dict.
        """
        import torch

        if not self._llm_model_name:
            raise ValueError("llm_model_name must be set to train the probe")

        # Build input texts
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]

        # Extract hidden states
        logger.info(f"Extracting hidden states from {len(texts)} examples...")
        hidden_states, layer_used = extract_hidden_states(
            model_name=self._llm_model_name,
            texts=texts,
            layer=self._layer,
            device=self._device,
            max_length=self._max_length,
        )
        self._layer = layer_used

        # Create probe
        input_dim = len(hidden_states[0])
        self._probe = MLPProbe(input_dim=input_dim)
        if self._device != "cpu":
            self._probe.to(self._device)

        # Train
        labels = [1.0 if c else 0.0 for c in is_correct]
        hs_tensor = torch.tensor(hidden_states, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        stats = self._probe.train_probe(
            hs_tensor, labels_tensor,
            epochs=epochs, lr=lr, verbose=True,
        )

        if save_path:
            self._probe.save(save_path)

        return stats

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence using hidden state probe.

        For each response:
        1. Concatenate question + response
        2. Extract hidden state from intermediate layer
        3. Run through the trained probe to get P(correct)

        The response with highest P(correct) is selected.
        Confidence score = max P(correct) across responses.
        """
        if self._probe is None:
            logger.warning("Hidden state probe not loaded/trained. Falling back to consistency.")
            from .consistency import ConsistencyConfidence
            return ConsistencyConfidence().evaluate(responses, extracted_answers, question, logprobs)

        if not responses or not question:
            return ConfidenceResult(
                score=0.0, selected_answer="",
                metadata={"method": "hidden_state", "error": "no responses or question"},
            )

        # Build texts for extraction
        texts = [f"Question: {question}\nAnswer: {r}" for r in responses]

        try:
            hidden_states, layer_used = extract_hidden_states(
                model_name=self._llm_model_name,
                texts=texts,
                layer=self._layer,
                device=self._device,
                max_length=self._max_length,
            )
        except Exception as e:
            logger.error(f"Hidden state extraction failed: {e}")
            return ConfidenceResult(
                score=0.0,
                selected_answer=extracted_answers[0] if extracted_answers else "",
                metadata={"method": "hidden_state", "error": str(e)},
            )

        # Get probe predictions
        probs = self._probe.predict(hidden_states)

        # Select response with highest predicted probability
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        confidence = probs[best_idx]

        selected = extracted_answers[best_idx] if best_idx < len(extracted_answers) else ""

        return ConfidenceResult(
            score=confidence,
            selected_answer=selected,
            metadata={
                "method": "hidden_state",
                "model": self._llm_model_name,
                "layer": self._layer,
                "probe_probs": probs,
                "best_index": best_idx,
                "num_responses": len(responses),
            },
        )
