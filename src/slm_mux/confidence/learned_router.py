import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import ConfidenceEvaluator, ConfidenceResult
from .consistency import ConsistencyConfidence

logger = logging.getLogger(__name__)


class LearnedRouterConfidence(ConfidenceEvaluator):
    """
    Learned router confidence (RouteLLM-style).

    Uses a pre-trained classifier that predicts per-model accuracy
    given the question. This is NOT a consistency-based method;
    instead, it uses a trained model to directly predict which
    SLM will answer correctly.

    The classifier can be trained on validation set data where we know
    which models answered correctly.

    Supported classifier types:
        - "sklearn": A scikit-learn classifier loaded via joblib/pickle.
          Input features: TF-IDF or embedding of the question.
          Output: probability of correct answer.
        - "torch": A PyTorch neural network classifier.
          TODO: Implement torch-based classifier loading.

    If no classifier is loaded, falls back to ConsistencyConfidence.
    """

    def __init__(
        self,
        classifier_path: str = "",
        model_index: Optional[int] = None,
    ):
        """
        Args:
            classifier_path: Path to a saved classifier file.
                Supported formats:
                - .joblib / .pkl: scikit-learn model loaded via joblib
                - .pt / .pth: PyTorch model (TODO)
                - .json: Config file pointing to model artifacts
            model_index: If the classifier predicts for multiple models,
                this specifies which model's prediction to use.
        """
        self._classifier: Any = None
        self._vectorizer: Any = None
        self._model_index = model_index
        self._classifier_path = classifier_path
        self._fallback = ConsistencyConfidence()

        if classifier_path:
            self._load_classifier(classifier_path)

    @property
    def name(self) -> str:
        return "learned_router"

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence using the learned router.

        If a classifier is loaded, predicts the confidence score for the
        given question. The score represents the predicted probability that
        this model will answer the question correctly.

        The selected answer is chosen by consistency voting among the
        extracted answers (the router only provides the confidence score,
        not the answer itself).

        If no classifier is loaded, falls back to ConsistencyConfidence entirely.

        Args:
            responses: Raw model outputs (k samples).
            extracted_answers: Post-extraction answers (k samples).
            question: Original question text (required for classifier prediction).
            logprobs: Not used by this method.

        Returns:
            ConfidenceResult with score from the learned classifier.
        """
        # Always use consistency to pick the answer
        consistency_result = self._fallback.evaluate(
            responses, extracted_answers, question, logprobs
        )

        if self._classifier is None:
            # No classifier loaded: fall back to consistency entirely
            consistency_result.metadata["method"] = "learned_router"
            consistency_result.metadata["fallback"] = True
            consistency_result.metadata["reason"] = "no classifier loaded"
            return consistency_result

        if question is None:
            logger.warning(
                "LearnedRouterConfidence: question is None. "
                "Cannot predict without a question. Falling back to consistency."
            )
            consistency_result.metadata["method"] = "learned_router"
            consistency_result.metadata["fallback"] = True
            consistency_result.metadata["reason"] = "question is None"
            return consistency_result

        # Predict confidence using the classifier
        predicted_score = self._predict(question)

        if predicted_score is None:
            consistency_result.metadata["method"] = "learned_router"
            consistency_result.metadata["fallback"] = True
            consistency_result.metadata["reason"] = "prediction failed"
            return consistency_result

        return ConfidenceResult(
            score=predicted_score,
            selected_answer=consistency_result.selected_answer,
            metadata={
                "method": "learned_router",
                "fallback": False,
                "predicted_score": predicted_score,
                "classifier_path": self._classifier_path,
                "consistency_score": consistency_result.score,
                "vote_counts": consistency_result.metadata.get("vote_counts", {}),
            },
        )

    def _load_classifier(self, path: str) -> None:
        """
        Load a pre-trained classifier from disk.

        Supports:
            - .joblib / .pkl: scikit-learn pipeline (vectorizer + classifier)
            - .json: Config file with paths to model artifacts
        """
        if not os.path.exists(path):
            logger.error("Classifier file not found: %s", path)
            return

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext in (".joblib", ".pkl"):
                self._load_sklearn_classifier(path)
            elif ext == ".json":
                self._load_from_config(path)
            elif ext in (".pt", ".pth"):
                # TODO: Implement PyTorch classifier loading
                logger.warning(
                    "PyTorch classifier loading not yet implemented. "
                    "File: %s",
                    path,
                )
            else:
                logger.error(
                    "Unsupported classifier format: %s (file: %s)", ext, path
                )
        except Exception as e:
            logger.error("Failed to load classifier from %s: %s", path, e)

    def _load_sklearn_classifier(self, path: str) -> None:
        """Load a scikit-learn classifier (and optional vectorizer) via joblib."""
        try:
            import joblib
        except ImportError:
            logger.error(
                "joblib is required to load sklearn classifiers. "
                "Install it with: pip install joblib"
            )
            return

        loaded = joblib.load(path)

        # The saved object can be either:
        # 1. A dict with "classifier" and "vectorizer" keys
        # 2. A sklearn Pipeline (which includes vectorizer)
        # 3. Just the classifier (vectorizer must be loaded separately)
        if isinstance(loaded, dict):
            self._classifier = loaded.get("classifier")
            self._vectorizer = loaded.get("vectorizer")
            logger.info(
                "Loaded sklearn classifier from %s (dict format, "
                "vectorizer=%s)",
                path,
                "present" if self._vectorizer else "absent",
            )
        else:
            # Assume it's a pipeline or standalone classifier
            self._classifier = loaded
            logger.info("Loaded sklearn classifier from %s", path)

    def _load_from_config(self, path: str) -> None:
        """Load classifier from a JSON config file specifying artifact paths."""
        with open(path, "r") as f:
            config = json.load(f)

        base_dir = os.path.dirname(os.path.abspath(path))

        classifier_path = config.get("classifier_path", "")
        if classifier_path and not os.path.isabs(classifier_path):
            classifier_path = os.path.join(base_dir, classifier_path)

        if classifier_path:
            self._load_sklearn_classifier(classifier_path)

        vectorizer_path = config.get("vectorizer_path", "")
        if vectorizer_path and not os.path.isabs(vectorizer_path):
            vectorizer_path = os.path.join(base_dir, vectorizer_path)

        if vectorizer_path and os.path.exists(vectorizer_path):
            try:
                import joblib

                self._vectorizer = joblib.load(vectorizer_path)
                logger.info("Loaded vectorizer from %s", vectorizer_path)
            except Exception as e:
                logger.error("Failed to load vectorizer from %s: %s", vectorizer_path, e)

        # Store additional config
        if self._model_index is None:
            self._model_index = config.get("model_index")

    def _predict(self, question: str) -> Optional[float]:
        """
        Predict confidence score for a question using the loaded classifier.

        Returns:
            Predicted probability in [0, 1], or None if prediction fails.
        """
        try:
            # Vectorize the question if a vectorizer is available
            if self._vectorizer is not None:
                features = self._vectorizer.transform([question])
            else:
                # If no vectorizer, assume the classifier handles raw text
                # (e.g., it's a Pipeline that includes vectorization)
                features = [question]

            # Get probability prediction
            if hasattr(self._classifier, "predict_proba"):
                proba = self._classifier.predict_proba(features)
                # proba shape: (1, num_classes) for binary or multi-class
                # For binary: proba[0][1] = probability of class 1 (correct)
                if self._model_index is not None and proba.shape[1] > self._model_index:
                    score = float(proba[0][self._model_index])
                else:
                    # Binary classification: take probability of positive class
                    score = float(proba[0][-1])
            elif hasattr(self._classifier, "decision_function"):
                # SVM-style: convert decision function to pseudo-probability
                import math

                decision = float(self._classifier.decision_function(features)[0])
                score = 1.0 / (1.0 + math.exp(-decision))  # sigmoid
            elif hasattr(self._classifier, "predict"):
                # Last resort: use raw prediction as score
                score = float(self._classifier.predict(features)[0])
            else:
                logger.error("Classifier has no predict_proba, decision_function, or predict")
                return None

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error("Classifier prediction failed: %s", e)
            return None
