from typing import Optional

from src.corrector.load.model import load_bidirectional_model
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.corrector.labeling.labeling_corrector import LabelingCorrector
from src.corrector.threshold_holder import FittingMethod, ThresholdHolder
from src.benchmark.benchmark import get_benchmark_name


def load_labeling_corrector(robust: bool,
                            typos: bool,
                            p: float,
                            model: Optional[BidirectionalLabelingEstimator] = None) -> LabelingCorrector:
    if model is None:
        model = load_bidirectional_model(robust)
    model_name = model.specification.name
    holder = ThresholdHolder(FittingMethod.LABELING)
    threshold_benchmark_name = get_benchmark_name(0.1 if typos else 0, p)
    insertion_threshold, deletion_threshold = holder.get_thresholds(model_name, noise_type=threshold_benchmark_name)
    corrector = LabelingCorrector(model_name,
                                  insertion_threshold,
                                  deletion_threshold,
                                  model=model)
    return corrector
