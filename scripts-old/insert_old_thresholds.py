from project import src
from src.corrector.threshold_holder import ThresholdHolder


if __name__ == "__main__":
    holder = ThresholdHolder()
    
    # combined
    holder.set_insertion_threshold(fwd_model_name="fwd_old", bwd_model_name="bwd_old", noise_type="none", threshold=0.999962)
    holder.set_deletion_threshold(fwd_model_name="fwd_old", bwd_model_name="bwd_old", noise_type="none", threshold=0.374470)
    
    # sigmoid
    holder.set_insertion_threshold(model_name="sigmoid_old", noise_type="none", threshold=0.837327)
    holder.set_deletion_threshold(model_name="sigmoid_old", noise_type="none", threshold=0.994371)
