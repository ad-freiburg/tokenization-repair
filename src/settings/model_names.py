FWD = "fwd1024"
FWD_ROBUST = "fwd1024_noise0.2"
BWD = "bwd1024"
BWD_ROBUST = "bwd1024_noise0.2"
BIDIR = "labeling_ce"
BIDIR_ROBUST = "labeling_noisy_ce"


def unidirectional_model_name(backward: bool, robust: bool) -> str:
    if backward:
        if robust:
            return BWD_ROBUST
        else:
            return BWD
    else:
        if robust:
            return FWD_ROBUST
        else:
            return FWD


def bidirectional_model_name(robust: bool) -> str:
    return BIDIR_ROBUST if robust else BIDIR
