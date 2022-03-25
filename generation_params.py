P = "TOPP"
K = "TOPK"
SAMPLE = "SAMPLE"
BEAM_SIZE = "BEAM_SIZE"
TEMPERATURE = "TEMPERATURE"
REPETITION_PENALTY = "REPETITION_PENALTY"
EARLY_STOPPING = "EARLY_STOPPING"
LENGTH_PENALTY = "LENGTH_PENALTY"
INFERENCE_BATCH_SIZE = "INFERENCE_BATCH_SIZE"
NAME = "NAME"
default_gen_params = {
    NAME: "default",
    P: 1,  # default
    K: 50,  # default
    SAMPLE: False,  # default,
    BEAM_SIZE: 2,
    TEMPERATURE: 1,  # default
    REPETITION_PENALTY: 2.5,
    EARLY_STOPPING: True,
    LENGTH_PENALTY: 1.0,
    INFERENCE_BATCH_SIZE: 1
}


def get_grid_search_params():
    all_params = []

    inference_batch_size = 16

    beam_sizes = [2, 4, 8]
    repetition_penalties = [1.0, 1.5, 2.5, 3.0]

    i = 0
    for bs in beam_sizes:
        for rp in repetition_penalties:
            # high beam sizes don't have repetitions
            if bs > 4 and rp > 2:
                continue

            all_params.append(
                {
                    NAME: f"config{i}",
                    P: 1,  # default
                    K: 50,  # default
                    SAMPLE: False,  # default,
                    BEAM_SIZE: bs,
                    TEMPERATURE: 1,  # default
                    REPETITION_PENALTY: rp,
                    EARLY_STOPPING: False,
                    LENGTH_PENALTY: 1.0,
                    INFERENCE_BATCH_SIZE: inference_batch_size
                }
            )

            i += 1

    # todo these don't do well, consider removing them
    ps = [1, 0.95]
    ks = [60, 70]
    ts = [1, 0.95]
    # top-k top-p works best without repitition penalty
    for rp in [1.0]:
        for p in ps:
            for k in ks:
                for t in ts:
                    all_params.append(
                        {
                            NAME: f"config{i}",
                            P: p,
                            K: k,
                            SAMPLE: True,
                            BEAM_SIZE: 1,
                            TEMPERATURE: t,
                            REPETITION_PENALTY: rp,
                            EARLY_STOPPING: False,
                            LENGTH_PENALTY: 1.0,
                            INFERENCE_BATCH_SIZE: inference_batch_size
                        }
                    )

    return all_params

if __name__ == "__main__":
    ps = get_grid_search_params()
    for i, p in enumerate(ps):
        print(f"{i} - {p}")