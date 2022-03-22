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
    BEAM_SIZE: 1,
    TEMPERATURE: 1,  # default
    REPETITION_PENALTY: 1,
    EARLY_STOPPING: False,
    LENGTH_PENALTY: 1.0,
    INFERENCE_BATCH_SIZE: 1
}


def get_grid_search_params():
    all_params = []

    inference_batch_size = 8

    beam_sizes = [16, 4, 8]
    repetition_penalties = [1.2, 1.7, 2.5, 3.0]

    i = 0
    for bs in beam_sizes:
        for rp in repetition_penalties:
            all_params.append(
                {
                    NAME: "config{i}",
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

    ps = [1.0, 0.95]
    ks = [60, 80]
    ts = [0.9, 0.95]

    for rp in repetition_penalties:
        for p in ps:
            for k in ks:
                for t in ts:
                    all_params.append(
                        {
                            NAME: "config{i}",
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

                    i += 1
