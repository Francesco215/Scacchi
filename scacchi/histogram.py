"""Shared concentration-histogram bins and series."""

# A fixed, config-independent grid makes concentration histograms directly
# comparable across runs. The first interval covers concentrations close to
# zero; the remaining intervals are approximately uniform in log2 space.
CONCENTRATION_HISTOGRAM_NUM_BINS = 100
CONCENTRATION_HISTOGRAM_BIN_EDGES: tuple[float, ...] = (
    0.0,
    *(
        2.0
        ** (
            -10.0
            + 20.0 * index / (CONCENTRATION_HISTOGRAM_NUM_BINS - 1)
        )
        for index in range(CONCENTRATION_HISTOGRAM_NUM_BINS)
    ),
)
CONCENTRATION_HISTOGRAM_SERIES = (
    "V_prior",
    "V_posterior",
    "Q_prior",
    "Q_posterior",
)
