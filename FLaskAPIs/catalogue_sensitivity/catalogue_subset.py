"""Deterministic nested, diversity-oriented catalogue subsets; no outcome inputs."""
import hashlib
import numpy as np

INITIAL = np.array([.6, .5, .7, .4, 85., 80., 90.])
SCALE = np.array([.9, .8, .8, .8, 75., 75., 75.])

def select_catalogue(catalogue, columns, fraction, seed):
    if fraction not in (0.25, 0.5, 1.0):
        raise ValueError('SCOPE_CATALOGUE_FRACTION must be 0.25, 0.5 or 1.0')
    a = catalogue.loc[:, columns].to_numpy(dtype=float)
    count = len(a)
    target = max(1, int(np.ceil(count * fraction)))
    # Full setting keeps original row ordering, preserving original argmin ties.
    if fraction == 1:
        result = catalogue.copy()
    else:
        # Canonical ordering makes membership independent of CSV row ordering.
        canonical = np.lexsort(tuple(a[:, j] for j in range(a.shape[1]-1, -1, -1)))
        rng = np.random.default_rng(seed)
        pool = canonical[rng.permutation(count)]
        points = a[pool] / SCALE
        initial_matches = np.flatnonzero(np.all(np.isclose(a[pool], INITIAL, atol=1e-10, rtol=0), axis=1))
        chosen = []
        taken = np.zeros(count, dtype=bool)
        # Keep the initial configuration when present in the catalogue. Otherwise
        # the unchanged service still adds the active configuration at each call.
        if len(initial_matches):
            chosen.append(int(initial_matches[0])); taken[chosen[-1]] = True
        # A seed-dependent starting point provides alternative subset realizations.
        if len(chosen) < target:
            j = int(np.flatnonzero(~taken)[0]); chosen.append(j); taken[j] = True
        distances = np.full(count, np.inf)
        for j in chosen:
            distances = np.minimum(distances, np.abs(points-points[j]).sum(axis=1))
        while len(chosen) < target:
            distances[taken] = -np.inf
            j = int(np.argmax(distances))
            chosen.append(j); taken[j] = True
            distances = np.minimum(distances, np.abs(points-points[j]).sum(axis=1))
        # Preserve original relative order for the service's existing tie behavior.
        indices = np.sort(pool[chosen])
        result = catalogue.iloc[indices].reset_index(drop=True)
    payload = result.loc[:, columns].to_csv(index=False, float_format='%.10f').encode()
    metadata = dict(catalogue_fraction=fraction, catalogue_subset_seed=int(seed),
                    full_catalogue_count=count, selected_catalogue_count=len(result),
                    catalogue_sha256=hashlib.sha256(payload).hexdigest())
    return result, metadata
