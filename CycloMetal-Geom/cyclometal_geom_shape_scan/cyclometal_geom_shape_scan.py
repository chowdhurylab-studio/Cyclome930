#!usrbinenv python3

cyclometal_geom_shape_scan.py

Optimizations
-------------
1. One BLAS thread per Python worker to prevent nested oversubscription.
2. Batched 3x3 SVD over vertex permutations instead of one SVD call per
   permutation.
3. Early rejection of permutations that fail the Cartesian cutoff.
4. Early rejection of permutations that cannot improve the best accepted fit
   for the current donor combination.
5. Vectorized metal-donor path blocking with cached atom coordinatesmasks.

The default permutation batch size is 4096. Override it with, for example
    export METAL_SCAN_BATCH_SIZE=8192

Example
-------
python cyclometal_geom_shape_scan.py input.pdb 
    --library metal_shape_library 
    --output-dir results 
    --min-cn 4 --max-rmsd 0.5 --max-angle-rmsd 10 --nproc 4


import __main__
import os

# These must be set before NumPy and v3 are imported. Each ProcessPool worker
# is already a unit of parallelism; nested BLAS threads usually slow this
# workload substantially through oversubscription.
os.environ[OMP_NUM_THREADS] = 1
os.environ[OPENBLAS_NUM_THREADS] = 1
os.environ[MKL_NUM_THREADS] = 1
os.environ[VECLIB_MAXIMUM_THREADS] = 1
os.environ[NUMEXPR_NUM_THREADS] = 1

import math
import sys
from pathlib import Path
import numpy as np

try
    import metal_shape_scan as v3
except ImportError as exc
    raise SystemExit(
        ERROR metal_shape_scan.py must be in the same directory as 
        cyclometal_geom_shape_scan.py or importable through PYTHONPATH.
    ) from exc


BATCH_SIZE = int(os.environ.get(METAL_SCAN_BATCH_SIZE, 4096))
if BATCH_SIZE  1
    raise SystemExit(ERROR METAL_SCAN_BATCH_SIZE must be at least 1.)

# Per-process cache. Each worker receives its own copy after process creation.
_ATOM_CACHE = {}


def batched(iterator, batch_size)
    Yield lists containing at most batch_size items.
    batch = []
    for item in iterator
        batch.append(item)
        if len(batch) = batch_size
            yield batch
            batch = []
    if batch
        yield batch


def get_atom_cache(atoms)
    Cache atom coordinates and same-residue masks for path blocking.
    cache_key = id(atoms)
    cached = _ATOM_CACHE.get(cache_key)
    if cached is not None and cached[atoms_object] is atoms
        return cached

    coordinates = np.asarray(
        [[atom[x], atom[y], atom[z]] for atom in atoms],
        dtype=np.float64,
    )

    residue_keys = [
        (atom[chain], atom[resid], atom[icode])
        for atom in atoms
    ]

    residue_masks = {}
    for index, residue_key in enumerate(residue_keys)
        if residue_key not in residue_masks
            residue_masks[residue_key] = []
        residue_masks[residue_key].append(index)

    boolean_masks = {}
    atom_count = len(atoms)
    for residue_key, indices in residue_masks.items()
        mask = np.zeros(atom_count, dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        boolean_masks[residue_key] = mask

    cached = {
        atoms_object atoms,
        coordinates coordinates,
        residue_masks boolean_masks,
    }
    _ATOM_CACHE.clear()
    _ATOM_CACHE[cache_key] = cached
    return cached


def coordination_paths_are_clear_vectorized(
    predicted_center,
    assigned_indices,
    donors,
    atoms,
)
    Vectorized equivalent of v3.coordination_paths_are_clear().
    atom_cache = get_atom_cache(atoms)
    atom_coordinates = atom_cache[coordinates]
    residue_masks = atom_cache[residue_masks]

    predicted_center = np.asarray(predicted_center, dtype=np.float64)
    relative_coordinates = atom_coordinates - predicted_center[None, ]

    for donor_index in assigned_indices
        donor = donors[donor_index]
        donor_xyz = np.asarray(
            [donor[x], donor[y], donor[z]],
            dtype=np.float64,
        )

        line_vector = donor_xyz - predicted_center
        line_length_squared = float(np.dot(line_vector, line_vector))
        if line_length_squared = 1.0e-12
            continue

        t = (relative_coordinates @ line_vector)  line_length_squared

        donor_residue_key = (
            donor[chain],
            donor[resid],
            donor[icode],
        )
        same_residue = residue_masks.get(donor_residue_key)
        if same_residue is None
            same_residue = np.zeros(len(atom_coordinates), dtype=bool)

        segment_mask = (t  0.0) & (t  0.9) & (~same_residue)
        if not np.any(segment_mask)
            continue

        masked_t = t[segment_mask]
        projections = (
            predicted_center[None, ]
            + masked_t[, None]  line_vector[None, ]
        )
        perpendicular_distances = np.linalg.norm(
            atom_coordinates[segment_mask] - projections,
            axis=1,
        )

        metal_donor_distance = math.sqrt(line_length_squared)
        blocked = (
            (perpendicular_distances  v3.METAL_DONOR_BLOCKING_RADIUS)
            & (
                perpendicular_distances  metal_donor_distance
                 0.10
            )
        )
        if np.any(blocked)
            return False

    return True


def batched_similarity_fit(
    template_points,
    target_batch,
)
    
    Batched equivalent of v3.similarity_fit().

    Parameters
    ----------
    template_points  (CN, 3) ndarray
    target_batch  (B, CN, 3) ndarray

    Returns a dictionary containing one value per batch member. Invalid
    non-positive-scale fits are marked by valid=False.
    
    template_points = np.asarray(template_points, dtype=np.float64)
    target_batch = np.asarray(target_batch, dtype=np.float64)

    template_mean = template_points.mean(axis=0)
    target_means = target_batch.mean(axis=1)

    centered_template = template_points - template_mean
    centered_targets = target_batch - target_means[, None, ]

    covariance = np.einsum(
        ni,bnj-bij,
        centered_template,
        centered_targets,
        optimize=True,
    )

    left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
    rotations = left_vectors @ right_vectors_transposed

    determinants = np.linalg.det(rotations)
    reflected = determinants  0.0
    if np.any(reflected)
        left_vectors = left_vectors.copy()
        left_vectors[reflected, , -1] = -1.0
        rotations = left_vectors @ right_vectors_transposed

    denominator = float(np.sum(centered_template  centered_template))
    if denominator = 0.0
        return None

    rotated_centered_template = np.einsum(
        ni,bij-bnj,
        centered_template,
        rotations,
        optimize=True,
    )

    scales = np.einsum(
        bnj,bnj-b,
        rotated_centered_template,
        centered_targets,
        optimize=True,
    )  denominator

    valid = scales  0.0

    rotated_template_mean = np.einsum(
        i,bij-bj,
        template_mean,
        rotations,
        optimize=True,
    )
    translations = target_means - scales[, None]  rotated_template_mean

    rotated_template_points = np.einsum(
        ni,bij-bnj,
        template_points,
        rotations,
        optimize=True,
    )
    fitted_points = (
        scales[, None, None]  rotated_template_points
        + translations[, None, ]
    )

    residuals = fitted_points - target_batch
    rmsd = np.sqrt(np.mean(np.sum(residuals  residuals, axis=2), axis=1))

    return {
        valid valid,
        rmsd_A rmsd,
        scale scales,
        rotation rotations,
        translation translations,
    }


def fit_one_combination_optimized(
    template_center,
    template_vertices,
    donor_coordinates,
    donors,
    atoms,
    combination,
    prefit_score,
    max_permutations,
    random_seed,
    max_rmsd,
    max_angle_rmsd,
)
    Drop-in optimized replacement for v3.fit_one_combination().
    selected_coordinates = donor_coordinates[list(combination)]
    coordination_number = len(combination)
    best_fit = None

    permutation_iterator = v3.generate_vertex_permutations(
        coordination_number,
        max_permutations,
        random_seed,
    )

    for permutation_batch in batched(permutation_iterator, BATCH_SIZE)
        permutation_array = np.asarray(permutation_batch, dtype=np.int16)
        target_batch = selected_coordinates[permutation_array]

        fits = batched_similarity_fit(template_vertices, target_batch)
        if fits is None
            continue

        candidate_mask = fits[valid] & (fits[rmsd_A]  max_rmsd)
        if best_fit is not None
            candidate_mask &= fits[rmsd_A]  best_fit[rmsd_A]

        candidate_positions = np.flatnonzero(candidate_mask)
        if candidate_positions.size == 0
            continue

        # Evaluate potentially acceptable permutations in ascending RMSD order.
        # Once an accepted fit is found, higher-RMSD candidates cannot replace it.
        candidate_positions = candidate_positions[
            np.argsort(fits[rmsd_A][candidate_positions], kind=stable)
        ]

        for batch_position in candidate_positions
            rmsd_A = float(fits[rmsd_A][batch_position])
            if best_fit is not None and rmsd_A = best_fit[rmsd_A]
                break

            target_vertices = target_batch[batch_position]
            scale = float(fits[scale][batch_position])
            rotation = fits[rotation][batch_position]
            translation = fits[translation][batch_position]

            predicted_center = (
                scale  (template_center @ rotation)
                + translation
            )

            metal_donor_distances = np.linalg.norm(
                target_vertices - predicted_center,
                axis=1,
            )
            mean_radius = float(np.mean(metal_donor_distances))
            if mean_radius = 0.0
                continue

            angle_rmsd = v3.calculate_angle_rmsd(
                template_vertices,
                template_center,
                target_vertices,
                predicted_center,
            )
            if (
                not np.isfinite(angle_rmsd)
                or angle_rmsd = max_angle_rmsd
            )
                continue

            permutation = permutation_batch[batch_position]
            assigned_indices = tuple(
                combination[position]
                for position in permutation
            )

            if not coordination_paths_are_clear_vectorized(
                predicted_center=predicted_center,
                assigned_indices=assigned_indices,
                donors=donors,
                atoms=atoms,
            )
                continue

            best_fit = {
                rmsd_A rmsd_A,
                normalized_rmsd rmsd_A  mean_radius,
                angle_rmsd_deg float(angle_rmsd),
                mean_radius_A mean_radius,
                scale scale,
                predicted_center predicted_center,
                assigned_indices assigned_indices,
                prefit_score prefit_score,
            }

            # This is the lowest-RMSD candidate in the current batch. Any later
            # candidate in this batch cannot improve it. Future batches can still
            # contain a lower-RMSD fit and are therefore processed.
            break

    return best_fit


def install_optimizations():
    """Install optimized kernels into v3 before workers are created."""
    v3.coordination_paths_are_clear = coordination_paths_are_clear_vectorized
    v3.fit_one_combination = fit_one_combination_optimized


if __name__ == "__main__":
    install_optimizations()
    print(
        f"[cyclometal_geom_shape_scan] permutation batch size={BATCH_SIZE}; "
        "BLAS threads per worker=1",
        file=sys.stderr,
    )
    v3.main()
