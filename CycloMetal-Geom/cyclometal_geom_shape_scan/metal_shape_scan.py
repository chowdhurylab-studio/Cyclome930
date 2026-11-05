#!/usr/bin/env python3

"""
metal_shape_scan.py

Integrated workflow:
    1. Read an input protein PDB.
    2. Select possible amino-acid metal-donor atoms.
    3. Exclude disulfide-bonded cysteine SG atoms.
    4. Read geometry templates assigned to each metal.
    5. Search local donor combinations matching each template CN.
    6. Fit each ideal geometry by rotation, translation, and uniform scaling.
    7. Rank shape matches by RMSD for every metal.
    8. Report predicted metal coordinates and coordinating donor residues.

Example:
    python metal_shape_scan.py test/1f2s_clean.pdb

Optional:
    python metal_shape_scan.py test/1f2s_clean.pdb \
        --library metal_shape_library \
        --prefit-keep 50 \
        --top-hits 5 \
        --max-distance 8.0 \
        --high-confidence 0.5

Outputs:
    <pdb_stem>_possible_donors.csv
    <pdb_stem>_possible_donors.pdb
    <pdb_stem>_metal_shape_rankings.csv
    <pdb_stem>_metal_shape_rankings.txt

Important:
    - This is a shape-only search.
    - v3 searches CN >= 5 by default and writes only accepted hits with
      Cartesian RMSD < 0.5 A and angular RMSD < 10 degrees by default.
    - Shape searches can run across CPU worker processes with --nproc.
    - This implementation is CPU-oriented; Slurm arrays should parallelize PDBs.
    - The true metal position is not used.
    - The predicted metal position is inferred from the fitted template.
    - RMSD <= 0.5 A is labeled HIGH confidence, but all retained fits
      are written and ranked.
    - The same geometry code is searched once and reused for every metal
      that contains that shape.
    - For very high-CN templates, exact vertex permutation searches can
      become expensive. The script limits the number of tested vertex
      permutations using a deterministic sampling strategy.
"""

import argparse
import os
import csv
import heapq
import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


##############################################################################
# DONOR-SELECTION SETTINGS
##############################################################################

# Maximum SG-SG distance treated as a disulfide bond.
SS_CUTOFF = 2.30
# Reject a fit if an atom lies directly between the predicted metal
# center and a coordinating donor atom.
METAL_DONOR_BLOCKING_RADIUS = 0.5

# Side-chain atoms allowed as potential protein metal donors.
SIDECHAIN_DONORS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "ASN": ["OD1"],
    "GLN": ["OE1"],
    "HIS": ["ND1", "NE2"],
    "CYS": ["SG"],
    "MET": ["SD"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TYR": ["OH"],
}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


##############################################################################
# METAL FOLDER DEFINITIONS
##############################################################################

LANTHANIDES = {
    "cerium",
    "dysprosium",
    "europium",
    "gadolinium",
    "holmium",
    "lanthanum",
    "lutetium",
    "praseodymium",
    "samarium",
    "terbium",
    "ytterbium",
}

METAL_ORDER = [
    "cobalt",
    "copper",
    "manganese",
    "nickel",
    "cerium",
    "dysprosium",
    "europium",
    "gadolinium",
    "holmium",
    "lanthanum",
    "lutetium",
    "praseodymium",
    "samarium",
    "terbium",
    "ytterbium",
]


##############################################################################
# COMMAND-LINE ARGUMENTS
##############################################################################

def parse_arguments():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Select possible protein metal donors from a PDB and scan "
            "metal-specific geometry templates."
        )
    )

    parser.add_argument(
        "pdb_file",
        type=Path,
        help="Input protein PDB file, for example test/1f2s_clean.pdb",
    )

    parser.add_argument(
        "--library",
        type=Path,
        default=script_dir,
        help=(
            "Metal shape library directory. Default: directory containing "
            "this script."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory. Default: current working directory.",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=8.0,
        help=(
            "Maximum allowed donor-donor distance in a candidate local "
            "combination. Default: 8.0 A."
        ),
    )

    parser.add_argument(
        "--prefit-keep",
        type=int,
        default=10,
        help=(
            "Number of best pairwise-signature combinations passed to the "
            "more expensive template fit. Default: 10."
        ),
    )

    parser.add_argument(
        "--top-hits",
        type=int,
        default=5,
        help=(
            "Number of ranked donor combinations retained per shape. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.5,
        help="High-confidence RMSD threshold in A. Default: 0.5.",
    )

    parser.add_argument(
        "--medium-confidence",
        type=float,
        default=1.0,
        help="Medium-confidence RMSD threshold in A. Default: 1.0.",
    )

    parser.add_argument(
        "--max-permutations",
        type=int,
        default=50000,
        help=(
            "Maximum vertex assignments tested for one donor combination. "
            "Default: 50000. Relevant primarily for CN 9 geometries."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="Random seed for deterministic permutation sampling.",
    )

    parser.add_argument(
        "--min-cn",
        type=int,
        default=5,
        help="Minimum coordination number searched. Default: 5 (CN > 4).",
    )

    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=0.5,
        help="Retain only fits with Cartesian RMSD strictly below this value in A. Default: 0.5.",
    )

    parser.add_argument(
        "--max-angle-rmsd",
        type=float,
        default=10.0,
        help="Retain only fits with angular RMSD strictly below this value in degrees. Default: 10.",
    )

    parser.add_argument(
        "--prefit-cutoff",
        type=float,
        default=0.15,
        help=(
            "Discard donor combinations whose scale-normalized pairwise-distance "
            "signature RMSD exceeds this value. Default: 0.15."
        ),
    )

    parser.add_argument(
        "--nproc",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of CPU worker processes used across shapes. Default: min(4, available CPUs).",
    )

    return parser.parse_args()


##############################################################################
# READ INPUT PDB
##############################################################################

def read_pdb(filename):
    atoms = []
    ssbond_residues = set()

    with open(filename, "r") as handle:
        for line in handle:

            # Explicit SSBOND records.
            if line.startswith("SSBOND"):
                chain1 = line[15].strip()
                resid1 = line[17:21].strip()
                chain2 = line[29].strip()
                resid2 = line[31:35].strip()

                ssbond_residues.add((chain1, resid1))
                ssbond_residues.add((chain2, resid2))
                continue

            if not line.startswith(("ATOM", "HETATM")):
                continue

            # Keep only blank or primary A alternate conformations.
            altloc = line[16].strip()

            if altloc not in ("", "A"):
                continue

            try:
                atom = {
                    "line": line.rstrip("\n"),
                    "record": line[0:6].strip(),
                    "serial": line[6:11].strip(),
                    "name": line[12:16].strip().upper(),
                    "resname": line[17:20].strip().upper(),
                    "chain": line[21].strip(),
                    "resid": line[22:26].strip(),
                    "icode": line[26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            except (ValueError, IndexError):
                continue

            atoms.append(atom)

    return atoms, ssbond_residues


##############################################################################
# DONOR SELECTION
##############################################################################

def atom_distance(atom_a, atom_b):
    dx = atom_a["x"] - atom_b["x"]
    dy = atom_a["y"] - atom_b["y"]
    dz = atom_a["z"] - atom_b["z"]

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def find_disulfides(atoms, ssbond_residues):
    cys_sg = [
        atom
        for atom in atoms
        if atom["resname"] == "CYS" and atom["name"] == "SG"
    ]

    excluded_sg = set()
    disulfide_pairs = []

    # Exclude cysteine SG atoms identified through SSBOND records.
    for atom in cys_sg:
        residue_key = (
            atom["chain"],
            atom["resid"],
        )

        if residue_key in ssbond_residues:
            excluded_sg.add(
                (
                    atom["chain"],
                    atom["resid"],
                    atom["icode"],
                    atom["name"],
                )
            )

    # Also detect disulfides geometrically.
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            atom_a = cys_sg[i]
            atom_b = cys_sg[j]

            distance = atom_distance(atom_a, atom_b)

            if distance <= SS_CUTOFF:
                key_a = (
                    atom_a["chain"],
                    atom_a["resid"],
                    atom_a["icode"],
                    atom_a["name"],
                )

                key_b = (
                    atom_b["chain"],
                    atom_b["resid"],
                    atom_b["icode"],
                    atom_b["name"],
                )

                excluded_sg.add(key_a)
                excluded_sg.add(key_b)

                disulfide_pairs.append(
                    (atom_a, atom_b, distance)
                )

    return excluded_sg, disulfide_pairs


def select_donors(atoms, excluded_sg):
    donors = []

    for original_atom in atoms:
        resname = original_atom["resname"]
        atom_name = original_atom["name"]

        # Protein amino acids only.
        if resname not in STANDARD_AA:
            continue

        atom = dict(original_atom)

        # Backbone carbonyl oxygen.
        if atom_name in ("O", "OXT"):
            atom["atom"] = atom_name
            atom["donor_type"] = "backbone_O"
            donors.append(atom)
            continue

        # Allowed side-chain donor atoms.
        if resname not in SIDECHAIN_DONORS:
            continue

        if atom_name not in SIDECHAIN_DONORS[resname]:
            continue

        # Exclude disulfide-bonded cysteine SG.
        if resname == "CYS" and atom_name == "SG":
            key = (
                atom["chain"],
                atom["resid"],
                atom["icode"],
                atom_name,
            )

            if key in excluded_sg:
                continue

        atom["atom"] = atom_name
        atom["donor_type"] = "sidechain"
        donors.append(atom)

    return donors


##############################################################################
# WRITE DONOR FILES
##############################################################################

def write_donor_outputs(donors, csv_file, pdb_file):
    with open(csv_file, "w", newline="") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "serial",
            "chain",
            "resid",
            "icode",
            "resname",
            "atom",
            "donor_type",
            "x",
            "y",
            "z",
        ])

        for atom in donors:
            writer.writerow([
                atom["serial"],
                atom["chain"],
                atom["resid"],
                atom["icode"],
                atom["resname"],
                atom["atom"],
                atom["donor_type"],
                f'{atom["x"]:.3f}',
                f'{atom["y"]:.3f}',
                f'{atom["z"]:.3f}',
            ])

    with open(pdb_file, "w") as handle:
        for atom in donors:
            handle.write(atom["line"] + "\n")

        handle.write("END\n")


##############################################################################
# METAL LIBRARY DISCOVERY
##############################################################################

def build_metal_folders(library_dir):
    folders = {}

    for metal in METAL_ORDER:
        if metal in LANTHANIDES:
            folders[metal] = library_dir / "lanthanide" / metal
        else:
            folders[metal] = library_dir / metal

    return folders


def discover_metal_templates(metal_folders):
    """
    Returns:
        metal_shapes:
            metal -> list of shape codes

        template_files:
            shape code -> one representative template file

    A shape is searched once even if it belongs to several metals.
    """

    metal_shapes = {}
    template_files = {}

    for metal, folder in metal_folders.items():
        metal_shapes[metal] = []

        if not folder.exists():
            print(f"WARNING: metal folder does not exist: {folder}")
            continue

        for template_file in sorted(folder.glob("*.pdb")):
            shape_code = template_file.stem.lower()

            metal_shapes[metal].append(shape_code)

            if shape_code not in template_files:
                template_files[shape_code] = template_file

    return metal_shapes, template_files


##############################################################################
# READ GEOMETRY TEMPLATE
##############################################################################

def read_template(filename):
    """
    Read the first ideal block of a FindGeo-style template.

    Expected atom names:
        M = ideal metal center
        L = ideal ligand vertex

    Reading stops at the first TER record.
    """

    center = None
    vertices = []

    with open(filename, "r") as handle:
        for line in handle:
            if line.startswith("TER"):
                break

            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip().upper()

            try:
                coordinate = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            except ValueError:
                continue

            if atom_name == "M":
                center = coordinate

            elif atom_name == "L":
                vertices.append(coordinate)

    if center is None:
        raise ValueError(
            f"No template metal atom named M found in {filename}"
        )

    if not vertices:
        raise ValueError(
            f"No template ligand atoms named L found in {filename}"
        )

    vertices = np.asarray(vertices, dtype=float)

    return center, vertices


##############################################################################
# DISTANCE MATRIX AND SHAPE SIGNATURE
##############################################################################

def make_distance_matrix(coordinates):
    differences = (
        coordinates[:, None, :]
        - coordinates[None, :, :]
    )

    return np.sqrt(
        np.sum(differences * differences, axis=2)
    )


def pairwise_signature(coordinates):
    distances = []

    number_of_points = len(coordinates)

    for i in range(number_of_points):
        for j in range(i + 1, number_of_points):
            distances.append(
                np.linalg.norm(
                    coordinates[i] - coordinates[j]
                )
            )

    distances = np.asarray(distances, dtype=float)

    if len(distances) == 0:
        return None

    mean_distance = float(np.mean(distances))

    if mean_distance <= 0.0:
        return None

    return np.sort(
        distances / mean_distance
    )


def signature_rmsd(signature_a, signature_b):
    if signature_a is None or signature_b is None:
        return float("inf")

    if len(signature_a) != len(signature_b):
        return float("inf")

    return float(
        np.sqrt(
            np.mean(
                (signature_a - signature_b) ** 2
            )
        )
    )


##############################################################################
# GENERATE LOCAL DONOR COMBINATIONS
##############################################################################

def generate_local_combinations(
    donor_distance_matrix,
    coordination_number,
    maximum_distance,
):
    """
    Generate combinations for which every donor-donor distance is no
    greater than maximum_distance.
    """

    donor_count = len(donor_distance_matrix)

    if coordination_number > donor_count:
        return

    for first in range(donor_count):
        neighbors = [
            index
            for index in range(first + 1, donor_count)
            if donor_distance_matrix[first, index] <= maximum_distance
        ]

        if len(neighbors) < coordination_number - 1:
            continue

        for tail in itertools.combinations(
            neighbors,
            coordination_number - 1,
        ):
            combination = (first,) + tail
            valid = True

            for i, j in itertools.combinations(combination, 2):
                if donor_distance_matrix[i, j] > maximum_distance:
                    valid = False
                    break

            if valid:
                yield combination


##############################################################################
# PREFIT CANDIDATE SELECTION
##############################################################################

def get_best_prefit_candidates(
    donor_coordinates,
    donor_distance_matrix,
    template_vertices,
    maximum_distance,
    prefit_keep,
    prefit_cutoff,
):
    coordination_number = len(template_vertices)

    template_signature = pairwise_signature(
        template_vertices
    )

    heap = []
    combinations_checked = 0

    for combination in generate_local_combinations(
        donor_distance_matrix,
        coordination_number,
        maximum_distance,
    ):
        combinations_checked += 1

        selected_coordinates = donor_coordinates[
            list(combination)
        ]

        candidate_signature = pairwise_signature(
            selected_coordinates
        )

        score = signature_rmsd(
            candidate_signature,
            template_signature,
        )

        if not np.isfinite(score):
            continue

        # Safe, permutation-invariant early rejection.
        if score > prefit_cutoff:
            continue

        # Negative score makes the worst retained score appear at heap[0].
        heap_item = (
            -score,
            combination,
        )

        if len(heap) < prefit_keep:
            heapq.heappush(heap, heap_item)
        else:
            worst_retained_score = -heap[0][0]

            if score < worst_retained_score:
                heapq.heapreplace(
                    heap,
                    heap_item,
                )

    candidates = [
        {
            "prefit_score": -negative_score,
            "combination": combination,
        }
        for negative_score, combination in heap
    ]

    candidates.sort(
        key=lambda item: item["prefit_score"]
    )

    return candidates, combinations_checked


##############################################################################
# SIMILARITY FIT
##############################################################################

def similarity_fit(template_points, target_points):
    """
    Best rotation, translation, and uniform scale.

    Returns:
        RMSD
        scale
        rotation matrix
        translation vector
        fitted template points
    """

    template_points = np.asarray(
        template_points,
        dtype=float,
    )

    target_points = np.asarray(
        target_points,
        dtype=float,
    )

    template_mean = template_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)

    centered_template = template_points - template_mean
    centered_target = target_points - target_mean

    covariance = centered_template.T @ centered_target

    left_vectors, _, right_vectors_transposed = np.linalg.svd(
        covariance
    )

    rotation = left_vectors @ right_vectors_transposed

    # Prevent mirror reflection.
    if np.linalg.det(rotation) < 0:
        left_vectors[:, -1] *= -1
        rotation = left_vectors @ right_vectors_transposed

    denominator = np.sum(
        centered_template * centered_template
    )

    if denominator <= 0.0:
        return None

    scale = (
        np.sum(
            (centered_template @ rotation)
            * centered_target
        )
        / denominator
    )

    # A negative scale would invert the geometry.
    if scale <= 0.0:
        return None

    translation = (
        target_mean
        - scale * (template_mean @ rotation)
    )

    fitted_points = (
        scale * (template_points @ rotation)
        + translation
    )

    residuals = fitted_points - target_points

    rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(residuals * residuals, axis=1)
            )
        )
    )

    return {
        "rmsd_A": rmsd,
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "fitted_points": fitted_points,
    }


##############################################################################
# ANGULAR RMSD
##############################################################################

def angle_degrees(point_a, center, point_b):
    vector_a = point_a - center
    vector_b = point_b - center

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a <= 0.0 or norm_b <= 0.0:
        return np.nan

    cosine = np.dot(vector_a, vector_b) / (
        norm_a * norm_b
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def calculate_angle_rmsd(
    template_vertices,
    template_center,
    target_vertices,
    predicted_center,
):
    angle_differences = []
    coordination_number = len(template_vertices)

    for i in range(coordination_number):
        for j in range(i + 1, coordination_number):
            template_angle = angle_degrees(
                template_vertices[i],
                template_center,
                template_vertices[j],
            )

            target_angle = angle_degrees(
                target_vertices[i],
                predicted_center,
                target_vertices[j],
            )

            if (
                np.isfinite(template_angle)
                and np.isfinite(target_angle)
            ):
                angle_differences.append(
                    target_angle - template_angle
                )

    if not angle_differences:
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                np.square(angle_differences)
            )
        )
    )


##############################################################################
# VERTEX PERMUTATIONS
##############################################################################

def generate_vertex_permutations(
    coordination_number,
    max_permutations,
    random_seed,
):
    """
    Generate all assignments when factorial(CN) is manageable.

    For larger coordination numbers, generate a deterministic subset:
        - identity permutation
        - reversed permutation
        - random unique permutations

    CN 8 has 40,320 assignments and is exhaustive under the default
    50,000 limit.

    CN 9 has 362,880 assignments, so the default searches a deterministic
    sample of 50,000 assignments.
    """

    total_permutations = math.factorial(
        coordination_number
    )

    if total_permutations <= max_permutations:
        yield from itertools.permutations(
            range(coordination_number)
        )
        return

    random_generator = np.random.default_rng(
        random_seed
    )

    seen = set()

    identity = tuple(
        range(coordination_number)
    )

    reversed_order = tuple(
        reversed(range(coordination_number))
    )

    seen.add(identity)
    yield identity

    if reversed_order not in seen:
        seen.add(reversed_order)
        yield reversed_order

    while len(seen) < max_permutations:
        permutation = tuple(
            random_generator.permutation(
                coordination_number
            ).tolist()
        )

        if permutation in seen:
            continue

        seen.add(permutation)
        yield permutation

##############################################################################
# METAL-DONOR PATH BLOCKING
##############################################################################

def atom_blocks_metal_donor_path(
    metal_xyz,
    donor_xyz,
    atom_xyz,
    blocking_radius=METAL_DONOR_BLOCKING_RADIUS,
):
    """
    True if atom_xyz lies on the metal->donor line segment and is located
    between the metal and donor.

    Atoms beyond the donor or behind the metal are ignored.
    """

    line_vector = donor_xyz - metal_xyz

    line_length_squared = np.dot(
        line_vector,
        line_vector,
    )

    if line_length_squared <= 1.0e-12:
        return False

    t = (
        np.dot(
            atom_xyz - metal_xyz,
            line_vector,
        )
        / line_length_squared
    )

    #
    # Only atoms between metal and donor count.
    #
    if t <= 0.0 or t >= 0.9:
        return False

    projection = (
        metal_xyz
        + t * line_vector
    )

    perpendicular_distance = np.linalg.norm(
    atom_xyz - projection
    )

    metal_donor_distance = np.linalg.norm(
    donor_xyz - metal_xyz
    )

    return (
        perpendicular_distance < 0.5
        and
        perpendicular_distance / metal_donor_distance < 0.10
    )


def coordination_paths_are_clear(
    predicted_center,
    assigned_indices,
    donors,
    atoms,
):
    """
    Reject sites where another atom lies directly between the predicted
    metal position and a coordinating donor atom.
    """

    for donor_index in assigned_indices:

        donor = donors[donor_index]

        donor_xyz = np.array(
            [
                donor["x"],
                donor["y"],
                donor["z"],
            ],
            dtype=float,
        )

        donor_key = (
            donor["chain"],
            donor["resid"],
            donor["icode"],
            donor["name"],
        )

        for atom in atoms:

            atom_key = (
                atom["chain"],
                atom["resid"],
                atom["icode"],
                atom["name"],
            )

            #
            # Ignore the donor itself.
            #
            if atom_key == donor_key:
                continue

            if (
                atom["chain"] == donor["chain"]
                and atom["resid"] == donor["resid"]
                and atom["icode"] == donor["icode"]
            ):
                continue

            atom_xyz = np.array(
                [
                    atom["x"],
                    atom["y"],
                    atom["z"],
                ],
                dtype=float,
            )

            if atom_blocks_metal_donor_path(
                predicted_center,
                donor_xyz,
                atom_xyz,
            ):
                return False

    return True


##############################################################################
# EXACT TEMPLATE FIT
##############################################################################

def fit_one_combination(
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
):
    selected_coordinates = donor_coordinates[
        list(combination)
    ]

    coordination_number = len(combination)

    best_fit = None

    for permutation in generate_vertex_permutations(
        coordination_number,
        max_permutations,
        random_seed,
    ):
        target_vertices = selected_coordinates[
            list(permutation)
        ]

        fit = similarity_fit(
            template_vertices,
            target_vertices,
        )

        if fit is None:
            continue

        # Reject poor Cartesian fits before center, angle, and blocking tests.
        if fit["rmsd_A"] >= max_rmsd:
            continue

        predicted_center = (
            fit["scale"]
            * (template_center @ fit["rotation"])
            + fit["translation"]
        )

        metal_donor_distances = np.linalg.norm(
            target_vertices - predicted_center,
            axis=1,
        )

        mean_radius = float(
            np.mean(metal_donor_distances)
        )

        if mean_radius <= 0.0:
            continue

        normalized_rmsd = (
            fit["rmsd_A"] / mean_radius
        )

        angle_rmsd = calculate_angle_rmsd(
            template_vertices,
            template_center,
            target_vertices,
            predicted_center,
        )

        if not np.isfinite(angle_rmsd) or angle_rmsd >= max_angle_rmsd:
            continue

        assigned_indices = tuple(
            combination[position]
            for position in permutation
        )
        #
        # Reject fits where another atom sits directly between
        # the predicted metal center and a donor atom.
        #
        if not coordination_paths_are_clear(
            predicted_center=predicted_center,
            assigned_indices=assigned_indices,
            donors=donors,
            atoms=atoms,
        ):
            continue

        result = {
            "rmsd_A": fit["rmsd_A"],
            "normalized_rmsd": normalized_rmsd,
            "angle_rmsd_deg": angle_rmsd,
            "mean_radius_A": mean_radius,
            "scale": fit["scale"],
            "predicted_center": predicted_center,
            "assigned_indices": assigned_indices,
            "prefit_score": prefit_score,
        }

        if (
            best_fit is None
            or result["rmsd_A"] < best_fit["rmsd_A"]
        ):
            best_fit = result

    return best_fit


##############################################################################
# SEARCH ONE SHAPE
##############################################################################

def search_shape(
    shape_code,
    template_file,
    donor_coordinates,
    donor_distance_matrix,
    donors,
    atoms,
    maximum_distance,
    prefit_keep,
    top_hits,
    max_permutations,
    random_seed,
    min_cn,
    max_rmsd,
    max_angle_rmsd,
    prefit_cutoff,
):
    template_center, template_vertices = read_template(
        template_file
    )

    coordination_number = len(template_vertices)

    if coordination_number < min_cn:
        print(f"  Shape {shape_code:4s} | CN {coordination_number} | skipped (minimum CN {min_cn})")
        return []

    print(
        f"  Shape {shape_code:4s} | "
        f"CN {coordination_number} | "
        f"{template_file}"
    )

    prefit_candidates, combinations_checked = (
        get_best_prefit_candidates(
            donor_coordinates,
            donor_distance_matrix,
            template_vertices,
            maximum_distance,
            prefit_keep,
            prefit_cutoff,
        )
    )

    print(
        f"       combinations checked: "
        f"{combinations_checked}"
    )

    print(
        f"       combinations fitted:  "
        f"{len(prefit_candidates)}"
    )

    results = []

    for candidate_index, candidate in enumerate(
        prefit_candidates
    ):
        # Modify the seed between combinations while retaining
        # reproducibility.
        combination_seed = (
            random_seed
            + candidate_index
            + coordination_number * 1000
        )

        fit = fit_one_combination(
            template_center=template_center,
            template_vertices=template_vertices,
            donor_coordinates=donor_coordinates,
            donors=donors,
            atoms=atoms,
            combination=candidate["combination"],
            prefit_score=candidate["prefit_score"],
            max_permutations=max_permutations,
            random_seed=combination_seed,
            max_rmsd=max_rmsd,
            max_angle_rmsd=max_angle_rmsd,
        )

        if fit is None:
            continue

        results.append({
            "shape": shape_code,
            "template_file": str(template_file),
            "coordination_number": coordination_number,
            "combinations_checked": combinations_checked,
            **fit,
        })

    # Best donor combination first.
    results.sort(
        key=lambda result: (
            result["rmsd_A"],
            result["angle_rmsd_deg"],
        )
    )

    # Deduplicate identical donor sets.
    unique_results = []
    seen_donor_sets = set()

    for result in results:
        donor_set = tuple(
            sorted(result["assigned_indices"])
        )

        if donor_set in seen_donor_sets:
            continue

        seen_donor_sets.add(donor_set)
        unique_results.append(result)

        if len(unique_results) >= top_hits:
            break

    return unique_results


def search_shape_worker(task):
    """Top-level worker so shape searches are pickle-safe under multiprocessing."""
    shape_code, template_file, common = task
    try:
        results = search_shape(
            shape_code=shape_code,
            template_file=template_file,
            **common,
        )
        return shape_code, results, None
    except Exception as error:
        return shape_code, [], f"{type(error).__name__}: {error}"


##############################################################################
# LABELS AND CONFIDENCE
##############################################################################

def donor_label(donor):
    chain = donor["chain"] if donor["chain"] else "-"
    insertion_code = donor.get("icode", "")

    residue_number = f"{donor['resid']}{insertion_code}"

    return (
        f"{chain}:"
        f"{donor['resname']}{residue_number}:"
        f"{donor['atom']}"
    )


def residue_label(donor):
    chain = donor["chain"] if donor["chain"] else "-"
    insertion_code = donor.get("icode", "")

    return (
        f"{chain}:"
        f"{donor['resname']}"
        f"{donor['resid']}{insertion_code}"
    )


def unique_residue_labels(donors, assigned_indices):
    labels = []
    seen = set()

    for donor_index in assigned_indices:
        label = residue_label(
            donors[donor_index]
        )

        if label not in seen:
            labels.append(label)
            seen.add(label)

    return labels


def classify_confidence(
    rmsd_A,
    high_confidence_cutoff,
    medium_confidence_cutoff,
):
    if rmsd_A <= high_confidence_cutoff:
        return "HIGH"

    if rmsd_A <= medium_confidence_cutoff:
        return "MEDIUM"

    return "LOW"


##############################################################################
# WRITE RANKED CSV
##############################################################################

def write_ranked_csv(
    output_file,
    metal_shapes,
    shape_results,
    donors,
    high_confidence_cutoff,
    medium_confidence_cutoff,
):
    output_rows = []

    for metal in METAL_ORDER:
        metal_results = []

        for shape_code in metal_shapes.get(metal, []):
            for shape_hit_rank, result in enumerate(
                shape_results.get(shape_code, []),
                start=1,
            ):
                metal_results.append({
                    "shape_hit_rank": shape_hit_rank,
                    **result,
                })

        # Rank all retained shape/site fits within this metal.
        metal_results.sort(
            key=lambda result: (
                result["rmsd_A"],
                result["angle_rmsd_deg"],
                result["shape"],
            )
        )

        # Shape rank is based on the best fit for each unique shape.
        best_shape_results = {}

        for result in metal_results:
            shape_code = result["shape"]

            if shape_code not in best_shape_results:
                best_shape_results[shape_code] = result

        ranked_shape_codes = sorted(
            best_shape_results,
            key=lambda code: (
                best_shape_results[code]["rmsd_A"],
                best_shape_results[code]["angle_rmsd_deg"],
                code,
            ),
        )

        shape_rank_lookup = {
            shape_code: rank
            for rank, shape_code in enumerate(
                ranked_shape_codes,
                start=1,
            )
        }

        for overall_rank, result in enumerate(
            metal_results,
            start=1,
        ):
            center = result["predicted_center"]

            donor_labels = [
                donor_label(donors[index])
                for index in result["assigned_indices"]
            ]

            residue_labels = unique_residue_labels(
                donors,
                result["assigned_indices"],
            )

            output_rows.append([
                metal,
                shape_rank_lookup[result["shape"]],
                overall_rank,
                result["shape"],
                result["shape_hit_rank"],
                result["coordination_number"],
                f'{result["rmsd_A"]:.6f}',
                f'{result["normalized_rmsd"]:.6f}',
                f'{result["angle_rmsd_deg"]:.3f}',
                f'{result["mean_radius_A"]:.3f}',
                classify_confidence(
                    result["rmsd_A"],
                    high_confidence_cutoff,
                    medium_confidence_cutoff,
                ),
                f"{center[0]:.3f}",
                f"{center[1]:.3f}",
                f"{center[2]:.3f}", 
                f'{result["prefit_score"]:.6f}',
                ";".join(donor_labels),
                ";".join(residue_labels),
                result["template_file"],
            ])

    with open(output_file, "w", newline="") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "metal",
            "shape_rank_within_metal",
            "overall_hit_rank_within_metal",
            "shape",
            "hit_rank_within_shape",
            "coordination_number",
            "rmsd_A",
            "normalized_rmsd",
            "angle_rmsd_deg",
            "mean_predicted_metal_donor_radius_A",
            "confidence",
            "predicted_metal_x",
            "predicted_metal_y",
            "predicted_metal_z",
            "prefit_score",
            "coordinating_donor_atoms",
            "coordinating_residues",
            "template_file",
        ])

        writer.writerows(output_rows)

    return output_rows


##############################################################################
# WRITE HUMAN-READABLE SUMMARY
##############################################################################

def write_ranked_summary(
    output_file,
    input_pdb,
    metal_shapes,
    shape_results,
    donors,
    high_confidence_cutoff,
    medium_confidence_cutoff,
):
    with open(output_file, "w") as handle:
        handle.write("METAL SHAPE SEARCH RESULTS\n")
        handle.write("==========================\n\n")
        handle.write(f"Input PDB: {input_pdb}\n")
        handle.write(f"Possible donor atoms: {len(donors)}\n")
        handle.write(
            f"High-confidence threshold: "
            f"RMSD <= {high_confidence_cutoff:.3f} A\n"
        )
        handle.write(
            f"Medium-confidence threshold: "
            f"RMSD <= {medium_confidence_cutoff:.3f} A\n\n"
        )

        for metal in METAL_ORDER:
            handle.write("=" * 72 + "\n")
            handle.write(metal.upper() + "\n")
            handle.write("=" * 72 + "\n\n")

            best_per_shape = []

            for shape_code in metal_shapes.get(metal, []):
                results = shape_results.get(shape_code, [])

                if not results:
                    continue

                best_per_shape.append(
                    results[0]
                )

            best_per_shape.sort(
                key=lambda result: (
                    result["rmsd_A"],
                    result["angle_rmsd_deg"],
                    result["shape"],
                )
            )

            if not best_per_shape:
                handle.write(
                    "No valid donor combinations found.\n\n"
                )
                continue

            for rank, result in enumerate(
                best_per_shape,
                start=1,
            ):
                center = result["predicted_center"]

                confidence = classify_confidence(
                    result["rmsd_A"],
                    high_confidence_cutoff,
                    medium_confidence_cutoff,
                )

                handle.write(
                    f"Rank {rank}: {result['shape']}\n"
                )

                handle.write(
                    f"  CN                    : "
                    f"{result['coordination_number']}\n"
                )

                handle.write(
                    f"  RMSD                  : "
                    f"{result['rmsd_A']:.6f} A\n"
                )

                handle.write(
                    f"  normalized RMSD       : "
                    f"{result['normalized_rmsd']:.6f}\n"
                )

                handle.write(
                    f"  angle RMSD            : "
                    f"{result['angle_rmsd_deg']:.3f} degrees\n"
                )

                handle.write(
                    f"  confidence            : "
                    f"{confidence}\n"
                )

                handle.write(
                f"  predicted metal center: "
                f"{center[0]:.3f}, "
                f"{center[1]:.3f}, "
                f"{center[2]:.3f}\n"
                )

                handle.write(
                    f"  mean metal-donor radius: "
                    f"{result['mean_radius_A']:.3f} A\n"
                )

                handle.write(
                    "  coordinating donor atoms:\n"
                )

                for vertex_number, donor_index in enumerate(
                    result["assigned_indices"],
                    start=1,
                ):
                    donor = donors[donor_index]

                    handle.write(
                        f"    V{vertex_number}: "
                        f"{donor_label(donor)} "
                        f"({donor['x']:.3f}, "
                        f"{donor['y']:.3f}, "
                        f"{donor['z']:.3f})\n"
                    )

                handle.write("\n")

            handle.write("\n")


##############################################################################
# MAIN
##############################################################################

def main():
    args = parse_arguments()

    pdb_file = args.pdb_file.expanduser().resolve()
    library_dir = args.library.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not pdb_file.exists():
        print(
            f"ERROR: input PDB does not exist: {pdb_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not pdb_file.is_file():
        print(
            f"ERROR: input PDB is not a file: {pdb_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not library_dir.exists():
        print(
            f"ERROR: metal library does not exist: {library_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.prefit_keep < 1:
        print(
            "ERROR: --prefit-keep must be at least 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.top_hits < 1:
        print(
            "ERROR: --top-hits must be at least 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.max_permutations < 1:
        print(
            "ERROR: --max-permutations must be at least 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.min_cn < 1 or args.nproc < 1:
        print("ERROR: --min-cn and --nproc must be at least 1.", file=sys.stderr)
        sys.exit(1)
    if args.max_rmsd <= 0 or args.max_angle_rmsd <= 0 or args.prefit_cutoff <= 0:
        print("ERROR: fit and prefit cutoffs must be positive.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_prefix = output_dir / pdb_file.stem

    donor_csv_file = Path(
        str(output_prefix)
        + "_possible_donors.csv"
    )

    donor_pdb_file = Path(
        str(output_prefix)
        + "_possible_donors.pdb"
    )

    ranking_csv_file = Path(
        str(output_prefix)
        + "_metal_shape_rankings.csv"
    )

    ranking_txt_file = Path(
        str(output_prefix)
        + "_metal_shape_rankings.txt"
    )

    print()
    print("=" * 72)
    print("INTEGRATED METAL-DONOR AND SHAPE SEARCH")
    print("=" * 72)
    print(f"Input PDB:       {pdb_file}")
    print(f"Shape library:   {library_dir}")
    print(f"Output directory:{output_dir}")
    print(f"Maximum donor-donor distance: {args.max_distance:.2f} A")
    print(f"Prefit combinations retained: {args.prefit_keep}")
    print(f"Top hits retained per shape:  {args.top_hits}")
    print(f"Maximum vertex permutations:  {args.max_permutations}")
    print(f"Minimum coordination number:  {args.min_cn}")
    print(f"Accepted Cartesian RMSD:      < {args.max_rmsd:.3f} A")
    print(f"Accepted angular RMSD:        < {args.max_angle_rmsd:.3f} degrees")
    print(f"Prefit signature cutoff:      <= {args.prefit_cutoff:.3f}")
    print(f"CPU shape workers:            {args.nproc}")
    print()

    ##########################################################################
    # STEP 1: READ STRUCTURE AND SELECT DONORS
    ##########################################################################

    atoms, ssbond_residues = read_pdb(
        pdb_file
    )

    excluded_sg, disulfide_pairs = find_disulfides(
        atoms,
        ssbond_residues,
    )

    donors = select_donors(
        atoms,
        excluded_sg,
    )

    print("STEP 1: DONOR SELECTION")
    print("-" * 72)
    print(f"Atoms read:                       {len(atoms)}")
    print(f"Possible donor atoms:             {len(donors)}")
    print(f"Disulfide-bonded SG atoms excluded: {len(excluded_sg)}")

    if disulfide_pairs:
        print("Distance-detected disulfide pairs:")

        for atom_a, atom_b, distance in disulfide_pairs:
            print(
                f"  {atom_a['chain'] or '-'}:"
                f"CYS{atom_a['resid']}:SG -- "
                f"{atom_b['chain'] or '-'}:"
                f"CYS{atom_b['resid']}:SG "
                f"({distance:.2f} A)"
            )

    if not donors:
        print(
            "ERROR: no possible donor atoms were selected.",
            file=sys.stderr,
        )
        sys.exit(1)

    write_donor_outputs(
        donors,
        donor_csv_file,
        donor_pdb_file,
    )

    print(f"Created donor CSV: {donor_csv_file}")
    print(f"Created donor PDB: {donor_pdb_file}")

    ##########################################################################
    # STEP 2: DISCOVER METAL-SPECIFIC SHAPES
    ##########################################################################

    metal_folders = build_metal_folders(
        library_dir
    )

    metal_shapes, template_files = discover_metal_templates(
        metal_folders
    )

    if not template_files:
        print(
            "ERROR: no shape template PDB files were found in the "
            "metal folders.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("STEP 2: TEMPLATE DISCOVERY")
    print("-" * 72)

    for metal in METAL_ORDER:
        shape_codes = metal_shapes.get(metal, [])

        print(
            f"{metal:14s}: "
            f"{len(shape_codes):2d} templates "
            f"{','.join(shape_codes)}"
        )

    print()
    print(
        f"Unique shapes to search: {len(template_files)}"
    )

    ##########################################################################
    # STEP 3: PREPARE DONOR COORDINATES
    ##########################################################################

    donor_coordinates = np.asarray(
        [
            [
                donor["x"],
                donor["y"],
                donor["z"],
            ]
            for donor in donors
        ],
        dtype=float,
    )

    donor_distance_matrix = make_distance_matrix(
        donor_coordinates
    )

    ##########################################################################
    # STEP 4: SEARCH EACH UNIQUE SHAPE ONCE
    ##########################################################################

    print()
    print("STEP 3: SHAPE SEARCH")
    print("-" * 72)

    shape_results = {}

    common_search_arguments = {
        "donor_coordinates": donor_coordinates,
        "donor_distance_matrix": donor_distance_matrix,
        "donors": donors,
        "atoms": atoms,
        "maximum_distance": args.max_distance,
        "prefit_keep": args.prefit_keep,
        "top_hits": args.top_hits,
        "max_permutations": args.max_permutations,
        "random_seed": args.random_seed,
        "min_cn": args.min_cn,
        "max_rmsd": args.max_rmsd,
        "max_angle_rmsd": args.max_angle_rmsd,
        "prefit_cutoff": args.prefit_cutoff,
    }

    tasks = [
        (shape_code, template_files[shape_code], common_search_arguments)
        for shape_code in sorted(template_files)
    ]

    if args.nproc == 1:
        completed = [search_shape_worker(task) for task in tasks]
    else:
        completed = []
        with ProcessPoolExecutor(max_workers=args.nproc) as executor:
            futures = [executor.submit(search_shape_worker, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())

    for shape_code, results, error in sorted(completed, key=lambda item: item[0]):
        shape_results[shape_code] = results
        if error is not None:
            print(f"ERROR while searching shape {shape_code}: {error}")
        elif results:
            best = results[0]
            print(
                f"  RESULT {shape_code:4s} | best RMSD {best['rmsd_A']:.4f} A | "
                f"angle RMSD {best['angle_rmsd_deg']:.2f} degrees"
            )
        else:
            print(f"  RESULT {shape_code:4s} | no accepted hit")

    ##########################################################################
    # STEP 5: WRITE METAL-SPECIFIC RANKINGS
    ##########################################################################

    output_rows = write_ranked_csv(
        output_file=ranking_csv_file,
        metal_shapes=metal_shapes,
        shape_results=shape_results,
        donors=donors,
        high_confidence_cutoff=args.high_confidence,
        medium_confidence_cutoff=args.medium_confidence,
    )

    write_ranked_summary(
        output_file=ranking_txt_file,
        input_pdb=pdb_file,
        metal_shapes=metal_shapes,
        shape_results=shape_results,
        donors=donors,
        high_confidence_cutoff=args.high_confidence,
        medium_confidence_cutoff=args.medium_confidence,
    )

    ##########################################################################
    # FINAL SUMMARY
    ##########################################################################

    high_confidence_count = sum(
        1
        for row in output_rows
        if row[10] == "HIGH"
    )

    medium_confidence_count = sum(
        1
        for row in output_rows
        if row[10] == "MEDIUM"
    )

    low_confidence_count = sum(
        1
        for row in output_rows
        if row[10] == "LOW"
    )

    print()
    print("=" * 72)
    print("SCAN COMPLETE")
    print("=" * 72)
    print(f"Total metal-specific ranked rows: {len(output_rows)}")
    print(f"High-confidence rows:             {high_confidence_count}")
    print(f"Medium-confidence rows:           {medium_confidence_count}")
    print(f"Low-confidence rows:              {low_confidence_count}")
    print()
    print("Created:")
    print(f"  {donor_csv_file}")
    print(f"  {donor_pdb_file}")
    print(f"  {ranking_csv_file}")
    print(f"  {ranking_txt_file}")
    print()


if __name__ == "__main__":
    main()