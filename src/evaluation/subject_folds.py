"""Subject-held-out folds for leakage-free reliability training."""

from collections import Counter
from typing import Dict, List


# ── 1. Participant configuration ───────────────────

TRAIN_SUBJECTS = tuple(range(1, 11))
VALIDATION_SUBJECTS = (11, 12)
TEST_SUBJECTS = (13, 14, 15)

HELD_OUT_GROUPS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
)


# ── 2. Construct subject-held-out folds ────────────

def create_subject_folds() -> List[Dict]:

    folds = []

    for fold_id, held_out in enumerate(
        HELD_OUT_GROUPS,
        start=1,
    ):

        training = tuple(
            subject
            for subject in TRAIN_SUBJECTS
            if subject not in held_out
        )

        folds.append({
            "fold": fold_id,
            "train_subjects": training,
            "held_out_subjects": held_out,
        })

    return folds


# ── 3. Validate fold integrity ─────────────────────

def validate_subject_folds(folds: List[Dict]) -> bool:

    assert len(folds) == 5, (
        "Expected exactly five subject-held-out folds."
    )

    held_out_occurrences = []

    for fold in folds:

        training = set(fold["train_subjects"])
        held_out = set(fold["held_out_subjects"])

        assert len(training) == 8
        assert len(held_out) == 2

        assert training.isdisjoint(held_out), (
            f"Fold {fold['fold']}: training/held-out overlap."
        )

        assert training | held_out == set(TRAIN_SUBJECTS), (
            f"Fold {fold['fold']}: incomplete training coverage."
        )

        assert (
            training | held_out
        ).isdisjoint(
            set(VALIDATION_SUBJECTS)
            | set(TEST_SUBJECTS)
        ), (
            f"Fold {fold['fold']}: validation/test leakage."
        )

        held_out_occurrences.extend(
            fold["held_out_subjects"]
        )

    counts = Counter(held_out_occurrences)

    assert set(counts) == set(TRAIN_SUBJECTS), (
        "Not all training participants were held out."
    )

    assert all(count == 1 for count in counts.values()), (
        "Each training participant must be held out once."
    )

    return True


# ── 4. Display fold configuration ──────────────────

if __name__ == "__main__":

    folds = create_subject_folds()

    validate_subject_folds(folds)

    print("Subject-Held-Out Fold Configuration\n")

    for fold in folds:

        print(
            f"Fold {fold['fold']}: "
            f"Train {list(fold['train_subjects'])} | "
            f"Held out {list(fold['held_out_subjects'])}"
        )

    print("\nFold integrity: PASSED")
    print("Validation/test isolation: PASSED")
    print("Step 50.3: COMPLETED")