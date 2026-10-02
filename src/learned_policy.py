import math
from collections import Counter

from sklearn.tree import DecisionTreeClassifier

from page_replacement import optimal_victim


class LearnedEvictionPolicy:
    def __init__(self, capacity):
        self.capacity = capacity

        self.model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced"
        )

        self.feature_names = [
            "recency",
            "frequency",
            "age",
            "frequency_per_age",
            "recency_ratio",
            "recency_rank",
            "frequency_rank",
            "age_rank",
            "frequency_relative",
            "recency_relative",
            "age_relative"
        ]

        self.is_trained = False

    # ============================================================
    # FEATURE CALCULATION
    # ============================================================

    def _get_features(
        self,
        t,
        page,
        frames,
        last_used,
        freq,
        loaded_at
    ):
        current_freq = freq.get(page, 0)

        if page in last_used:
            recency = t - last_used[page]
        else:
            recency = t + 1

        if page in loaded_at:
            age = t - loaded_at[page]
        else:
            age = t + 1

        frequency_per_age = (
            current_freq / (age + 1)
        )

        recency_ratio = (
            recency / (t + 1)
        )

        frequencies = [
            freq.get(p, 0)
            for p in frames
        ]

        recencies = [
            t - last_used[p]
            if p in last_used
            else t + 1
            for p in frames
        ]

        ages = [
            t - loaded_at[p]
            if p in loaded_at
            else t + 1
            for p in frames
        ]

        # --------------------------------------------------------
        # Rank features
        # --------------------------------------------------------

        frequency_rank = (
            sum(
                f <= current_freq
                for f in frequencies
            )
            / len(frequencies)
            if frequencies
            else 0.0
        )

        recency_rank = (
            sum(
                r >= recency
                for r in recencies
            )
            / len(recencies)
            if recencies
            else 0.0
        )

        age_rank = (
            sum(
                a <= age
                for a in ages
            )
            / len(ages)
            if ages
            else 0.0
        )

        # --------------------------------------------------------
        # Relative features
        # --------------------------------------------------------

        avg_frequency = (
            sum(frequencies) / len(frequencies)
            if frequencies
            else 0.0
        )

        avg_recency = (
            sum(recencies) / len(recencies)
            if recencies
            else 0.0
        )

        avg_age = (
            sum(ages) / len(ages)
            if ages
            else 0.0
        )

        frequency_relative = (
            current_freq / (avg_frequency + 1e-6)
        )

        recency_relative = (
            recency / (avg_recency + 1e-6)
        )

        age_relative = (
            age / (avg_age + 1e-6)
        )

        return [
            recency,
            current_freq,
            age,
            frequency_per_age,
            recency_ratio,
            recency_rank,
            frequency_rank,
            age_rank,
            frequency_relative,
            recency_relative,
            age_relative
        ]

    # ============================================================
    # BUILD TRAINING DATASET
    # ============================================================

    def _build_dataset(self, traces):
        X = []
        y = []

        for trace in traces:

            frames = []
            last_used = {}
            freq = Counter()
            loaded_at = {}

            for t, page in enumerate(trace):

                # ------------------------------------------------
                # PAGE HIT
                # ------------------------------------------------

                if page in frames:

                    freq[page] += 1
                    last_used[page] = t

                    continue

                # ------------------------------------------------
                # FREE FRAME
                # ------------------------------------------------

                if len(frames) < self.capacity:

                    frames.append(page)

                    loaded_at[page] = t
                    freq[page] += 1
                    last_used[page] = t

                    continue

                # ------------------------------------------------
                # FRAME FULL
                # ------------------------------------------------

                victim = optimal_victim(
                    trace,
                    t,
                    frames
                )

                # Create one training sample for every
                # possible victim page.
                for candidate in frames:

                    features = self._get_features(
                        t,
                        candidate,
                        frames,
                        last_used,
                        freq,
                        loaded_at
                    )

                    X.append(features)

                    if candidate == victim:
                        y.append(1)
                    else:
                        y.append(0)

                # Follow Optimal policy while creating
                # the training dataset.

                frames.remove(victim)
                frames.append(page)

                loaded_at[page] = t
                freq[page] += 1
                last_used[page] = t

        return X, y

    # ============================================================
    # TRAIN MODEL
    # ============================================================

    def train(self, traces):
        print()
        print("Building training dataset...")

        # --------------------------------------------------------
        # Split traces into training and validation
        # --------------------------------------------------------

        split_index = int(len(traces) * 0.8)

        train_traces = traces[:split_index]
        validation_traces = traces[split_index:]

        print("Building validation dataset...")

        X_train, y_train = self._build_dataset(
            train_traces
        )

        X_validation, y_validation = self._build_dataset(
            validation_traces
        )

        print()
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_validation)}")

        # --------------------------------------------------------
        # Train Decision Tree
        # --------------------------------------------------------

        self.model.fit(
            X_train,
            y_train
        )

        self.is_trained = True

        # --------------------------------------------------------
        # Accuracy
        # --------------------------------------------------------

        train_accuracy = self.model.score(
            X_train,
            y_train
        )

        validation_accuracy = self.model.score(
            X_validation,
            y_validation
        )

        print(
            f"Training accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        # --------------------------------------------------------
        # Feature importance
        # --------------------------------------------------------

        print()
        print("Feature importance:")

        importances = self.model.feature_importances_

        feature_importance = sorted(
            zip(
                self.feature_names,
                importances
            ),
            key=lambda x: x[1],
            reverse=True
        )

        for name, importance in feature_importance:

            print(
                f"  {name:<22}"
                f"{importance:.4f}"
            )

        print()
        print("Training complete.")

    # ============================================================
    # PREDICT VICTIM
    # ============================================================

    def _predict_victim(
        self,
        t,
        frames,
        last_used,
        freq,
        loaded_at
    ):
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before prediction."
            )

        candidate_features = []

        for candidate in frames:

            features = self._get_features(
                t,
                candidate,
                frames,
                last_used,
                freq,
                loaded_at
            )

            candidate_features.append(
                features
            )

        # --------------------------------------------------------
        # Get probability that each candidate is the victim.
        # --------------------------------------------------------

        probabilities = self.model.predict_proba(
            candidate_features
        )

        classes = list(
            self.model.classes_
        )

        # Probability corresponding to class 1
        if 1 in classes:

            victim_class_index = classes.index(1)

            victim_probabilities = [
                p[victim_class_index]
                for p in probabilities
            ]

        else:

            victim_probabilities = [
                0.0
                for _ in frames
            ]

        # --------------------------------------------------------
        # Select candidate with highest victim probability.
        # --------------------------------------------------------

        best_index = max(
            range(len(frames)),
            key=lambda i: victim_probabilities[i]
        )

        return frames[best_index]

    # ============================================================
    # CORRECTED VICTIM DECISION DIAGNOSTIC
    # ============================================================

    def evaluate_victim_accuracy(
        self,
        trace,
        split
    ):
        """
        Diagnostic comparison between the Learned policy
        and the Optimal policy.

        Both decisions are made from the SAME frame state.

        After comparing the decisions, the OPTIMAL decision
        is applied to the frame state so that the next
        comparison also uses the Optimal reference state.

        This function does NOT change the actual Learned
        simulation.
        """

        frames = []
        last_used = {}
        freq = Counter()
        loaded_at = {}

        before_correct = 0
        before_total = 0

        after_correct = 0
        after_total = 0

        for t, page in enumerate(trace):

            after_shift = t >= split

            # --------------------------------------------------
            # PAGE HIT
            # --------------------------------------------------

            if page in frames:

                freq[page] += 1
                last_used[page] = t

                continue

            # --------------------------------------------------
            # FREE FRAME AVAILABLE
            # --------------------------------------------------

            if len(frames) < self.capacity:

                frames.append(page)

                loaded_at[page] = t
                freq[page] += 1
                last_used[page] = t

                continue

            # --------------------------------------------------
            # FRAME IS FULL
            # --------------------------------------------------

            # Optimal victim
            optimal_victim_page = optimal_victim(
                trace,
                t,
                frames
            )

            # Learned victim
            learned_victim_page = self._predict_victim(
                t,
                frames,
                last_used,
                freq,
                loaded_at
            )

            # --------------------------------------------------
            # COMPARE DECISIONS
            # --------------------------------------------------

            if after_shift:

                after_total += 1

                if (
                    learned_victim_page
                    == optimal_victim_page
                ):
                    after_correct += 1

            else:

                before_total += 1

                if (
                    learned_victim_page
                    == optimal_victim_page
                ):
                    before_correct += 1

            # --------------------------------------------------
            # IMPORTANT:
            #
            # Apply OPTIMAL decision here.
            #
            # This keeps the diagnostic state aligned with
            # the Optimal reference policy.
            # --------------------------------------------------

            frames.remove(
                optimal_victim_page
            )

            frames.append(page)

            loaded_at[page] = t
            freq[page] += 1
            last_used[page] = t

        # ------------------------------------------------------
        # CALCULATE ACCURACY
        # ------------------------------------------------------

        before_accuracy = (
            before_correct / before_total
            if before_total > 0
            else 0.0
        )

        after_accuracy = (
            after_correct / after_total
            if after_total > 0
            else 0.0
        )

        total_correct = (
            before_correct
            + after_correct
        )

        total_decisions = (
            before_total
            + after_total
        )

        total_accuracy = (
            total_correct / total_decisions
            if total_decisions > 0
            else 0.0
        )

        # ------------------------------------------------------
        # PRINT RESULTS
        # ------------------------------------------------------

        print(
            "============================================================"
        )

        print(
            "VICTIM DECISION DIAGNOSTIC"
        )

        print(
            "============================================================"
        )

        print(
            f"Total victim decisions: "
            f"{total_decisions}"
        )

        print(
            f"Correct victim decisions: "
            f"{total_correct}"
        )

        print(
            f"Overall victim accuracy: "
            f"{total_accuracy:.4f}"
        )

        print()

        print("Before Shift:")

        print(
            f"  Decisions: "
            f"{before_total}"
        )

        print(
            f"  Correct: "
            f"{before_correct}"
        )

        print(
            f"  Victim accuracy: "
            f"{before_accuracy:.4f}"
        )

        print()

        print("After Shift:")

        print(
            f"  Decisions: "
            f"{after_total}"
        )

        print(
            f"  Correct: "
            f"{after_correct}"
        )

        print(
            f"  Victim accuracy: "
            f"{after_accuracy:.4f}"
        )

        print(
            "============================================================"
        )

    # ============================================================
    # LEARNED POLICY SIMULATION
    # ============================================================

    def simulate(self, trace):
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before simulation."
            )

        frames = []
        last_used = {}
        freq = Counter()
        loaded_at = {}

        hits = 0
        faults = 0

        for t, page in enumerate(trace):

            # --------------------------------------------------
            # PAGE HIT
            # --------------------------------------------------

            if page in frames:

                hits += 1

                freq[page] += 1
                last_used[page] = t

                continue

            # --------------------------------------------------
            # PAGE FAULT
            # --------------------------------------------------

            faults += 1

            # --------------------------------------------------
            # FREE FRAME
            # --------------------------------------------------

            if len(frames) < self.capacity:

                frames.append(page)

                loaded_at[page] = t
                freq[page] += 1
                last_used[page] = t

                continue

            # --------------------------------------------------
            # LEARNED EVICTION
            # --------------------------------------------------

            victim = self._predict_victim(
                t,
                frames,
                last_used,
                freq,
                loaded_at
            )

            frames.remove(victim)
            frames.append(page)

            loaded_at[page] = t
            freq[page] += 1
            last_used[page] = t

        return hits, faults

    # ============================================================
    # PHASE-BASED LEARNED SIMULATION
    # ============================================================

    def simulate_phases(
        self,
        trace,
        split
    ):
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before simulation."
            )

        frames = []
        last_used = {}
        freq = Counter()
        loaded_at = {}

        before_hits = 0
        before_faults = 0

        after_hits = 0
        after_faults = 0

        for t, page in enumerate(trace):

            after_shift = t >= split

            # --------------------------------------------------
            # PAGE HIT
            # --------------------------------------------------

            if page in frames:

                if after_shift:
                    after_hits += 1
                else:
                    before_hits += 1

                freq[page] += 1
                last_used[page] = t

                continue

            # --------------------------------------------------
            # PAGE FAULT
            # --------------------------------------------------

            if after_shift:
                after_faults += 1
            else:
                before_faults += 1

            # --------------------------------------------------
            # FREE FRAME
            # --------------------------------------------------

            if len(frames) < self.capacity:

                frames.append(page)

                loaded_at[page] = t
                freq[page] += 1
                last_used[page] = t

                continue

            # --------------------------------------------------
            # LEARNED EVICTION
            # --------------------------------------------------

            victim = self._predict_victim(
                t,
                frames,
                last_used,
                freq,
                loaded_at
            )

            frames.remove(victim)
            frames.append(page)

            loaded_at[page] = t
            freq[page] += 1
            last_used[page] = t

        return (
            before_hits,
            before_faults,
            after_hits,
            after_faults
        )