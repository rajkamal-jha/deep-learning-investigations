import torch
import copy


class RollbackLROnPlateau:

    def __init__(
        self,
        model,
        optimizer,
        factor=0.25,
        patience=6,
        threshold=1e-4,
        min_lr=1e-6,
        max_rollbacks=4,
        reset_optimizer=True
    ):

        self.model = model
        self.optimizer = optimizer

        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.max_rollbacks = max_rollbacks
        self.reset_optimizer = reset_optimizer

        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.rollback_count = 0

        self.best_model_state = None
        self.best_optimizer_state = None

        self.best_epoch = None

    # ==================================================
    # SAVE BEST MODEL
    # ==================================================

    def _save_best(self, epoch, val_loss):

        self.best_loss = val_loss
        self.best_epoch = epoch
        self.bad_epochs = 0

        self.best_model_state = copy.deepcopy(
            self.model.state_dict()
        )

        if not self.reset_optimizer:

            self.best_optimizer_state = copy.deepcopy(
                self.optimizer.state_dict()
            )

    # ==================================================
    # RESTORE BEST MODEL
    # ==================================================

    def _restore_best(self):

        self.model.load_state_dict(
            self.best_model_state
        )

        if self.reset_optimizer:

            # Start optimizer fresh from restored weights
            self.optimizer.state.clear()

        elif self.best_optimizer_state is not None:

            self.optimizer.load_state_dict(
                self.best_optimizer_state
            )

    # ==================================================
    # REDUCE LEARNING RATE
    # ==================================================

    def _reduce_lr(self):

        lr_changes = []

        for param_group in self.optimizer.param_groups:

            old_lr = param_group["lr"]

            new_lr = max(
                old_lr * self.factor,
                self.min_lr
            )

            param_group["lr"] = new_lr

            lr_changes.append(
                (old_lr, new_lr)
            )

        return lr_changes

    # ==================================================
    # STEP
    # ==================================================

    def step(self, val_loss, epoch):

        # --------------------------------------------------
        # Validation improved
        # --------------------------------------------------

        if val_loss < self.best_loss - self.threshold:

            self._save_best(
                epoch,
                val_loss
            )

            return {
                "improved": True,
                "rollback": False,
                "lr_changed": False
            }

        # --------------------------------------------------
        # No improvement
        # --------------------------------------------------

        self.bad_epochs += 1

        # --------------------------------------------------
        # Plateau detected
        # --------------------------------------------------

        if self.bad_epochs >= self.patience:

            # ==============================================
            # CASE 1:
            # Rollback still available
            # ==============================================

            if self.rollback_count < self.max_rollbacks:

                self.rollback_count += 1

                restored_epoch = self.best_epoch

                self._restore_best()

                lr_changes = self._reduce_lr()

                self.bad_epochs = 0

                print()
                print("=" * 60)
                print(
                    f"ROLLBACK #{self.rollback_count}"
                )
                print(
                    f"Restoring epoch "
                    f"{restored_epoch + 1}"
                )
                print(
                    f"Best Val Loss: "
                    f"{self.best_loss:.6f}"
                )

                for old_lr, new_lr in lr_changes:

                    print(
                        f"LR: "
                        f"{old_lr:.2e} → "
                        f"{new_lr:.2e}"
                    )

                print("=" * 60)

                return {
                    "improved": False,
                    "rollback": True,
                    "lr_changed": True
                }

            # ==============================================
            # CASE 2:
            # Maximum rollbacks reached
            # ==============================================

            else:

                lr_changes = self._reduce_lr()

                self.bad_epochs = 0

                print()
                print("=" * 60)
                print(
                    "LR REDUCTION"
                )
                print(
                    "Maximum rollbacks reached."
                )
                print(
                    "Continuing without rollback."
                )

                for old_lr, new_lr in lr_changes:

                    print(
                        f"LR: "
                        f"{old_lr:.2e} → "
                        f"{new_lr:.2e}"
                    )

                print("=" * 60)

                return {
                    "improved": False,
                    "rollback": False,
                    "lr_changed": True
                }

        # --------------------------------------------------
        # Nothing happened yet
        # --------------------------------------------------

        return {
            "improved": False,
            "rollback": False,
            "lr_changed": False
        }