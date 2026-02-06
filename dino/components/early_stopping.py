import tqdm
import torch

from typing import Optional
from pathlib import Path


class EarlyStoppingDINO:
    """
    Leverage a downstream classification task to know if teacher still outperforms student
    """

    def __init__(
        self,
        tracking: str,
        min_max: str,
        patience: int = 20,
        min_epoch: int = 50,
        checkpoint_dir: Optional[Path] = None,
        save_every: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = patience
        self.min_epoch = min_epoch
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.verbose = verbose

        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, epoch, results, snapshot):
        if results is not None:
            teacher_score = results["teacher"][self.tracking]
            student_score = results["student"][self.tracking]

            if self.min_max == "min":
                teacher_score = -1 * teacher_score
                student_score = -1 * student_score

            if self.best_score is None or (
                teacher_score >= self.best_score and teacher_score > student_score
            ):
                self.best_score = teacher_score
                torch.save(snapshot, Path(self.checkpoint_dir, "best.pt"))
                self.counter = 0

            elif teacher_score < self.best_score or teacher_score <= student_score:
                self.counter += 1
                if epoch <= self.min_epoch + 1 and self.verbose:
                    tqdm.tqdm.write(
                        f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                    )
                elif self.verbose:
                    tqdm.tqdm.write(
                        f"EarlyStopping counter: {self.counter}/{self.patience}"
                    )
                if self.counter >= self.patience and epoch > self.min_epoch:
                    self.early_stop = True

        if self.save_every and epoch % self.save_every == 0:
            fname = f"snapshot_epoch_{epoch+1:03}.pt"
            torch.save(snapshot, Path(self.checkpoint_dir, fname))

        # override latest
        torch.save(snapshot, Path(self.checkpoint_dir, "latest.pt"))
