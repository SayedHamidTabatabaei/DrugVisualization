from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import colorama  # For cross-platform ANSI color support

colorama.init()


class CustomizeEpochProgress(Callback):
    def __init__(self):
        super().__init__()
        self.pbar = None
        self.steps = None
        self.epochs = None

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params.get('steps', None)  # Total steps per epoch

    def on_epoch_begin(self, epoch, logs=None):
        epoch_desc = f"\033[1;34mEpoch {epoch + 1}/{self.epochs}\033[0m"  # Blue text with bold
        self.pbar = tqdm(
            total=self.steps,
            desc=epoch_desc,
            unit="batch",
            leave=True,
            ncols=100,
            colour="green",
            bar_format=(
                "{l_bar}\033[1;33m{bar}\033[0m | "
                "\033[1;36mElapsed: {elapsed}\033[0m | "
                "\033[1;31mRemaining: {remaining}\033[0m | "
                "\033[1;35m{n_fmt}/{total_fmt} Batches\033[0m"
            )
        )

    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)  # Update progress bar by one step
        self.pbar.set_postfix(loss=logs.get('loss', 0))

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()  # Close progress bar at epoch end

