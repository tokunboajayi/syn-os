"""
Syn OS — Nightly Self-Training Pipeline (``synapse train``)

Reads the day's experience-replay data from disk, trains the
HardwareFingerprinter autoencoder, and saves the updated weights.

Intended to be triggered via cron / systemd timer:
    0 3 * * *  cd /opt/syn-os && python -m synos_ml.train_online

It can also be invoked manually from the API via the /synapse/train endpoint.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")


def run_training(
    model_dir: str = "models",
    replay_dir: str = "data/replay",
    epochs: int = 5,
    batch_size: int = 64,
    max_replay_files: int = 20,
) -> dict:
    """
    Execute one training session.

    Returns a summary dict with loss curve and sample counts.
    """
    from synos_ml.models.fingerprinter import HardwareFingerprinter
    from synos_ml.core.replay_buffer import ExperienceReplay

    logger.info("═══ Synapse Nightly Training ═══")
    start = time.time()

    # 1. Initialise model & replay buffer
    fp = HardwareFingerprinter(model_dir=model_dir)
    replay = ExperienceReplay(persist_dir=replay_dir)
    loaded = replay.load_from_disk(max_files=max_replay_files)

    if replay.size < batch_size:
        msg = f"Not enough data ({replay.size} < {batch_size}). Skipping training."
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "buffer_size": replay.size}

    # 2. Train
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        # Iterate through the buffer in mini-batches
        steps = max(1, replay.size // batch_size)
        for _ in range(steps):
            batch = replay.sample_states(batch_size)
            loss = fp.train_step(batch)
            epoch_losses.append(loss)

        avg = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg)
        logger.info(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.6f}")

    # 3. Save updated model
    fp.save()

    duration = time.time() - start
    result = {
        "status": "completed",
        "epochs": epochs,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_curve": losses,
        "samples_used": replay.size,
        "total_model_samples": fp.total_trained_samples,
        "duration_seconds": round(duration, 2),
    }
    logger.info(f"═══ Training Complete in {duration:.1f}s ═══")
    logger.info(f"  Final loss: {result['final_loss']:.6f}")
    return result


if __name__ == "__main__":
    result = run_training()
    print(result)
