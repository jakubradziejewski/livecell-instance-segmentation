# src/utils/wandb_tracker.py

import wandb
import torch
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class WandBTracker:
    """
    Weights & Biases experiment tracker with secure API key handling.
    
    Environment variables (set in .env):
        WANDB_API_KEY: Your W&B API key
        WANDB_ENTITY: Your W&B username (optional)
        WANDB_PROJECT: Project name (optional, defaults to 'livecell-instance-segmentation')
    """
    
    def __init__(
        self, 
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize W&B tracking.
        
        Args:
            run_name: Name for this specific run (e.g., "maskrcnn-resnet50-v1")
            config: Dictionary with hyperparameters
            tags: List of tags (e.g., ["transfer-learning", "resnet50"])
            notes: Description of this experiment
        """
        # Get credentials from environment
        api_key = os.getenv('WANDB_API_KEY')
        entity = os.getenv('WANDB_ENTITY')
        project = os.getenv('WANDB_PROJECT', 'livecell-instance-segmentation')
        
        # Validate API key exists (REMOVED LENGTH CHECK!)
        if not api_key:
            raise ValueError(
                "WANDB_API_KEY not found in environment variables!\n"
                "Please set it in your .env file or run: wandb login"
            )
        
        # Login with API key
        wandb.login(key=api_key)
        
        # Initialize run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True
        )
        
        print(f"✅ W&B initialized: {self.run.url}")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        wandb.log(metrics, step=step)
    
    def log_images(self, images: Dict[str, Any], step: Optional[int] = None):
        """Log images to W&B."""
        wandb_images = {}
        for key, img in images.items():
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            wandb_images[key] = wandb.Image(img)
        
        wandb.log(wandb_images, step=step)
    
    def log_model(self, model_path: str, name: str = "model", aliases: list = None):
        """Save model artifact to W&B."""
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(model_path)
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        print(f"✅ Model saved to W&B: {name}")
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Track model gradients and parameters."""
        wandb.watch(model, log='all', log_freq=log_freq)
        print("✅ Model watching enabled (gradients + parameters)")
    
    def finish(self):
        """End the W&B run."""
        wandb.finish()
        print("✅ W&B run finished")


# Example usage
if __name__ == "__main__":
    """Test W&B tracker."""
    
    test_config = {
        "model": "maskrcnn_resnet50",
        "learning_rate": 0.001,
        "batch_size": 4,
        "epochs": 50,
        "optimizer": "adam",
        "dataset": "LIVECell",
        "num_classes": 2,
    }
    
    try:
        # Initialize tracker
        tracker = WandBTracker(
            run_name="test-run-delete-me",
            config=test_config,
            tags=["test", "setup"],
            notes="Testing W&B integration"
        )
        
        # Simulate training loop
        print("\n🚀 Simulating training...")
        for epoch in range(5):
            train_loss = 0.8 - epoch * 0.1
            val_loss = 0.9 - epoch * 0.08
            val_map = 0.5 + epoch * 0.08
            
            tracker.log_metrics({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mAP": val_map,
            }, step=epoch)
            
            print(f"  Epoch {epoch}: train_loss={train_loss:.3f}, val_mAP={val_map:.3f}")
        
        print(f"\n✅ Test successful! Check dashboard: {tracker.run.url}")
        tracker.finish()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")