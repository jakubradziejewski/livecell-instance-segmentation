import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_wandb_login():
    """Simple test to verify W&B login works."""
    
    api_key = os.getenv('WANDB_API_KEY')
    
    if not api_key:
        print("WANDB_API_KEY not found in .env file")
        return False
    
    try:
        wandb.login(key=api_key)
        print("W&B login successful!")
        print(f"Logged in as: {wandb.api.viewer()['entity']}")
        return True
    
    except Exception as e:
        print(f"W&B login failed: {e}")
        return False


if __name__ == "__main__":
    test_wandb_login()