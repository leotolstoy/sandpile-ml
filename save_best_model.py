import torch

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation score is greater than the previous best score, then save the
    model state.
    """
    def __init__(self, path, best_valid_score=-float('inf')):
        self.path = path
        self.best_valid_score = best_valid_score
        
    def __call__(
        self, current_valid_score, 
        episode_i, model, optimizer
    ):
        if current_valid_score > self.best_valid_score:
            self.best_valid_score = current_valid_score
            print(f"\nBest validation loss: {self.best_valid_score}")
            print(f"\nSaving best model for episode: {episode_i}\n")
            torch.save({
                'episode': episode_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_valid_score,
                }, self.path)