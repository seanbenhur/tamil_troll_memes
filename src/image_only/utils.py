
def load_checkpoint(checkpoint,model,optimizer,scheduler):
    print(" == LOADING CHECKPOINT")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpont["optimizer_state_dict"])
    loss.state_dict(checkpoint["loss"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint['epoch']

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
        