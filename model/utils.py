import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    #print("=> Saving checkpoint")
    torch.save(state, filename)
    return

def load_checkpoint(checkpoint, model):
    #print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return

def save_model(model, optimizer, ruta):
    # print('=> Saving model...\t')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        }
    save_checkpoint(checkpoint, filename=ruta+"/my_checkpoint.pth.tar")
    del checkpoint
    torch.cuda.empty_cache()
    return

def load_model(model, ruta, filename):
    # print('=> Loading model...\t')
    load_checkpoint(ruta+'/'+filename, model)
    return