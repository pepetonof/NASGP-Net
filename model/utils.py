import torch
import matplotlib.pyplot as plt

def save_checkpoint(state, file_path="my_checkpoint.pth.tar"):
    #print("=> Saving checkpoint")
    torch.save(state, file_path)
    return

def load_checkpoint(checkpoint, model):
    #print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return

def save_model(model, optimizer, file_path):
    # print('=> Saving model...\t')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        }
    save_checkpoint(checkpoint, filename=file_path)#ruta+"/my_checkpoint.pth.tar")
    del checkpoint
    torch.cuda.empty_cache()
    return

def load_model(model, ruta, filename):
    # print('=> Loading model...\t')
    load_checkpoint(ruta+'/'+filename, model)
    return

def save_graphtv(train_loss, valid_loss, ruta, filename):
    epochs=[i for i in range(len(train_loss))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_loss, "b-", label="Train loss")
    line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)
    return

def save_graphtvd(train_dice, valid_dice, ruta, filename):
    epochs=[i for i in range(len(train_dice))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_dice, "b-", label="Train Dice")
    line2 = ax1.plot(epochs, valid_dice, "r-", label="Valid Dice")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)
    return