import pkbar
import torch

def model_loss(model, sentiment, review):
    sentiment = sentiment.type(torch.cuda.LongTensor)           
    review = review.type(torch.cuda.LongTensor)  
    
    # loss is computed by BertForSequenceClassification
    loss, _ = model(review, sentiment)      # discarding pred here

    return loss


def save_checkpoint(model, optimizer, epoch, valid_loss, final=False):
    beep = "chkpt"
    if final:
        beep = "final"

    torch.save(
        {'model_state_dict' : model.state_dict(),
         'optimizer_state_dict' : optimizer.state_dict(),
         'epoch' : epoch,
         'valid_loss' : valid_loss}, 
         './checkpoints/model_{}.pt'.format(beep))

    print("\nSaved checkpoint... Validation loss reported: {}\n".format(valid_loss))


def train(model, optimizer, t_loader, v_loader, num_epochs=5, best_loss = float("Inf")):
    valid_loss = 0.0

    model.train()
    for epoch in range(num_epochs):
        kbar = pkbar.Kbar(target=len(t_loader), epoch=epoch, 
                          num_epochs=num_epochs, width=10, always_stateful=False)

        for p_t, ((review, sentiment), _) in enumerate(t_loader):
            optimizer.zero_grad()       # zero grads 
            loss = model_loss(model, sentiment, review)     # calc loss
            loss.backward()         # update grads
            optimizer.step()        # update parameters

            # update progress bar
            kbar.update(p_t, values=[("training loss", loss.item())])

            # validation 
            if p_t != 0 and p_t % 17085 == 0:        # => 5 times per epoch
                print("\n\n Performing validation cycle.\n")

                model.eval()    # pause training

                with torch.no_grad():
                    for p_v, ((review, sentiment), _) in enumerate(v_loader):
                        loss = model_loss(model, sentiment, review)
                        valid_loss += loss.item()
                
                avg_valid_loss = valid_loss/len(v_loader)

                if best_loss > avg_valid_loss:
                    best_loss = avg_valid_loss

                    # save checkpoint
                    save_checkpoint(model, optimizer, epoch, avg_valid_loss)

                # reset validation loss
                valid_loss = 0.0

    save_checkpoint(model, optimizer, 0, 0, True)