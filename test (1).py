import torch

def test_model(model, loaders, device, criterion):
    # TODO: Do validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loaders["testloader"]:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    length = len(loaders["testloader"])
    print(f"Test loss: {test_loss/length:.3f}.. "
          f"Test accuracy: {accuracy/length:.3f}")
    running_loss = 0
    model.train()