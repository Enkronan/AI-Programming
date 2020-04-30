from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from helper import get_train_args, save_checkpoint
from test import test_model


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def create_datasets(train_dir, valid_dir, test_dir):
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=train_transforms),
        'test': datasets.ImageFolder(test_dir, transform=test_transforms),
        'valid': datasets.ImageFolder(valid_dir, transform=test_transforms)
    }
    
    return image_datasets
   
def create_loaders(image_datasets):
    
    loaders = {
        "trainloader": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "testloader": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64),
        "validloader": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64)
    }
    
    return loaders

def create_model(model, device, learn, layers):
    
    try:
        learn = float(learn)
    except:
        return "learn rate is not a float"
        
    if model == "vgg16":
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(nn.Linear(25088,layers),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(layers,layers),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(layers,102),
                                   nn.LogSoftmax(dim=1))
        print("vgg16")
        
    elif model == "densenet121":
        model = models.densenet121(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(nn.Linear(1024,layers),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(layers,layers),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(layers,102),
                                   nn.LogSoftmax(dim=1))
        print("densenet121")
    else:
        return "incorrect model. Try vgg16 or densenet121."
    
    if device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learn)
    model.to(device);
    
    return model, device, criterion, optimizer
    
    
def train_model(model, loaders, optimizer, device, criterion, epochs):
    

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in loaders["trainloader"]:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in loaders["validloader"]:
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
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/length:.3f}.. "
                      f"Test accuracy: {accuracy/length:.3f}")
                running_loss = 0
                model.train()
                
    return model



    
    
if __name__ == "__main__":
    in_arg = get_train_args()
    
    model, device, criterion, optimizer = create_model(in_arg.arch, in_arg.dev, in_arg.lr, in_arg.hid)
    
    datasets = create_datasets(in_arg.dir + "/train", in_arg.dir + "/valid", in_arg.dir + "test")
    
    loaders = create_loaders(datasets)
    
    trained_model = train_model(model, loaders, optimizer, device, criterion, in_arg.epo)
    
    test_model(model, loaders, device, criterion)
    
    save_checkpoint(trained_model, datasets, in_arg.arch, in_arg.hid)
    
