from helper import load_checkpoint, process_image, create_labels, get_predict_args
import torch
import torch.nn.functional as F

def predict(img, model, dev="gpu", topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        if dev == "gpu" and torch.cuda.is_available():
            output = model.forward(img.cuda())
        else:
            output = model.forward(img.cpu())
        
    result = F.softmax(output.data,dim=1)
    probabilities, labels = result.topk(topk)
    
    if dev == "gpu" and torch.cuda.is_available():
        probabilities = probabilities.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
    else:
        probabilities = probabilities.data.numpy()
        labels = labels.data.numpy()

    return probabilities[0], labels[0]


def print_results(probabilities, labels):
    
    for i in range(len(probabilities)):    
        print(f"Predicted Flower: {labels[i]}, "
              f"Predicted Probability: {probabilities[i]:.3f}")


if __name__ == "__main__":
    in_arg = get_predict_args()
    
    model = load_checkpoint(in_arg.check)
    
    img = process_image(in_arg.img)
    
    probabilities, labels = predict(img, model, in_arg.dev)
    
    labels = create_labels(labels, in_arg.json)
    
    print_results(probabilities, labels)
    
    
    
    