# Fast Neural Style Transfer with Batch Normalization/ Instance Normalization

import numpy as np 
import os
import sys
import time
import re
import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import cv2

# Import the Image Transform Net, VGG pre-trained Network
from imageTransformNet import TransformNet_BN
from imageTransformNet import TransformNet_IN
from vgg16 import Vgg16
from vgg19 import Vgg19


# Helper Functions
def load_image(filename, size=None, scale=None):
    '''
    To Load image w/o size specification or scaling factor
    '''
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def load_dataset(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    data = datasets.ImageFolder(args.dataset, transform)
    data_count = len(data)
    loader = DataLoader(data, batch_size=args.batch_size)
    return data_count, loader
    

def train(args):
    # To select between "cpu" and "gpu"
    device = torch.device("cuda" if args.cuda else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # To load the dataset using the DataLoader Module
    train_count, train_loader = load_dataset(args)
    # To select the Image Transfrom Net:  with Batch Normalization (BN) / Instance Normalization (IN)
    if args.normalization == 'Instance':
        transformer = TransformNet_IN().to(device)
    else:
        transformer = TransformNet_BN().to(device)
    # To select the pre-trained VGG model (vgg16 / vgg19)
    if (args.vgg_model == 'vgg16'):
        vgg = Vgg16(requires_grad=False).to(device)
        print('\nSelect Vgg16')
    else:
       vgg = Vgg19(requires_grad=False).to(device)
       print('\nSelect Vgg19')

    # Load the style image 
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = style_transform(load_image(args.style_image, size=args.style_size)).repeat(args.batch_size, 1, 1, 1)
    style = style.to(device)

    # Compute gram matrix of the style features
    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    # Traning Starts!!!
    print("Training Starts!!!")
    # Use the Adam for optimization by providing the learning rate(lr)
    optimizer = Adam(transformer.parameters(), args.lr)
    # Define the loss function
    mse_loss = torch.nn.MSELoss()

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = Variable(x)
            # Forward pass
            # y = transformer(x)
            y_hat = transformer(x.to(device))
            
            y_hat = normalize_batch(y_hat)
            x = normalize_batch(x)
            # To extract the features using VGG pre-trained Network
            features_y_hat = vgg(y_hat)
            features_x = vgg(x)
            # To Calculate the Content Loss
            content_loss = args.content_weight * mse_loss(features_y_hat.relu2_2, features_x.relu2_2)
            # To Calculate the Style Loss
            style_loss = 0.
            for ft, gm_s in zip(features_y_hat, gram_style):
                gm = gram_matrix(ft)
                style_loss += mse_loss(gm, gm_s[:n_batch, :, :]) * args.style_weight
            # To Combine as Total Loss to be minimized in optimization
            total_loss = content_loss + style_loss
            # Backpropagation
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "Epoch {}: [{}/{}]  content loss: {:.2f}  style loss: {:.2f}  total loss: {:.2f}".format(
                        e + 1,                                                    # Current Epoch Number
                        count,                                                    # Finished samples in current epoch
                        train_count,                                              # Total training samplings
                        agg_content_loss / (batch_id + 1),                        # Content Loss
                        agg_style_loss / (batch_id + 1),                          # Style Loss
                        (agg_content_loss + agg_style_loss) / (batch_id + 1)      # Total Loss
                )
                print(mesg)

    # Save the model after Training            
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nTraining ", save_model_path)

    
    
def stylize(args):
    # To select between "cpu" and "gpu"
    device = torch.device("cuda" if args.cuda else "cpu")

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(load_image(args.content_image, scale=args.content_scale))
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        state_dict = torch.load(args.model)
        if args.normalization == 'Instance':
            style_model = TransformNet_IN()
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
        else:
            style_model = TransformNet_BN()
            
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        print("Start stylize image . . .")
        output = style_model(content_image).cpu()
    # Save the Stylized Image
    img = output[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(args.output_image)
    print("Stylized output generated . . .")
    
    

def stylize_live(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    def transform_live(x):
        content_image = x 
        content_image = Image.fromarray(content_image)

        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            state_dict = torch.load(args.model)
            if args.normalization == 'Instance':
                style_model = TransformNet_IN()
                # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
                for k in list(state_dict.keys()):
                    if re.search(r'in\d+\.running_(mean|var)$', k):
                        del state_dict[k]
            else:
                style_model = TransformNet_BN()
                
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            print("Start stylize image . . .")
            output = style_model(content_image).cpu()
        # Save the Stylized Image
        img = output[0].clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        return img
    
    # use CV2
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_changed =  transform_live(frame)
        out.write(frame_changed)

        # Display the resulting frame
        cv2.imshow('Live_Model',frame_changed)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--normalization", type=str, default='Instance',
                                  help="the normalization method used in the Image Transform Net")
    train_arg_parser.add_argument("--vgg_model", type=str, default='vgg16',
                                  help="the pre-trained VGG16 or VGG19 model")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image.")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--normalization", type=str, default='Instance',
                                  help="the normalization method used in the Image Transform Net")

    # live model 
    live_arg_parser = subparsers.add_parser("live", help="parser for live-model arguments")
    live_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    live_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    live_arg_parser.add_argument("--normalization", type=str, default='Instance',
                                  help="the normalization method used in the Image Transform Net")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "eval":
        stylize(args)
    elif args.subcommand == "live":
        stylize_live(args)


if __name__ == "__main__":
    main()