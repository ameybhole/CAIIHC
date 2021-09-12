import argparse
import numpy as np
from src.data_augmentation import Augmentation
from src.data_load import load_images, binarize_labels
from src.evaluation import model_predictions, printWrongPredictions
from src.models import Models
from src.train import train_test_split, train_model
from src.visualizations import visualize_scatter_with_images, tsne, plot_data_graph

if __name__ == '__main__':

    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--mode', type=str, default="train", help='Select "train", or "augment" mode')
    # parser.add_argument('--continue_training', type=str2bool, default=False,
    #                     help='Whether to continue training from a checkpoint')
    parser.add_argument('--dataset', type=str, default="./data", help='Dataset you are using.')
    parser.add_argument('--resize', type=int, default=224,
                        help='Height of cropped input image to network')
    parser.add_argument('--tsne', type=str2bool, default=False,
                        help='Plot tsne visualization')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of images in each batch')
    parser.add_argument('--classes', type=int, default=136,
                        help='Number of images in each batch')
    parser.add_argument('--trainable', type=str2bool, default=True,
                        help='Whether to train all the layers')
    parser.add_argument('--pretrained_dataset', type=str, default='imagenet',
                        help='specify which pre-trained dataset to use')
    parser.add_argument('--pooling', type=str, default='avg',
                        help='specify which pooling to use')
    parser.add_argument('--activation', type=str, default='softmax',
                        help='specify which pooling to use')
    parser.add_argument('--include_top', type=str2bool, default=False,
                        help='specify if to use the top layer of pre-trained model')
    parser.add_argument('--flip', type=str2bool, default=False,
                        help='Whether to randomly flip the image horizontally for data '
                             'augmentation')
    parser.add_argument('--rotation_left', type=float, default=-10,
                        help='Whether to randomly rotate left the image for data augmentation')
    parser.add_argument('--rotation_right', type=float, default=-10,
                        help='Whether to randomly rotate right the image for data augmentation')
    parser.add_argument('--bright', type=float, default=1.5,
                        help='Whether to randomly rotate right the image for data augmentation')
    parser.add_argument('--dark', type=float, default=0.5,
                        help='Whether to randomly rotate right the image for data augmentation')
    parser.add_argument('--gaussian_nosie', type=str2bool, default=False,
                        help='Whether to add gaussian noise to image for data '
                             'augmentation')
    parser.add_argument('--model', type=str, default="resnet50",
                        help='Your pre-trained classification model of choice')
    args = parser.parse_args()

    list_of_image_paths, labels, labels_list = load_images(args.dataset, args.resize)
    binarizelabels = binarize_labels(labels)
    skf = train_test_split(5)
    all_fold_accuracy = []
    all_fold_loss = []
    model = Models(args.trainable, args.classes, args.pretrained_dataset, args.include_top,
                   args.pooling, args.activation)

    if args.model == 'resnet50':
        model.resnet50()
    elif args.model == 'vgg16':
        model.vgg16()
    elif args.model == 'facenet':
        model.facenet()
    elif args.model == 'vggface':
        model.vggface()

    model.model_trainable()
    complied_model = model.model_compile()

    if args.tsne:
        tsne_result_scaled, float_labels = tsne(list_of_image_paths, labels_list)
        visualize_scatter_with_images(tsne_result_scaled, list_of_image_paths)

    for train_index, test_index in skf.split(list_of_image_paths, binarizelabels):
        X_train, X_test = list_of_image_paths[train_index], list_of_image_paths[test_index]
        y_train, y_test = binarizelabels[train_index], binarizelabels[test_index]

        if args.mode == "train":
            model, hist, loss, accuracy = train_model(complied_model, X_train,
                                                      y_train, args.batch_size, args.num_epochs,
                                                      X_test, y_test, args.model)

            plot_data_graph(hist, args.num_epochs, args.model)
            pred = model_predictions(model, X_test)
            printWrongPredictions(pred, y_test, labels)

            all_fold_accuracy.append(accuracy * 100)
            all_fold_loss.append(loss)

        elif args.mode == "augment":
            augmentations = Augmentation(args.rotation_left, args.rotation_right, args.bright,
                                         args.dark)

            augmented_data = augmentations.process_data(X_train)

            X_train_augmented = np.concatenate((X_train, augmented_data), axis=0)
            y_train_augmented = np.concatenate((y_train, y_train), axis=0)

            model, hist, loss, accuracy = train_model(complied_model, X_train_augmented,
                                                      y_train_augmented, args.batch_size,
                                                      args.num_epochs,
                                                      X_test, y_test, args.model)

            plot_data_graph(hist, args.num_epochs)
            pred = model_predictions(model, X_test, labels)
            printWrongPredictions(pred, y_test)

            all_fold_accuracy.append(accuracy * 100)
            all_fold_loss.append(loss)





