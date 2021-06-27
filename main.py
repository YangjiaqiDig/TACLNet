from modules import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# logger = logging.getLogger(__file__).setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)


def train():
    database = "CREMI"
    parser = ArgumentParser()
    parser.add_argument("--database", type=str, default="{0}".format(database),
                        help="Hepatic, CREMI, ISBI12, ISBI13")
    parser.add_argument("--train_type", type=str, default="clstm",
                        help="unet, unet-clstm, or clstm")
    parser.add_argument("--topo_attention", type=bool, default=True,
                        help="Add topo attention loss to train")
    parser.add_argument("--topo_iteration", type=bool, default=True,
                        help="Add topo attention loss to train")
    parser.add_argument("--crop", type=bool, default=True,
                        help="Need crop for large dataset")
    parser.add_argument("--dataset_path_train", type=str, default="database/{0}/train-volume.tif".format(database),
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_path_label", type=str, default="database/{0}/train-labels.tif".format(database),
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_cache", type=str,
                        default='dataset_cache/dataset_cache_{0}'.format(database), help="Path or url of the preprocessed dataset cache")
    parser.add_argument("--save_folder", type=str, default="results_clstm/{0}_5step".format(database),
                        help="Path or url of the dataset")
    # TODO: batch size enlarge, need fit the total number of input, dividable
    parser.add_argument("--train_batch_size", type=int,
                        default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=10, help="Batch size for validation")
    parser.add_argument("--valid_round", type=int,
                        default=1, help="validation part: 1, 2, 3")
    parser.add_argument("--lr", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--lr_topo", type=float,
                        default=0.00001, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--check_point", type=str, default="/model_epoch_350.pwf",
                        help="Path of the pre-trained CNN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
    else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--topo_size", type=int, default=39, help="Crop size for topo input")

    parser.add_argument("--step_size", type=int, default=5, help="sequence length for LSTM")

    args = parser.parse_args()


    if args.train_type == 'clstm':
        logging.info("---------Prepare DataSet for CLSTM--------")
        trainDataset, validDataset = get_dataset_topoClstm(args)
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=8,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset=validDataset, num_workers=8, batch_size=args.valid_batch_size,
                                                 shuffle=False)

        train_LSTM_TopoAttention(train_loader, val_loader, args)
        
if __name__ == "__main__":
    train()
