import argparse

parser = argparse.ArgumentParser('private_flocking')

# General settings
parser.add_argument('--device', type=str, default='cpu', help='device used for training')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')

# Training settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--distributed', action='store_true', default=False, help='true if using multi-GPU training')
parser.add_argument('--port', type=int, default=23333, help='distributed port')

parser.add_argument('--update_epochs', type=int, default=1, help='num of online update training epochs')

parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--bn_affine', action='store_true', default=False, help='update para in BatchNorm or not')

# Discriminator settings
parser.add_argument('--num_drones', type=int, default=9, help='num of drones used in the set of simulations')
parser.add_argument('--trajectory', type=str, default='Sinusoidal', help='trajectory type of the simulations')
parser.add_argument('--observ_window', type=int, default=5, help='length of the discriminator observation in seconds')
parser.add_argument('--downsampling', type=int, default=1, help='downsampling rate of the observation')
parser.add_argument('--slice_stride', type=int, default=1, help='ratio of stride/obsv when sliding observation window')
parser.add_argument('--multi_slice', action='store_true', default=False,
                    help='whether use multiple slices from a single simulation')
parser.add_argument('--save', type=str, default='../exp_logs', help='experiment log dir')

args = parser.parse_args()