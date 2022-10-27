import os

import sys
sys.path.append('../..')
from train_transformer_encoder import train_encoder
from train_transformer_vpg import vpg





def train(args):
	for iters in range(args.num_iters):
		# training the encoder with RL policy
		train_encoder(input_dim=args.input_dim, steps_per_epoch=args.steps_per_epoch, 
			n_env=args.n_env, seed=args.seed, map_path=args.map_path, pooling=args.pooling, 
			alpha=args.alpha, beta=args.beta, batch_size=args.batch_size,
			eval_steps=args.eval_steps, epochs=args.encoder_epochs, model_path=args.encoder_model_path, 
			device=args.device, d_model=args.d_model, n_head=args.n_head,
			resume=args.resume_encoder, pretrained=args.pretrained_encoder, exp_name=args.encoder_exp_name, 
			train_shape=args.train_shape, eval_shape=args.eval_shape)

		# training the policy with fixed encoder
		vpg(gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps, epochs=args.rl_epochs,
			model_path=args.rl_model_path, exp_name=args.rl_exp_name, save_freq=args.save_freq, 
			d_model=args.d_model, n_head=args.n_head, device=args.device, 
			encoder_path=args.encoder_model_path, actor_type=args.actor_type, input_dim=args.input_dim,
			map_path=args.map_path, train_shape=args.train_shape, eval_shape=args.eval_shape,
			pooling=args.pooling, alpha=args.alpha, beta=args.beta)




if __name__ == '__main__':
	# !!!!!!!!!
	root = '~/models/demos'

	import argparse
	parser = argparse.ArgumentParser()

	# args from iterative training
	parser.add_argument('--seed', '-s', type=int, default=0)
	parser.add_argument('--alpha', type=float, default=2.0)
	parser.add_argument('--beta', type=float, default=0.1)
	parser.add_argument('--pooling', type=bool, default=True)
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--d_model', type=int, default=16)
	parser.add_argument('--n_head', type=int, default=4)
	parser.add_argument('--input_dim', type=int, default=3)
	parser.add_argument('--train_shape', type=str, default='square')
	parser.add_argument('--eval_shape', type=str, default='triangle')
	parser.add_argument('--map_path', type=str, 
		default=os.path.join(root, 'map/0717'))
	parser.add_argument('--num_iters', type=int, default=2)
	parser.add_argument('--encoder_model_path', type=str,
		default=os.path.join(root, 'encoder/0727_pca'))


	# args from encoder
	parser.add_argument('--encoder_exp_name', type=str, default='encoder')
	parser.add_argument('--encoder_epochs', type=int, default=2000000000000)
	parser.add_argument('--resume_encoder', type=bool, default=False)
	parser.add_argument('--pretrained_encoder', type=str,
		default=os.path.join(root, 'encoder/0726/model_state_dict.pt'))
	parser.add_argument('--n_env', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--steps_per_epoch', type=int, default=10000)
	parser.add_argument('--eval_steps', type=int, default=1000)


	# args from RL policy
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--rl_epochs', type=int, default=5000000000)
    parser.add_argument('--rl_exp_name', type=str, default='vpg')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--rl_model_path', type=str,
        default=os.path.join(root, 'ppo/0726'))
    parser.add_argument('--actor_type', type=str, default='Categorical')
    args = parser.parse_args()


	from utils.run_utils import setup_logger_kwargs
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, 
		data_dir=args.model_path)
	logger = EpochLogger(**logger_kwargs)

	encoder = TransformerEncoder(input_dim=args.input_dim, d_model=args.d_model, n_head=args.n_head).to(args.device)
	if args.resume_encoder:
		encoder.load_state_dict(torch.load(args.pretrained_encoder, map_location=args.device))
		print('Resume training !!!')
	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(encoder.parameters(), lr=1e-4)
	writer = SummaryWriter(args.model_path)

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	train(args)








