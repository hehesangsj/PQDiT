import os
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


from pq.utils_model import init_data
from pq.utils_traineval import save_ckpt, dit_generator
from diffusion import create_diffusion

LINEAR_COMPENSATION_SAMPLES=1000

class FeatureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]
    

class SideNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SideNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out


def lienar_regression(X, Y, block_id=0, logger=None):
    X = X.reshape(-1, X.size(-1)).cuda()
    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1)).cuda()
    logger.info('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))
    W_overall = torch.linalg.lstsq(X_add_one.cpu(), Y.cpu()).solution.cuda()

    W = W_overall[:-1, :]
    b = W_overall[-1, :]
    Y_pred = X @ W + b
    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot
    logger.info(f'block: {block_id}, bs: {abs_loss:.6f}, r2: {r2_score:.3f}')

    return W, b, r2_score

class CompensationBlock(nn.Module):
    def __init__(self, block, side_net, r2_scores, device):
        super(CompensationBlock, self).__init__()
        self.block = block
        self.side_network = side_net.to(device)
        alpha_value = 1 if r2_scores >= 0 else 0
        self.register_buffer('alpha', torch.tensor(alpha_value, dtype=torch.float32))

    def forward(self, x, c):
        out = self.block(x, c)
        side_out = self.side_network(x)
        out = out + side_out * self.alpha
        return out

def generate_compensation_model(args, logger, model, model_pq, vae, checkpoint_dir):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    loader, sampler = init_data(args, rank, logger)
    diffusion_qwerty = create_diffusion(timestep_respacing="")
    latent_size = args.image_size // 8
    diffusion_gen = dit_generator('250', latent_size=latent_size, device=device)

    model.eval()
    model_pq.eval()
    
    if args.qwerty_ckpt:
        for block_id in range(len(model_pq.blocks)):
            side_net = SideNetwork(1152, 4*1152, 1152)
            model_pq.blocks[block_id] = CompensationBlock(block=model_pq.blocks[block_id], side_net=deepcopy(side_net), r2_scores=1, device=device)
        model_pq.load_state_dict(torch.load(args.qwerty_ckpt)['model'])
        model_pq.cuda()
    else:
        output_x = torch.zeros(size=[0,]).to(device)
        output_c = torch.zeros(size=[0,]).to(device)

        sampler.set_epoch(0)
        for i, (x, y) in tqdm(enumerate(loader), desc="Sampling"):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                t = torch.randint(0, diffusion_qwerty.num_timesteps, (x.shape[0],), device=device)
                noise = torch.randn_like(x)
                x_t = diffusion_qwerty.q_sample(x, t, noise=noise)
                x_t = model.x_embedder(x_t) + model.pos_embed
                t = model.t_embedder(t)
                y = model.y_embedder(y, False)
                c = t + y
            output_x = torch.cat([output_x, x_t.detach()], dim=0)
            output_c = torch.cat([output_c, c.detach()], dim=0)
            if i >= LINEAR_COMPENSATION_SAMPLES:
                break
        
        feature_set = FeatureDataset(output_x.detach().cpu(), output_c.detach().cpu())
        feature_loader = DataLoader(feature_set, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        for block_id in range(len(model_pq.blocks)):
            feature_set.X = output_x.detach().cpu()
            feature_set.Y = output_c.detach().cpu()
            block_q = deepcopy(model_pq.blocks[block_id])
            block_origin = model.blocks[block_id]
            
            if args.qwerty_mode == 'mlp':
                side_net = SideNetwork(1152, 4 * 1152, 1152).to(device)
                side_net = DDP(side_net, device_ids=[device])
                opt = torch.optim.AdamW(side_net.parameters(), lr=1e-5, weight_decay=0.02)
                criterion = nn.MSELoss()
            elif args.qwerty_mode == 'linear':
                output_full_precision = torch.zeros(size=[0,])
                output_quant = torch.zeros(size=[0,])
                output_t_ = torch.zeros(size=[0,])
            elif args.qwerty_mode == 'distill':
                block_distill = deepcopy(model_pq.blocks[block_id]).to(device)
                block_distill.train()
                block_distill = DDP(block_distill, device_ids=[device])
                opt = torch.optim.AdamW(block_distill.parameters(), lr=1e-5, weight_decay=0.02)
                criterion = nn.MSELoss()

            for epoch in range(args.epochs):
                for i, (x_out, c_out) in tqdm(enumerate(feature_loader), desc=f"Training block {block_id}"):
                    with torch.no_grad():
                        x_out = x_out.cuda()
                        c_out = c_out.cuda()
                        full_precision_out = block_origin(x_out, c_out).detach()
                        quant_out = block_q(x_out, c_out).detach()

                    if args.qwerty_mode == 'mlp':
                        predict_diff = side_net(x_out)
                        gt_diff = full_precision_out - quant_out
                        loss = criterion(predict_diff, gt_diff)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        with torch.no_grad():
                            loss_l1 = (full_precision_out - quant_out - predict_diff).detach().abs().mean()
                            ss_tot = torch.sum((gt_diff - gt_diff.mean(dim=0)).pow(2))
                            ss_res = torch.sum((gt_diff - predict_diff).pow(2))
                            r2_score = 1 - ss_res / ss_tot
                        if i % 100 == 0 and rank == 0:
                            logger.info(f"[{block_id}/{len(model_pq.blocks)}, {epoch}/{args.epochs}] MSE Loss: {loss.item():.4f}, L1 Loss: {loss_l1.item():.4f}, r2 score: {r2_score.item():.4f}")

                    elif args.qwerty_mode == 'linear':
                        output_t_ = torch.cat([output_t_, x_out.detach().cpu()], dim=0)
                        output_full_precision = torch.cat([output_full_precision, full_precision_out.detach().cpu()], dim=0)
                        output_quant = torch.cat([output_quant, quant_out.detach().cpu()], dim=0)
                    
                    elif args.qwerty_mode == 'distill':
                        distill_out = block_distill(x_out, c_out)
                        loss = criterion(distill_out, full_precision_out.cuda())
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        if i % 100 == 0 and rank == 0:
                            logger.info(f"[{block_id}/{len(model_pq.blocks)}, {epoch}/{args.epochs}] MSE Loss: {loss.item():.4f}")

            if args.qwerty_mode == 'mlp':
                dist.all_reduce(r2_score, op=dist.ReduceOp.SUM)
                r2_score = r2_score / dist.get_world_size()
                model_pq.blocks[block_id] = CompensationBlock(block=model_pq.blocks[block_id], side_net=deepcopy(side_net.module), r2_scores=r2_score, device=device)
                model_pq.cuda()
            elif args.qwerty_mode == 'linear':
                W, b, r2_score = lienar_regression(output_t_, output_full_precision - output_quant, block_id=block_id, logger=logger)
                in_features = W.shape[1]
                out_features = W.shape[0]
                side_net = nn.Linear(in_features, out_features)
                with torch.no_grad():
                    side_net.weight.copy_(W)
                    side_net.bias.copy_(b)
                model_pq.blocks[block_id] = CompensationBlock(block=model_pq.blocks[block_id], side_net=deepcopy(side_net), r2_scores=r2_score, device=device)
                model_pq.cuda()
                side_net.cuda()
            elif args.qwerty_mode == 'distill':
                model_pq.blocks[block_id] = deepcopy(block_distill.module)
                model_pq.cuda()

            output_x_previous = torch.zeros(size=[0,]).to(device)
            quant_error = []
            qwerty_error = []
            side_error = []
            for i, (x_out, c_out) in tqdm(enumerate(feature_loader), desc=f"Evaluating block {block_id}"):
                with torch.no_grad():
                    x_out = x_out.cuda()
                    c_out = c_out.cuda()
                    previous_out = model_pq.blocks[block_id](x_out, c_out)
                    output_x_previous = torch.cat([output_x_previous, previous_out.detach()], dim=0)

                    quant_out = block_q(x_out, c_out)
                    full_precision_out = block_origin(x_out, c_out)
                    quant_error.append((quant_out - full_precision_out).abs().mean())
                    qwerty_error.append((previous_out - full_precision_out).abs().mean())

                    if args.qwerty_mode == 'distill':
                        side_error.append((previous_out - full_precision_out).abs().mean())
                    else:
                        predict_diff = side_net(x_out)
                        side_error.append((full_precision_out - quant_out - predict_diff).abs().mean())

            quant_error = torch.Tensor(quant_error).mean()
            qwerty_error = torch.Tensor(qwerty_error).mean()
            side_error = torch.Tensor(side_error).mean()
            if rank == 0:
                logger.info(f"[{block_id}/{len(model_pq.blocks)}] Quantization error: {quant_error.item():.4f}; Qwerty error: {qwerty_error.item():.4f}; Side error: {side_error.item():.4f}")
            
            if rank == 0:
                image_name = f"pq_qwerty_block{block_id}"
                image_dir = f"{os.path.dirname(checkpoint_dir)}/gen"
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                diffusion_gen.forward_val(vae, model.forward, model_pq.forward, cfg=False, name=f"{image_dir}/{image_name}", logger=logger)
                
    if rank == 0:
        save_ckpt(model_pq, args, checkpoint_dir, logger)