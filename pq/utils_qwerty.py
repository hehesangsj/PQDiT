import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from pq.utils_model import init_data
from pq.utils_traineval import save_ckpt
from diffusion import create_diffusion

LINEAR_COMPENSATION_SAMPLES=100

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

class CompensationBlock(nn.Module):
    def __init__(self, block, side_net, device):
        super(CompensationBlock, self).__init__()
        self.block = block
        self.side_network = side_net.to(device)
    def forward(self, x, c):
        out = self.block(x, c)
        side_out = self.side_network(x)
        out = out + side_out
        return out

def generate_compensation_model(args, logger, model, model_pq, vae, checkpoint_dir):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    loader, sampler = init_data(args, rank, logger)
    diffusion_qwerty = create_diffusion(timestep_respacing="") 

    model.eval()
    model_pq.eval()
    if args.qwerty_ckpt:
        for block_id in range(len(model_pq.blocks)):
            model_pq.blocks[block_id] = CompensationBlock(block=model_pq.blocks[block_id], dim=1152, device=device)
        model_pq.load_state_dict(torch.load(args.qwerty_ckpt)['model'])
        model_pq.cuda()
    else:
        output_x = torch.zeros(size=[0,])
        output_c = torch.zeros(size=[0,])

        sampler.set_epoch(0)
        for i, (x, y) in tqdm(enumerate(loader)):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # t = torch.randint(0, 1, (x.shape[0],), device=device)
                t = torch.randint(0, diffusion_qwerty.num_timesteps, (x.shape[0],), device=device)
                noise = torch.randn_like(x)
                x_t = diffusion_qwerty.q_sample(x, t, noise=noise)
                x_t = model.x_embedder(x_t) + model.pos_embed
                t = model.t_embedder(t)
                y = model.y_embedder(y, False)
                c = t + y

            output_x = torch.cat([output_x, x_t.detach().cpu()], dim=0)
            output_c = torch.cat([output_c, c.detach().cpu()], dim=0)
            if i >= LINEAR_COMPENSATION_SAMPLES:
                break
        
        feature_set = FeatureDataset(output_x.detach().cpu(), output_c.detach().cpu())
        sampler = DistributedSampler(
            feature_set,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        feature_loader = DataLoader(
            feature_set,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        output_x_previous = output_x
        output_c_previous = output_c

        for block_id in range(len(model_pq.blocks)):
            feature_set.X = output_x_previous.detach().cpu()
            feature_set.Y = output_c_previous.detach().cpu()
            block_q = deepcopy(model_pq.blocks[block_id])
            block_origin = model.blocks[block_id]
            side_net = SideNetwork(1152, 4*1152, 1152).to(device)
            opt = torch.optim.AdamW(side_net.parameters(), lr=1e-4, weight_decay=0)
            criterion = nn.MSELoss()

            for i, (x_out, c_out) in tqdm(enumerate(feature_loader)):
                with torch.no_grad():
                    x_out = x_out.cuda()
                    c_out = c_out.cuda()
                    full_precision_out = block_origin(x_out, c_out).detach()
                    quant_out = block_q(x_out, c_out).detach()
                
                predict_diff = side_net(x_out)
                loss = criterion(predict_diff, full_precision_out - quant_out)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if i % 10 == 0:
                    with torch.no_grad():
                        loss_l1 = (full_precision_out - quant_out - predict_diff).abs().mean()
                    logger.info(f"[{block_id}/{len(model_pq.blocks)}] MSE Loss: {loss.item():.4f}, L1 Loss: {loss_l1.item():.4f}")
                if i >= LINEAR_COMPENSATION_SAMPLES:
                    break

            model_pq.blocks[block_id] = CompensationBlock(block=model_pq.blocks[block_id], side_net=deepcopy(side_net), device=device)
            model_pq.cuda()
            qwerty_block = model_pq.blocks[block_id]

            output_x_previous = torch.zeros(size=[0,])
            quant_error = []
            qwerty_error = []
            side_error = []
            for i, (x_out, c_out) in tqdm(enumerate(feature_loader)):
                with torch.no_grad():
                    x_out = x_out.cuda()
                    c_out = c_out.cuda()
                    previous_out = qwerty_block(x_out, c_out)
                    output_x_previous = torch.cat([output_x_previous, previous_out.detach().cpu()], dim=0)

                    quant_out = block_q(x_out, c_out)
                    full_precision_out = block_origin(x_out, c_out)
                    quant_error.append((quant_out - full_precision_out).abs().mean())
                    qwerty_error.append((previous_out - full_precision_out).abs().mean())

                    predict_diff = side_net(x_out)
                    side_error.append((full_precision_out - quant_out - predict_diff).abs().mean())
                    if i >= LINEAR_COMPENSATION_SAMPLES:
                        break
            quant_error = torch.Tensor(quant_error).mean()
            qwerty_error = torch.Tensor(qwerty_error).mean()
            side_error = torch.Tensor(side_error).mean()
            logger.info(f"[{block_id}/{len(model_pq.blocks)}] Quantization error: {quant_error.item():.4f}; \
                Qwerty error: {qwerty_error.item():.4f}; Side error: {side_error.item():.4f}")
        save_ckpt(model_pq, args, checkpoint_dir, logger)
