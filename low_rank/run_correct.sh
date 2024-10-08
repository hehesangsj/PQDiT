CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 0 --fc-ind 1
python cal_correct.py --block-ind 0 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval


CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 1 --fc-ind 0
python cal_correct.py --block-ind 1 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 1 --fc-ind 1
python cal_correct.py --block-ind 1 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 2 --fc-ind 0
python cal_correct.py --block-ind 2 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 2 --fc-ind 1
python cal_correct.py --block-ind 2 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 3 --fc-ind 0
python cal_correct.py --block-ind 3 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 3 --fc-ind 1
python cal_correct.py --block-ind 3 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 4 --fc-ind 0
python cal_correct.py --block-ind 4 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 4 --fc-ind 1
python cal_correct.py --block-ind 4 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 5 --fc-ind 0
python cal_correct.py --block-ind 5 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 5 --fc-ind 1
python cal_correct.py --block-ind 5 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 6 --fc-ind 0
python cal_correct.py --block-ind 6 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 6 --fc-ind 1
python cal_correct.py --block-ind 6 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 7 --fc-ind 0
python cal_correct.py --block-ind 7 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 7 --fc-ind 1
python cal_correct.py --block-ind 7 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 8 --fc-ind 0
python cal_correct.py --block-ind 8 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 8 --fc-ind 1
python cal_correct.py --block-ind 8 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 9 --fc-ind 0
python cal_correct.py --block-ind 9 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 9 --fc-ind 1
python cal_correct.py --block-ind 9 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 10 --fc-ind 0
python cal_correct.py --block-ind 10 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 10 --fc-ind 1
python cal_correct.py --block-ind 10 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 11 --fc-ind 0
python cal_correct.py --block-ind 11 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 11 --fc-ind 1
python cal_correct.py --block-ind 11 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 12 --fc-ind 0
python cal_correct.py --block-ind 12 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 12 --fc-ind 1
python cal_correct.py --block-ind 12 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 13 --fc-ind 0
python cal_correct.py --block-ind 13 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 13 --fc-ind 1
python cal_correct.py --block-ind 13 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 14 --fc-ind 0
python cal_correct.py --block-ind 14 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 14 --fc-ind 1
python cal_correct.py --block-ind 14 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval


CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 15 --fc-ind 0
python cal_correct.py --block-ind 15 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 15 --fc-ind 1
python cal_correct.py --block-ind 15 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 16 --fc-ind 0
python cal_correct.py --block-ind 16 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 16 --fc-ind 1
python cal_correct.py --block-ind 16 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 17 --fc-ind 0
python cal_correct.py --block-ind 17 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 17 --fc-ind 1
python cal_correct.py --block-ind 17 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 18 --fc-ind 0
python cal_correct.py --block-ind 18 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 18 --fc-ind 1
python cal_correct.py --block-ind 18 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 19 --fc-ind 0
python cal_correct.py --block-ind 19 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 19 --fc-ind 1
python cal_correct.py --block-ind 19 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 20 --fc-ind 0
python cal_correct.py --block-ind 20 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 20 --fc-ind 1
python cal_correct.py --block-ind 20 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 21 --fc-ind 0
python cal_correct.py --block-ind 21 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 21 --fc-ind 1
python cal_correct.py --block-ind 21 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 22 --fc-ind 0
python cal_correct.py --block-ind 22 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 22 --fc-ind 1
python cal_correct.py --block-ind 22 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 23 --fc-ind 0
python cal_correct.py --block-ind 23 --fc-ind 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume swin/ffn_output_correct/swin_pca_all_uvb_correct.pth --batch-size 16  --eval --block-ind 23 --fc-ind 1
python cal_correct.py --block-ind 23 --fc-ind 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume /home/yuh/Swin-Transformer/swin/ffn_output_correct/swin_pca_all_uvb_correct.pth  --batch-size 32 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 1 --fc-ind 0
# python cal_correct.py --block-ind 1 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 1 --fc-ind 1
# python cal_correct.py --block-ind 1 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 2 --fc-ind 0
# python cal_correct.py --block-ind 2 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 2 --fc-ind 1
# python cal_correct.py --block-ind 2 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 3 --fc-ind 0
# python cal_correct.py --block-ind 3 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 3 --fc-ind 1
# python cal_correct.py --block-ind 3 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 4 --fc-ind 0
# python cal_correct.py --block-ind 4 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 4 --fc-ind 1
# python cal_correct.py --block-ind 4 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 5 --fc-ind 0
# python cal_correct.py --block-ind 5 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 5 --fc-ind 1
# python cal_correct.py --block-ind 5 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 6 --fc-ind 0
# python cal_correct.py --block-ind 6 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 6 --fc-ind 1
# python cal_correct.py --block-ind 6 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 7 --fc-ind 0
# python cal_correct.py --block-ind 7 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 7 --fc-ind 1
# python cal_correct.py --block-ind 7 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 8 --fc-ind 0
# python cal_correct.py --block-ind 8 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 8 --fc-ind 1
# python cal_correct.py --block-ind 8 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 9 --fc-ind 0
# python cal_correct.py --block-ind 9 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 9 --fc-ind 1
# python cal_correct.py --block-ind 9 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 10 --fc-ind 0
# python cal_correct.py --block-ind 10 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 10 --fc-ind 1
# python cal_correct.py --block-ind 10 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 11 --fc-ind 0
# python cal_correct.py --block-ind 11 --fc-ind 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12338  main_eval_correction.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt --batch-size 16  --eval --block-ind 11 --fc-ind 1
# python cal_correct.py --block-ind 11 --fc-ind 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12340  main_eval.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/  --resume resmlp_12/ffn_output_correct/resmlp_12_no_dist_pca_fewshot2_train_all_uvb_correct.pt  --batch-size 64 --tag lr5e-5_wodrop  --eval