CUDA_VISIBLE_DEVICES=0,1,2,3  python coco_vit.py train
               --arch vitl16_384
               --fold {0, 1, 2, 3}  --batch_size 6
               --random-scale 2 --random-rotate 10 
               --lr 0.001  --lr-mode poly
               --benchmark pascal
CUDA_VISIBLE_DEVICES=0,1,2,3  python pascal_vit.py train
               --arch vitl16_384
               --fold {0, 1, 2, 3}  --batch_size 6
               --random-scale 2 --random-rotate 10 
               --lr 0.001  --lr-mode poly
               --benchmark pascal
CUDA_VISIBLE_DEVICES=0,1,2,3  python coco_drn.py train
               --arch drn_d_105
               --fold {0, 1, 2, 3}  --batch_size 6
               --random-scale 2 --random-rotate 10 --drate 0.9 
               --lr 0.001  --lr-mode poly
               --benchmark pascal
CUDA_VISIBLE_DEVICES=0,1,2,3  python pascal_drn.py train
               --arch drn_d_105
               --fold {0, 1, 2, 3}  --batch_size 6
               --random-scale 2 --random-rotate 10 --drate 0.95
               --lr 0.001  --lr-mode poly
               --benchmark pascal