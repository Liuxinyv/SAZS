CUDA_VISIBLE_DEVICES=0  python coco_vit.py test
               --arch vitl16_384
               --fold {0, 1, 2, 3}  --batch_size 1 
               --benchmark coco
               --eig_dir ./datasets/COCO2014/coco_k5/
               --resume "path_to_trained_model/best_model.pt"
CUDA_VISIBLE_DEVICES=0  python pascal_vit.py test
               --arch vitl16_384
               --fold {0, 1, 2, 3}  --batch_size 1 
               --benchmark pascal
               --eig_dir ./datasets/VOC2012/pascal_k5/
               --resume "path_to_trained_model/best_model.pt"
CUDA_VISIBLE_DEVICES=0  python coco_drn.py test
               --arch drn_d_105
               --fold {0, 1, 2, 3}  --batch_size 1 
               --benchmark coco
               --eig_dir ./datasets/COCO2014/coco_k5/
               --resume "path_to_trained_model/best_model.pt"
CUDA_VISIBLE_DEVICES=0  python pascal_drn.py test
               --arch drn_d_105
               --fold {0, 1, 2, 3}  --batch_size 1 
               --benchmark pascal
               --eig_dir ./datasets/VOC2012/pascal_k5/
               --resume "path_to_trained_model/best_model.pt"