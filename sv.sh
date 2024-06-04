bash train.sh 0,1\
    --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_mae_plusplus.yaml \
    --name mae_vit \
    --data-path data \
    --output-basedir ckpt
