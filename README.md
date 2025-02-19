

## Run 
### CIFAR-80

```
python Train_cifar.py --dataset cifar100 --num_class 80 --noise_mode sym --r 0.2 --lambda_u 25 --gpuid 0

python Train_cifar.py --dataset cifar100 --num_class 80 --noise_mode sym --r 0.8 --lambda_u 150 --gpuid 0

python Train_cifar.py --dataset cifar100 --num_class 80 --noise_mode asym --r 0.4 --lambda_u 25 --gpuid 0
```


### CIFAR-100

```
python Train_cifar.py --dataset cifar100 --num_class 100 --noise_mode sym --r 0.2 --lambda_u 25 --gpuid 0

python Train_cifar.py --dataset cifar100 --num_class 100 --noise_mode sym --r 0.8 --lambda_u 150 --gpuid 0

python Train_cifar.py --dataset cifar100 --num_class 100 --noise_mode asym --r 0.4 --lambda_u 25 --gpuid 0
``` 
