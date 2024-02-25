#base
device_id=0
seed=15
epochs=200
d_1=0.2
d_2=0.3
knn=5
lambda_1=2.0
tau=0.5
attack_method=meta
#cora meta posion
dataset=cora
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --dataset $dataset --ptb_rate $p --attack_type poison --attack_method $attack_method \
--device_id $device_id --seed $seed --epochs $epochs --d_1 $d_1 --d_2 $d_2 --knn $knn --tau $tau --lambda_1 $lambda_1
done


#citeseer meta posion
dataset=citeseer
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --dataset $dataset --ptb_rate $p --attack_type poison --attack_method $attack_method \
--device_id $device_id --seed $seed --epochs $epochs --d_1 $d_1 --d_2 $d_2 --knn $knn --tau $tau --lambda_1 $lambda_1
done

#pubmed meta posion
dataset=pubmed
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --dataset $dataset --ptb_rate $p --attack_type poison --attack_method $attack_method \
--device_id $device_id --seed $seed --epochs $epochs --d_1 $d_1 --d_2 $d_2 --knn $knn --tau $tau --lambda_1 $lambda_1
done




#p=0.0
##cora meta evasive
#dataset=cora
#python main.py --dataset $dataset --ptb_rate $p --attack_type evasive --attack_method $attack_method \
#--device_id $device_id --seed $seed --epochs $epochs --d_1 $d_1 --d_2 $d_2 --knn $knn --tau $tau --lambda_1 $lambda_1
##citeseer meta evasive
#dataset=citeseer
#python main.py --dataset $dataset --ptb_rate $p --attack_type evasive --attack_method $attack_method \
#--device_id $device_id --seed $seed --epochs $epochs --d_1 $d_1 --d_2 $d_2 --knn $knn --tau $tau --lambda_1 $lambda_1
##pubmed meta evasive
#dataset=pubmed
#python main.py --dataset $dataset --ptb_rate $p --attack_type evasive --attack_method $attack_method \
#--device_id $device_id --seed $seed --epochs $epochs --d_1 $d_1 --d_2 $d_2 --knn $knn --tau $tau --lambda_1 $lambda_1

