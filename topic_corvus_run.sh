#!/bin/bash



for num_epochs in 100
do
for batch_size in 250
do
for topic in  10
do
for phi_batch in  1
do
for beta_batch in  1 
do
for theta_batch in 0 
do
for frequency_limit in 25
do	
for num_units in 100
do	
for num_hidden in 500
do	
for dim_emb in 300
do	
for dropout in  1.0
do	
for mixture_lambda in 0.5
do	
for prior in 0.5
do
		printf "\n"
		# echo "num_topics ${topic} beta_batch ${beta_b} phi_batch ${phi_b} theta_batch ${theta_b} batch_size ${batch_size}  frequency_limit 100 num_epochs 200  dropout 0.6 max_seqlen 80 lambda 1"
		echo "--num_topics ${topic} --prior ${prior} --dropout ${dropout} --mixture_lambda ${mixture_lambda} --frequency_limit ${frequency_limit}  --num_hidden ${num_hidden}  --dim_emb ${dim_emb} --dropout ${dropout} --num_units ${num_units} --max_seqlen 50 --theta_batch ${theta_b}   --batch_size ${batch_size}  --num_epochs ${num_epochs} --beta_batch ${beta_b} --phi_batch ${phi_b} "
		printf "\n"
		python main.py \
		--batch_size ${batch_size} --num_topics ${topic} --frequency_limit ${frequency_limit} \
		--num_epochs ${num_epochs} --beta_batch ${beta_batch} --phi_batch ${phi_batch} \
		--dropout ${dropout} --max_seqlen 50  --theta_batch ${theta_batch} \
		--num_units ${num_units} --num_hidden ${num_hidden}  --dim_emb ${dim_emb} \
		--dropout ${dropout} --mixture_lambda ${mixture_lambda} --prior ${prior} 2>&1 | tee ${topic}_log_prior_${prior}_droput_${dropout}.txt 
done
done
done 
done
done 
done
done 
done
done 
done
done 
done
done 

