#!/bin/bash



for num_epochs in 50
do
for batch_size in 50
do
for topic in  50
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
for dataset in vist
do
for gen_len in 89
do	
for max_seqlen in 90
do	
		printf "\n"
		echo "--max_seqlen ${max_seqlen}  --dataset ${dataset} --num_topics ${topic} --prior ${prior} --dropout ${dropout} --mixture_lambda ${mixture_lambda} --frequency_limit ${frequency_limit}  --num_hidden ${num_hidden}  --dim_emb ${dim_emb} --dropout ${dropout} --num_units ${num_units} --max_seqlen 50 --theta_batch ${theta_b}   --batch_size ${batch_size}  --num_epochs ${num_epochs} --beta_batch ${beta_b} --phi_batch ${phi_b} "
		printf "\n"
		python main.py \
		--batch_size ${batch_size} --num_topics ${topic} --frequency_limit ${frequency_limit} \
		--num_epochs ${num_epochs} --beta_batch ${beta_batch} --phi_batch ${phi_batch} \
		--dropout ${dropout} --max_seqlen ${max_seqlen}  --theta_batch ${theta_batch} \
		--num_units ${num_units} --num_hidden ${num_hidden}  --dim_emb ${dim_emb} \
		--dropout ${dropout} --mixture_lambda ${mixture_lambda} --prior ${prior} --dataset ${dataset} --generate_len ${gen_len} 2>&1 | tee ${topic}_${dataset}_log_prior_${prior}_droput_GRU.txt 
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
done 
done
done

