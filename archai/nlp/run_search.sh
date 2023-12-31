## mem_transformer, titanXP
## run search 
# python archai/nlp/search.py --n_iter 3 --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
#         --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp
## profile the baseline n_layer scaling experiment 
# python archai/nlp/search.py --profile_baseline --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
#         --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp
## select the pareto 
# python archai/nlp/search.py --select_pareto --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
#         --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp
## plot baseline versus selected pareto (after training)
python archai/nlp/search.py --plot_pareto_baseline --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp --dataset wt103
## generate tables for the paper listing the architectures on the selected pareto
# python archai/nlp/search.py --gen_tables --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --div_val 4 \
#         --param_constraint_lower 4000000 --param_constraint_upper 100000000 --device cpu --device_name arm --dataset lm1b

## hf_gpt2_flex, titanXP
## run search 
# python archai/nlp/search.py --n_iter 30 --model_type hf_gpt2_flex --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --n_head 2 4 8 --div_val 4 \
#         --param_constraint_lower 6000000 --param_constraint_upper 100000000 --device cuda --device_name titanxp \
#         --vocab_type gpt2 --vocab_size 50257
## profile the baseline n_layer scaling experiment 
# python archai/nlp/search.py --profile_baseline --model_type hf_gpt2_flex --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --n_head 2 4 8 12 --div_val 4 \
#         --param_constraint_lower 6000000 --param_constraint_upper 100000000 --device cpu --device_name corei7 \
#         --vocab_type gpt2 --vocab_size 50257
# # select the pareto 
# python archai/nlp/search.py --select_pareto --model_type hf_gpt2_flex --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --n_head 2 4 8 --div_val 4 \
#         --param_constraint_lower 6000000 --param_constraint_upper 100000000 --device cpu --device_name arm \
#         --vocab_type gpt2 --vocab_size 50257
# plot baseline versus selected pareto (after training)
# python archai/nlp/search.py --plot_pareto_baseline --model_type hf_gpt2_flex --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --n_head 2 4 8 --div_val 4 \
#         --param_constraint_lower 6000000 --param_constraint_upper 100000000 --device cpu --device_name arm \
#         --vocab_type gpt2 --vocab_size 50257 --dataset wt103
# generate tables for the paper listing the architectures on the selected pareto
# python archai/nlp/search.py --gen_tables --model_type hf_gpt2_flex --n_layer 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --n_head 2 4 8 12 --div_val 4 \
#         --param_constraint_lower 6000000 --param_constraint_upper 100000000 --device cpu --device_name arm \
#         --vocab_type gpt2 --vocab_size 50257 --dataset lm1b