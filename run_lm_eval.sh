



lm_eval --model hf \
    --model_args pretrained=/0203_ConceptLM_Pythia_410M \
    --tasks arc_easy,lambada,winogrande,piqa,sciq,race,hellaswag,wikitext,arc_challenge \
    --device cuda \
    --output_path results \
    --batch_size auto \




# lm_eval --model hf \
#     --model_args pretrained=/0203_ConceptLM_Llama31_8B \
#     --tasks arc_challenge \
#     --device cuda \
#     --num_fewshot 25 \
#     --output_path results \
#     --batch_size auto \



# lm_eval --model hf \
#     --model_args pretrained=/0203_ConceptLM_Llama31_8B \
#     --tasks mmlu \
#     --device cuda \
#     --num_fewshot 5 \
#     --output_path results \
#     --batch_size auto \


# lm_eval --model hf \
#     --model_args pretrained=/0203_ConceptLM_Llama31_8B \
#     --tasks agieval \
#     --device cuda \
#     --num_fewshot 3 \
#     --output_path results \
#     --batch_size auto \

# lm_eval --model hf \
#     --model_args pretrained=/0203_ConceptLM_Llama31_8B \
#     --tasks squadv2 \
#     --device cuda \
#     --num_fewshot 1 \
#     --output_path results \
#     --batch_size auto \
