
python main.py --model DialogPrompt --model_size base --do_train --do_validate --do_test  --num_trig_tokens 5 --learning_rate 1e-3 --max_steps 200000 --warmup_steps 5000 --validating_steps 2000 --start_eval 10000 --fast_eval_ratio 1.0 

### fewshot (variant=MLP, prompt size=5, data size vs. lr = 1000:5e-5, 500:1e-5, 200:5e-6, 100:1e-6) 
python main.py --model DialogPrompt --model_size base --fewshot 100 --do_train --do_validate --do_test  --num_trig_tokens 5 --learning_rate 1e-6 --max_steps 2000 --warmup_steps 100 --validating_steps 100 --start_eval 100 --fast_eval_ratio 1.0
  
