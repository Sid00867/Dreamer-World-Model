from environment_variables import *
import torch
from fitter import rssmmodel 

previous_mean = None

def plan(h_t, s_t):
    global previous_mean
    rssmmodel.eval()
    
    with torch.no_grad():
        
        if previous_mean is not None:
            new_mean = torch.cat([
                previous_mean[1:], 
                torch.zeros((1, action_dim), device=DEVICE)
            ], dim=0)
            mean = new_mean
            std = torch.ones((planning_horizon, action_dim), device=DEVICE) * 0.5 
        else:
            mean = torch.zeros((planning_horizon, action_dim), device=DEVICE)
            std  = torch.ones((planning_horizon, action_dim), device=DEVICE)

        h_init = h_t.expand(candidates, -1).contiguous()   
        s_init = s_t.expand(candidates, -1).contiguous()  

        for _ in range(optimization_iters):
            candidates_actions_logits = torch.normal(
                mean.expand(candidates, -1, -1), 
                std.expand(candidates, -1, -1)
            ) 
            
            candidates_actions_prob = torch.softmax(candidates_actions_logits, dim=-1)

            #JIT Rollout
            rewards = rssmmodel.rollout_horizon(h_init, s_init, candidates_actions_prob)

            top_values, top_idx = torch.topk(rewards, K, dim=0)
            
            # refit based on the distribution parameters,
            # not the softmaxed probabs, to keep the Gaussian sampling valid.
            top_actions = candidates_actions_logits[top_idx]    

            # refit
            mean = top_actions.mean(dim=0)                
            std  = top_actions.std(dim=0) + 1e-5   

        previous_mean = mean          

        softmax_best_action = torch.softmax(mean[0], dim=-1)       
        best_action = torch.nn.functional.one_hot(softmax_best_action.argmax(dim=-1), num_classes=action_dim).float()
        return best_action