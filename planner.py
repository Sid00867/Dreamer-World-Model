from environment_variables import *
import torch
from fitter import rssmmodel 

def plan(h_t, s_t):

    rssmmodel.eval()
    with torch.no_grad():
        mean = torch.zeros((planning_horizon, action_dim), device=DEVICE)
        std  = torch.ones((planning_horizon, action_dim), device=DEVICE)
        for _ in range(optimization_iters):
            candidates_actions = torch.normal(
                mean.expand(candidates, -1, -1), 
                std.expand(candidates, -1, -1)
            )  # (candidates, planning_horizon, action_dim)

            h = h_t.expand(candidates, -1).contiguous()    # (J, D)
            s = s_t.expand(candidates, -1).contiguous()    # (J, Z)
            rewards = torch.zeros(candidates, device=DEVICE)

            for t in range(planning_horizon):
                softmax_actions = torch.softmax(candidates_actions[:, t, :], dim=1) 
                # onehot_actions = torch.nn.functional.one_hot(softmax_actions.argmax(dim=1), num_classes=action_dim).float()
                h, s = rssmmodel.imagine_step(h, s, softmax_actions) # no need to discretize action space during imagination apparently

                r_t = rssmmodel.reward(s, h)      
                rewards += r_t

            top_values, top_idx = torch.topk(rewards, K, dim=0)
            top_actions = candidates_actions[top_idx]    

            #refit
            mean = top_actions.mean(dim=0)                
            std  = top_actions.std(dim=0) + 1e-5         

        softmax_best_action = torch.softmax(mean[0], dim=-1)       
        best_action = torch.nn.functional.one_hot(softmax_best_action.argmax(dim=-1), num_classes=action_dim).float()
        return best_action    
    
