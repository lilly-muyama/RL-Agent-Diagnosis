from stable_baselines3 import DQN
import torch as th
import numpy as np
import random
import os
from modules.constants import constants
from torch.nn import functional as F



SEED = constants.SEED
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
th.manual_seed(SEED)
th.use_deterministic_algorithms(True)

class RobustDQN(DQN):
    def __init__(self, *args, beta, al_r, al_p, p_proxy, **kwargs):
        self.beta = beta #(from fig 2)
        self.al_r = al_r
        self.al_p = al_p
        self.p_proxy = p_proxy
        self.norm_estimate = 0
        super(RobustDQN, self).__init__(*args, **kwargs)
        
    
    def estimate_norm_r2(self, four_tuple_batch, q_net, q_net_target):
        state = four_tuple_batch[0]  # (batch, state)
        q_net_val = q_net(state)  # (batch, actions)
        q_net_argmax = q_net_val.argmax(dim=1)  # (batch, )
        q_target_val = q_net_target(state)  # (batch, actions)
        qmax = q_target_val.gather(dim=1, index=q_net_argmax.unsqueeze(-1))  # (batch, )

        # calculate norm p for qmax
        if self.p_proxy == 'l2-norm':
            norm_estimate = (sum(qmax ** 2)) ** (1/2)  # dual norm (itself)
        elif self.p_proxy == 'l1-norm':
            norm_estimate = max(abs(qmax))  # dual norm (l_infinity)
        elif self.p_proxy == 'var-norm': #change this 
            norm_estimate = np.var(qmax.numpy())**(1/2) #XOR
        return norm_estimate, q_net_argmax

        
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            s = replay_data.observations
            a = replay_data.actions
            r = replay_data.rewards
            s_prime = replay_data.next_observations
            d = replay_data.dones
            
            with th.no_grad():
                next_q_values = self.q_net_target(s_prime)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1) #max_q_prime
                
                #added
                current_norm_estimate, _ = self.estimate_norm_r2((s, a, r, s_prime), self.q_net, self.q_net_target)
                norm_estimate = self.beta * self.norm_estimate + (1 - self.beta) * current_norm_estimate  # moving avg
                self.norm_estimate = norm_estimate  # update last norm
                
                #changed this
                # try also without the 1-d. check github for reference
                target_q_values = r - self.al_r + (1 - d) * self.gamma * (next_q_values - self.al_p * norm_estimate)
                

            # Get current Q-values estimates
            current_q_values = self.q_net(s)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=a.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))