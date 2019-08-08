// Code by Nick van het Nederend and Paula Lisson

functions {
  real direct_access(int accuracy, real RT, real theta, real P_b, real theta_b, real mu, real delta, real sigma_e){
    
    real p_answer_correct;
    real p_answer_correct_direct_access;
    real p_answer_correct_reanalysis;
    real p_answer_incorrect;
    real p_answer_incorrect_direct_access;
    real p_answer_incorrect_reanalysis;
    
    // CORRECT
    // -- theta + 1-theta * P_b * theta_b   
    p_answer_correct = log_sum_exp(log(theta), log1m(theta) + log(P_b) + log(theta_b));
    // theta / p_answer_correct
    // -- Proportion initial correct
    p_answer_correct_direct_access = log(theta) - p_answer_correct;
    // (P_b * 1-theta) / p_answer_correct
    // -- Proportion reanalysis, equation (21) in log
    p_answer_correct_reanalysis = log1m(theta) + log(P_b) + log(theta_b) - p_answer_correct;

    // INCORRECT
    
    p_answer_incorrect = log_sum_exp(log1m(theta), log1m(P_b) + log1m(theta) + log(P_b) + log1m(theta_b)); 
    p_answer_incorrect_direct_access = log1m(theta) + log1m(P_b) - p_answer_incorrect; 
    p_answer_incorrect_reanalysis = log1m(theta) + log(P_b) + log1m(theta_b) - p_answer_incorrect;

    
    if(accuracy==1) {
              // Increment on likelihood if accuracy=1
      return (p_answer_correct + 
            // Increment on likelihood due to RT:
            log_sum_exp(
              p_answer_correct_direct_access + lognormal_lpdf(RT| mu, sigma_e),
              p_answer_correct_reanalysis + lognormal_lpdf(RT| mu + delta, sigma_e) ));
    } else {
      return (p_answer_incorrect + 
            // Increment on likelihood due to RT:
            log_sum_exp(
              p_answer_incorrect_direct_access + lognormal_lpdf(RT| mu, sigma_e),
              p_answer_incorrect_reanalysis + lognormal_lpdf(RT| mu + delta, sigma_e) ));
    }
  }

  vector direct_access_rng(real theta, real P_b, real theta_b, real mu, real delta, real sigma_e){
    int init_acc;
    int backtrack;
    int acc_backtracking;
    vector[2] gen;
    
    init_acc = bernoulli_rng(theta);
    backtrack = 0;
    acc_backtracking = 0;
    
    if (init_acc==1){
      gen[1] = lognormal_rng(mu, sigma_e);
      gen[2] = init_acc;
      
      } else {
      backtrack = bernoulli_rng(P_b);
      
      if (backtrack==1){
        acc_backtracking = bernoulli_rng(theta_b);
          if (acc_backtracking==1){
              gen[1] = lognormal_rng(mu + delta, sigma_e);
              gen[2] = 1;
              } else if(acc_backtracking!=1) {
                gen[1] = lognormal_rng(mu + delta, sigma_e);
                gen[2] = 0;
              }
      } else if(backtrack!=1){
        gen[1] = lognormal_rng(mu, sigma_e);
        gen[2] = init_acc;
    }
  }
    return(gen);  
}
}

data {
  int<lower=1> N_obs;                   
  real RT[N_obs];                       
  int<lower=0,upper=1> accuracy[N_obs]; 
  int<lower=-1,upper=1> rctype[N_obs];  
  int<lower=-1,upper=1> group[N_obs];   
  
  int<lower = 1> n_u;
  int<lower = 1> n_w;
  int<lower=1> N_subj;         
  int<lower=1> N_item; 
  int<lower=1> subj[N_obs];    
  int<lower=1> item[N_obs];    
}

parameters {
  vector[6] beta;                      //slopes per main effect
  real mu_0;                           //logmean
  
  real alpha;
  real alpha_b;                     
  real<lower=fabs(beta[4])> delta_0;  //effect of reanalysis 
  real<lower=fabs(beta[6])> sigma_e_0; //logsd
  real gamma;
  
  // betas added to constrain the model 
  real<upper=0> beta_group;  // because of the contrast coding, bc controls are -1 
  real<upper=fabs(beta_group)> beta_interaction; // smaller than beta_group
  real beta_group_b;  // because of the contrast coding, bc controls are -1 
  real beta_interaction_b; // smaller than beta_group

  cholesky_factor_corr[n_u] L_u;  
  cholesky_factor_corr[n_w] L_w;  
  vector<lower=0>[n_u] tau_u; 
  vector<lower=0>[n_w] tau_w; 
  vector[n_u] z_u[N_subj];      
  vector[n_w] z_w[N_item];      
}

transformed parameters {
  vector[n_u] u[N_subj];             
  vector[n_w] w[N_item];            
  {
    matrix[n_u,n_u] Sigma_u;    
    matrix[n_w,n_w] Sigma_w;
    Sigma_u = diag_pre_multiply(tau_u,L_u);
    Sigma_w = diag_pre_multiply(tau_w,L_w);
    for(j in 1:N_subj)
      u[j] = Sigma_u * z_u[j];
    for(k in 1:N_item)
      w[k] = Sigma_w * z_w[k];
  }
}

model {
  //priors
  alpha ~ normal(1,1); 
  alpha_b ~ normal(1,1); 
  beta ~ normal(0,3);
  beta_group ~ normal(0,3);
  beta_interaction ~ normal(0,3);
  delta_0 ~ normal(0,0.1);
  mu_0 ~ normal(0,10);
  sigma_e_0 ~ normal(0,1);
  tau_u ~ normal(0,1);
  tau_w ~ normal(0,1);
  gamma ~ normal(0,1);
  
  
  L_u ~ lkj_corr_cholesky(2.0);
  L_w ~ lkj_corr_cholesky(2.0);
  for (s in 1:N_subj)
    z_u[s] ~ normal(0,1);
  for (i in 1:N_item)
    z_w[i] ~ normal(0,1);
    
  // log likelihood
  for (i in 1:N_obs){
    // Adjust parameters with random and fixed effects, 
    real theta = inv_logit(alpha + rctype[i]*beta[1] + group[i]*beta_group + rctype[i]*group[i]*beta_interaction + u[subj[i],2] + w[item[i],2]);
    real theta_b = inv_logit(alpha_b + rctype[i]*beta[2] + group[i]*beta_group_b + rctype[i]*group[i]*beta_interaction_b);
    real mu = mu_0 + group[i]*beta[3] + u[subj[i],1] + w[item[i],1];
    real delta = delta_0 + group[i]*beta[4];
    real P_b = inv_logit(gamma + group[i]*beta[5]);
    real sigma_e = sigma_e_0 + group[i]*beta[6];
    
    target += direct_access(accuracy[i], RT[i], theta, theta_b, P_b, mu, delta, sigma_e);
  }
}

generated quantities {
  // probabilities of iniital retrieval
  real prob_or_i;
  real prob_or_c;
  real prob_sr_i;
  real prob_sr_c;
  
  // probabilities of second retrieval (after backtracking)
  real prob_or_i_b;
  real prob_or_c_b;
  real prob_sr_i_b;
  real prob_sr_c_b;
  
  real mu_i;
  real mu_c;
  
  real delta_i_0;
  real delta_c_0;
  
  real delta_i;
  real delta_c;
  
  real P_b_i;
  real P_b_c;
  
  real sigma_e_i;
  real sigma_e_c;
  
  vector[N_obs] log_lik;
  
  vector[2] gen;
  vector[N_obs] gen_acc;
  vector[N_obs] gen_rctype;
  vector[N_obs] gen_RT;
  vector[N_obs] gen_group;
  
  // for first retrieval (theta):
  prob_or_i = inv_logit(alpha + beta[1] + beta_group + beta_interaction);
  prob_or_c = inv_logit(alpha + beta[1] - beta_group - beta_interaction);
  prob_sr_i = inv_logit(alpha - beta[1] + beta_group - beta_interaction);
  prob_sr_c = inv_logit(alpha - beta[1] - beta_group + beta_interaction);
  
  prob_or_i_b = inv_logit(alpha_b + beta[2] + beta_group_b + beta_interaction_b);
  prob_or_c_b = inv_logit(alpha_b + beta[2] - beta_group_b - beta_interaction_b);
  prob_sr_i_b = inv_logit(alpha_b - beta[2] + beta_group_b - beta_interaction_b);
  prob_sr_c_b = inv_logit(alpha_b - beta[2] - beta_group_b + beta_interaction_b);
  
  mu_i = mu_0 + beta[3];
  mu_c = mu_0 - beta[3];
  
  delta_i_0 = delta_0 + beta[4];
  delta_c_0 = delta_0 - beta[4];
  
  delta_i = exp(mu_i+delta_i_0)-exp(mu_i);
  delta_c = exp(mu_c+delta_c_0)-exp(mu_c);
  
  P_b_i = inv_logit(gamma + beta[5]);
  P_b_c = inv_logit(gamma - beta[5]);
  
  sigma_e_i = sigma_e_0 + beta[6];
  sigma_e_c = sigma_e_0 - beta[6];
  
  
  for (i in 1:N_obs){
    // Adjust parameters with random and fixed effects, 
    real theta = inv_logit(alpha + rctype[i]*beta[1] + group[i]*beta_group + rctype[i]*group[i]*beta_interaction + u[subj[i],2] + w[item[i],2]);
    real theta_b = inv_logit(alpha_b + rctype[i]*beta[2] + group[i]*beta_group_b + rctype[i]*group[i]*beta_interaction_b);
    real mu = mu_0 + group[i]*beta[3] + u[subj[i],1] + w[item[i],1];
    real delta = delta_0 + group[i]*beta[4];
    real P_b = inv_logit(gamma + group[i]*beta[5]);
    real sigma_e = sigma_e_0 + group[i]*beta[6];
    // Add this for loo comparison later
    log_lik[i] = direct_access(accuracy[i], RT[i], theta, theta_b, P_b, mu, delta, sigma_e);
    // Generate data from the sampled parameters
    gen = direct_access_rng(theta, P_b, theta_b, mu, delta, sigma_e);
    gen_RT[i] = gen[1];
    gen_acc[i] = gen[2];
    gen_rctype[i] = rctype[i];
    gen_group[i] = group[i];

  }
} 