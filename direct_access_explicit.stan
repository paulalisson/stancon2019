// Code by Nick van het Nederend and Paula Lisson

functions {
  real direct_access(int accuracy, real RT, real theta, real P_b, real mu, real delta, real sigma_e){
    
    real p_answer_correct;
    real p_answer_correct_direct_access;
    real p_answer_correct_reanalysis;
    real p_answer_incorrect;
    
    // theta * (P_b * 1-theta)
    // -- Combination of equation (20) and (21) in log
    p_answer_correct = log_sum_exp(log(theta), log(P_b) + log1m(theta));
    // theta / p_answer_correct
    // -- Proportion initial correct, equation (20) in log
    p_answer_correct_direct_access = log(theta) - p_answer_correct;
    // (P_b * 1-theta) / p_answer_correct
    // -- Proportion reanalysis, equation (21) in log
    p_answer_correct_reanalysis = log(P_b) + log1m(theta) - p_answer_correct;
    
    // (1-theta) * (1-P_b)
    // -- Equation (22) in log
    p_answer_incorrect = log1m(P_b) + log1m(theta);
    
    if(accuracy==1) {
              // Increment on likelihood if accuracy=1
      return (p_answer_correct + 
            // Increment on likelihood due to RT:
            log_sum_exp(
              p_answer_correct_direct_access + lognormal_lpdf(RT| mu, sigma_e),
              p_answer_correct_reanalysis + lognormal_lpdf(RT| mu + delta, sigma_e) ));
    } else {
      return (p_answer_incorrect + 
              lognormal_lpdf(RT| mu, sigma_e));
    }
  }

vector direct_access_rng(real theta, real P_b, real mu, real delta, real sigma_e){
    int init_acc;
    int backtrack;
    vector[2] gen;
    
    init_acc = bernoulli_rng(theta);
    backtrack = 0; 
    if (init_acc==1){
      gen[1] = lognormal_rng(mu, sigma_e);
      gen[2] = init_acc;
    } else if(init_acc!=1){
      backtrack = bernoulli_rng(P_b);
    } if(backtrack==1){
        gen[1] = lognormal_rng(mu + delta, sigma_e);
        gen[2] = 1; 
      } else {
        gen[1] = lognormal_rng(mu, sigma_e);
        gen[2] = 0;
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
  
  int<lower=1> N_subj;         
  int<lower=1> N_item; 
  int<lower=1> subj[N_obs];    
  int<lower=1> item[N_obs];    
}

parameters {
  vector[5] beta;                      //slopes per main effect
  real mu_0;                           //logmean
  real<lower=fabs(beta[3])>  delta_0;  //effect of reanalysis 
  real gamma;                          //prob of backtracking in logit space
  real alpha;                          
  real<lower=fabs(beta[5])> sigma_e_0; //logsd
  
  // betas added to constrain the model 
  real<upper=0> beta_group;  // because of the contrast coding, bc controls are -1 
  real<upper=fabs(beta_group)> beta_interaction; // smaller than beta_group

  cholesky_factor_corr[2] L_u;  
  cholesky_factor_corr[2] L_w;  
  vector<lower=0>[2] tau_u; 
  vector<lower=0>[2] tau_w; 
  vector[2] z_u[N_subj];      
  vector[2] z_w[N_item];      
}

transformed parameters {
  vector[2] u[N_subj];             
  vector[2] w[N_item];             
  {
    matrix[2,2] Sigma_u;    
    matrix[2,2] Sigma_w;
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
  beta ~ normal(0,3);
  beta_group ~ normal(0,3);
  beta_interaction ~ normal(0,3);
  delta_0 ~ normal(0,0.1);
  mu_0 ~ normal(0,10);
  sigma_e_0 ~ normal(0,1);
  tau_u ~ normal(0,1);
  tau_w ~ normal(0,1);
  gamma ~ normal(-2,1);  
  
  
  L_u ~ lkj_corr_cholesky(2.0);
  L_w ~ lkj_corr_cholesky(2.0);
  for (j in 1:N_subj)
    z_u[j] ~ normal(0,1);
  for (k in 1:N_item)
    z_w[k] ~ normal(0,1);
  
  // log likelihood
  for (i in 1:N_obs){
    //Adjust parameters with random and fixed effects, 
    real theta = inv_logit(alpha + rctype[i]*beta[1] + group[i]*beta_group + rctype[i]*group[i]*beta_interaction + u[subj[i],2] + w[item[i],2]);
    real mu = mu_0 + group[i]*beta[2] + u[subj[i],1] + w[item[i],1];
    real delta = delta_0 + group[i]*beta[3];
    real P_b = inv_logit(gamma + group[i]*beta[4]);
    real sigma_e = sigma_e_0 + group[i]*beta[5];
    
    target += direct_access(accuracy[i], RT[i], theta, P_b, mu, delta, sigma_e);
  }
}

generated quantities {
  real prob_or_i;
  real prob_or_c;
  real prob_sr_i;
  real prob_sr_c;
  
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
  
  prob_or_i = inv_logit(alpha + beta[1] + beta_group + beta_interaction);
  prob_or_c = inv_logit(alpha + beta[1] - beta_group - beta_interaction);
  prob_sr_i = inv_logit(alpha - beta[1] + beta_group - beta_interaction);
  prob_sr_c = inv_logit(alpha - beta[1] - beta_group + beta_interaction);
  
  mu_i = mu_0 + beta[2];
  mu_c = mu_0 - beta[2];
  
  delta_i_0 = delta_0 + beta[3];
  delta_c_0 = delta_0 - beta[3];
  
  delta_i = exp(mu_i+delta_i_0)-exp(mu_i);
  delta_c = exp(mu_c+delta_c_0)-exp(mu_c);
  
  P_b_i = inv_logit(gamma + beta[4]);
  P_b_c = inv_logit(gamma - beta[4]);
  
  sigma_e_i = sigma_e_0 + beta[5];
  sigma_e_c = sigma_e_0 - beta[5];
  
  
  for (i in 1:N_obs){
    // Adjust parameters with random and fixed effects, 
    real theta = inv_logit(alpha + rctype[i]*beta[1] + group[i]*beta_group + rctype[i]*group[i]*beta_interaction + u[subj[i],2] + w[item[i],2]);
    real mu = mu_0 + group[i]*beta[2] + u[subj[i],1] + w[item[i],1];
    real delta = delta_0 + group[i]*beta[3];
    real P_b = inv_logit(gamma + group[i]*beta[4]);
    real sigma_e = sigma_e_0 + group[i]*beta[5];
    // Add this for loo comparison later
    log_lik[i] = direct_access(accuracy[i], RT[i], theta, P_b, mu, delta, sigma_e);
    // Generate data from the sampled parameters
    gen = direct_access_rng(theta, P_b, mu, delta, sigma_e);
    gen_RT[i] = gen[1];
    gen_acc[i] = gen[2];
    gen_rctype[i] = rctype[i];
    gen_group[i] = group[i];
  }
}
