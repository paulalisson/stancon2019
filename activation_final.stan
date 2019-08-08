// Code by Nick van het Nederend and Paula Lisson

functions {
  real race(int winner, real RT, real accum_1_mu, real accum_1_sig, real accum_2_mu, real accum_2_sig){
    
    real log_lik;
    log_lik = 0;
    
    if(winner==1){
      log_lik += lognormal_lpdf(RT| accum_1_mu, accum_1_sig);
      log_lik += lognormal_lccdf(RT|accum_2_mu, accum_2_sig); 
    }
    else {
      log_lik += lognormal_lpdf(RT| accum_2_mu, accum_2_sig);
      log_lik += lognormal_lccdf(RT|accum_1_mu, accum_1_sig); 
    }
    return(log_lik);
  }
  
  // RTs for ppc
  vector race_rng(real mu_1, real sig_1, real mu_2, real sig_2, int rctype){
    vector[2] gen;
    real accum_1_RT = lognormal_rng(mu_1, sig_1);
    real accum_2_RT = lognormal_rng(mu_2, sig_2);
    
    if(accum_1_RT < accum_2_RT){
      gen[1] = accum_1_RT;
      if(rctype == -1){
        gen[2] = 1;
      }
      else {
        gen[2] = 0;
      }
    }
    else {
      gen[1] = accum_2_RT;
      if(rctype == 1){
        gen[2] = 1;
      }
      else {
        gen[2] = 0;
      }
    }
    return(gen);
  }
}
data { 
  int<lower = 1> N_obs; 
  int<lower = 1> N_choices; 
  int<lower = 1> n_u;
  int<lower = 1> n_w;
  int<lower =-1, upper = 1> rctype[N_obs];
  int<lower =-1, upper = 1> group[N_obs];
  int<lower = 1, upper = N_choices> winner[N_obs];
  vector<lower = 0>[N_obs] RT;
  
  int<lower = 1> N_subj;
  int<lower = 1> N_item;
  int<lower=1> subj[N_obs];    
  int<lower=1> item[N_obs]; 
}
transformed data {
  real  b; //arbitrary threshold
  real min_RT;
  b = 30;
  min_RT = min(RT);
}
parameters{
  vector[8] beta;
  real alpha[N_choices]; 
  
  real<lower=fmax(fabs(beta[7]),fabs(beta[8]))> sigma_e;
  
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
  alpha ~ normal(0,10);
  beta ~ normal(0,1);
  sigma_e ~ normal(0,2);
  tau_u ~ normal(0,1);
  tau_w ~ normal(0,1);
  
  //priors
  L_u ~ lkj_corr_cholesky(2.0);
  L_w ~ lkj_corr_cholesky(2.0);
  for (s in 1:N_subj)
    z_u[s] ~ normal(0,1);
  for (i in 1:N_item)
    z_w[i] ~ normal(0,1);
  
  
  
  
  for (n in 1:N_obs) {
    real accum_1_mu = b - (alpha[1] + u[subj[n],1] + w[item[n],1] + beta[1]*group[n] + rctype[n]*(beta[3]+u[subj[n],3]) + group[n]*rctype[n]*beta[5]);
    real accum_2_mu = b - (alpha[2] + u[subj[n],2] + w[item[n],2] + beta[2]*group[n] + rctype[n]*(beta[4]+u[subj[n],4]) + group[n]*rctype[n]*beta[6]);
    
    real accum_1_sig = sigma_e + group[n]*beta[7];
    real accum_2_sig = sigma_e + group[n]*beta[8];
    
    target += race(winner[n], RT[n], accum_1_mu, accum_1_sig, accum_2_mu, accum_2_sig);
  }
}



generated quantities {
  vector[N_obs] rt_1;
	vector[N_obs] rt_2;
  vector[N_obs] gen_acc;
  vector[N_obs] gen_rctype;
  vector[N_obs] gen_RT;
  vector[N_obs] gen_group;
  
  vector[N_obs] log_lik;
  
  for (n in 1:N_obs){
    vector[2] gen;  
    real mu_1;
    real mu_2;
    real sig_1;
    real sig_2;
    
    mu_1 = b - (alpha[1] + u[subj[n],1] + w[item[n],1] + beta[1]*group[n] + rctype[n]*(beta[3]+u[subj[n],3]) + group[n]*rctype[n]*beta[5]);
    mu_2 = b - (alpha[2] + u[subj[n],2] + w[item[n],2] + beta[2]*group[n] + rctype[n]*(beta[4]+u[subj[n],4]) + group[n]*rctype[n]*beta[6]);
    
    sig_1 = sigma_e + group[n]*beta[7];
    sig_2 = sigma_e + group[n]*beta[8];
    
    gen = race_rng(mu_1, sig_1, mu_2, sig_2, rctype[n]);
    gen_RT[n] = gen[1];
    gen_acc[n] = gen[2];
    gen_rctype[n] = rctype[n];
    gen_group[n] = group[n];
    
    rt_1[n] = lognormal_rng(mu_1, sig_1);
		rt_2[n] = lognormal_rng(mu_2, sig_2);
		
    log_lik[n] = race(winner[n], RT[n], mu_1, sig_1, mu_2, sig_2);
  }
} 
