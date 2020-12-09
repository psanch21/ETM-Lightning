
'''
*    Title: Topic Modeling in Embedding Spaces
*    Author: Dieng, Adji B
*    Date: 2019
*    Availability: https://github.com/adjidieng/ETM
'''
import torch
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
import os

from torch import nn
from utils.activations import get_activation
from utils.optimizers import  get_scheduler, get_optimizer
from utils.metrics.time import Timer


class ETM(pl.LightningModule):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size,
                 theta_act, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.theta_act_name = theta_act
        self.train_emb = train_embeddings


        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = get_activation(theta_act)

        self.rho = nn.Linear(rho_size, vocab_size, bias=False)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)  # nn.Parameter(torch.randn(rho_size, num_topics))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size),
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)


        self.norm_bow = True
        self.timer = Timer()
        self.save_hyperparameters()



    def get_arguments(self):
        return {'num_topics': self.num_topics,
                'vocab_size': self.vocab_size,
                't_hidden_size': self.t_hidden_size,
                'rho_size': self.rho_size,
                'enc_drop': self.enc_drop,
                'theta_act_name': self.theta_act_name,
                'train_emb': self.train_emb}



    def get_model_folder(self):
        model_params = '_'.join([str(i) for i in list(self.get_arguments().values())])
        model_params = f'{ETM.__name__}_{model_params}'
        optim_params = '_'.join([f'{k}_{v}' for k,v in self.optim_params['params'].items()])
        optim_params = f"{self.optim_params['name']}_{optim_params}"
        if self.sched_params is not None:
            sched_params = '_'.join([f'{k}_{v}' for k,v in self.sched_params['params'].items()])
            sched_params = f"{self.sched_params['name']}_{sched_params}"
            optim_params = f"{optim_params}_{sched_params}"

        return os.path.join(model_params, optim_params)



    def set_norm_bow(self, value):
        self.norm_bow = value


    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.
        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        return mu_theta, logsigma_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight)  # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0)  ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, mu_theta, logsigma_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds


    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu



    def set_optim_params(self, optim_params, sched_params):
        self.optim_params = optim_params
        self.sched_params = sched_params


    def configure_optimizers(self):
        optim = get_optimizer(self.optim_params['name'])(self.parameters(), **self.optim_params['params'])
        if  isinstance(self.sched_params, dict):
            sched = get_scheduler(self.sched_params['name'])(optim, **self.sched_params['params'])
        else:
            sched = []
        return [optim], sched



    def forward(self, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        assert theta is None
        theta, mu_theta, logsigma_theta = self.get_theta(normalized_bows)

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        other = {'mu_theta': mu_theta, 'logsigma_theta': logsigma_theta}
        return preds, theta, other



    def loss(self, preds, bows, logsigma_theta, mu_theta, aggregate=True):
        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss
        kld_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1)

        if aggregate:
            recon_loss = recon_loss.mean()
            kld_theta = kld_theta.mean()

        total_loss = recon_loss + kld_theta
        return total_loss, recon_loss, kld_theta



    def training_step(self, batch, batch_idx):
        bows = batch
        if self.norm_bow:
            sums = bows.sum(1).unsqueeze(1)
            normalized_bows = bows / sums
        else:
            normalized_bows = bows
        preds, theta, other = self(normalized_bows)

        total_loss, recons_loss, kld_theta = self.loss(preds=preds,
                                           bows=bows,
                                           logsigma_theta=other['logsigma_theta'],
                                           mu_theta=other['mu_theta'])

        elbo = -total_loss
        self.log('train_ELBO', elbo, on_step=True, on_epoch=True)

        return {'loss': total_loss}


    def on_train_epoch_start(self) -> None:
        self.timer.tic('train')
    def on_train_epoch_end(self, outputs) -> None:
        time = self.timer.toc('train')
        self.logger.experiment.add_scalar('train_time', time, self.current_epoch)



    def validation_step(self,batch, batch_idx):
        bows = batch
        if self.norm_bow:
            sums = bows.sum(1).unsqueeze(1)
            normalized_bows = bows / sums
        else:
            normalized_bows = bows
        preds, theta, other = self(normalized_bows)

        total_loss, recons_loss, kld_theta = self.loss(preds=preds,
                                           bows=bows,
                                           logsigma_theta=other['logsigma_theta'],
                                           mu_theta=other['mu_theta'])

        elbo = -total_loss
        self.log('val_ELBO', elbo, on_step=False, on_epoch=True)
        return {'loss': total_loss}


    def test_step(self, batch, batch_idx):
        bows = batch
        if self.norm_bow:
            sums = bows.sum(1).unsqueeze(1)
            normalized_bows = bows / sums
        else:
            normalized_bows = bows
        preds, theta, other = self(normalized_bows)

        total_loss, recons_loss, kld_theta = self.loss(preds=preds,
                                                       bows=bows,
                                                       logsigma_theta=other['logsigma_theta'],
                                                       mu_theta=other['mu_theta'])

        elbo = -total_loss
        self.log('test_ELBO', elbo, on_step=True, on_epoch=True)
        return {'loss': total_loss}




    # %% Visualization

    def show_topics(self, vocab, num_words=10):
        with torch.no_grad():
            beta = self.get_beta()
            print('\n')
            for k in range(self.num_topics):  # topic_indices:
                gamma = beta[k]
                top_words = list(gamma.cpu().numpy().argsort()[-num_words + 1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                print('Topic {}: {}'.format(k, topic_words))

    def nearest_neighbors(self, word, vocab):
        with torch.no_grad():
            vectors = self.rho.weight.cpu().numpy()
            index = vocab.index(word)
            print('vectors: ', vectors.shape)
            query = vectors[index]
            print('query: ', query.shape)
            ranks = vectors.dot(query).squeeze()
            denom = query.T.dot(query).squeeze()
            denom = denom * np.sum(vectors ** 2, 1)
            denom = np.sqrt(denom)
            ranks = ranks / denom
            mostSimilar = []
            [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
            nearest_neighbors = mostSimilar[:20]
            nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
            return nearest_neighbors

    @torch.no_grad()
    def get_topic_diversity(self,topk):
        beta = self.get_beta().numpy()
        num_topics = beta.shape[0]
        list_w = np.zeros((num_topics, topk))
        for k in range(num_topics):
            idx = beta[k, :].argsort()[-topk:][::-1]
            list_w[k, :] = idx
        n_unique = len(np.unique(list_w))
        TD = n_unique / (topk * num_topics)
        print('Topic diveristy is: {}'.format(TD))
        return TD

    @torch.no_grad()
    def get_document_frequency(self, corpus, wi, wj=None):
        if wj is None:
            wi_docs = corpus[:, wi]
            D_wi = np.sum(wi_docs>0)
            return  D_wi
        D_wj = 0
        D_wi_wj = 0
        wi_docs = corpus[:, wi]
        wj_docs = corpus[:, wj]
        wij_docs = (wi_docs>0) * (wj_docs>0)

        D_wj =  np.sum(wi_docs>0)
        D_wi_wj =  np.sum(wij_docs)
        return D_wj, D_wi_wj

    @torch.no_grad()
    def get_topic_coherence(self, corpus):
        beta = self.get_beta().numpy()
        D = corpus.shape[0]  ## number of docs...data is list of documents
        print('D: ', D)
        TC = []
        num_topics = len(beta)
        for k in range(num_topics):
            print('k: {}/{}'.format(k, num_topics))
            top_10 = list(beta[k].argsort()[-11:][::-1])
            TC_k = 0
            counter = 0
            for i, word in enumerate(top_10):

                D_wi = self.get_document_frequency(corpus, word)
                j = i + 1
                tmp = 0
                while j < len(top_10) and j > i:
                    # get D(w_j) and D(w_i, w_j)
                    D_wj, D_wi_wj = self.get_document_frequency(corpus, word, top_10[j])
                    # get f(w_i, w_j)
                    if D_wi_wj == 0:
                        f_wi_wj = -1
                    else:
                        f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                    # update tmp:
                    tmp += f_wi_wj
                    j += 1
                    counter += 1
                # update TC_k
                TC_k += tmp
            TC.append(TC_k)
        print('counter: ', counter)
        print('num topics: ', len(TC))
        TC = np.mean(TC) / counter
        print('Topic coherence is: {}'.format(TC))
        return TC




