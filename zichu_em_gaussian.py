#!/usr/bin/python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

#DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)
DATA_PATH = 'c:/users/LZC/desktop/ML'
def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    clusters = []
    if args.cluster_num:
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        lambdas=np.random.rand(args.cluster_num)
        mus=np.random.rand(args.cluster_num,2)
        if not args.tied:
            for z in range(args.cluster_num):
                rand_sigmas = list(np.random.rand(3))
                # print(rand_sigmas)
                for i in range(2):
                    for j in range(i, 2):
                        sigmas[z][i][j] = rand_sigmas.pop()
                        # print(sigmas[z][i][j] )
                for i in range(1, 2):
                    for j in range(i):
                        sigmas[z][i][j] = sigmas[z][j][i]
                sigmas[z]+=np.identity(2)
        else:
            rand_sigmas = list(np.random.rand(int((args.cluster_num ** 2 - args.cluster_num) / 2) + args.cluster_num))
            # print(rand_sigmas)
            for i in range(2):
                for j in range(i, 2):
                    sigmas[i][j] = rand_sigmas.pop()+1
            for i in range(1, 2):
                for j in range(i):
                    sigmas[j][i] = sigmas[i][j]
            sigmas+=np.identity(2)
        # sigmas = np.random.rand(args.cluster_num,2,2)
        # print(sigmas)
        # raise NotImplementedError #remove when random initialization is implemented
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    class model_:
        def __init__(self,lambdas,mus,sigmas):
            self.lambdas=lambdas
            self.mus=mus
            self.sigmas=sigmas
    model = model_(lambdas,mus,sigmas)
    # raise NotImplementedError #remove when model initialization is implemented
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    mu=model.mus
    sigma=model.sigmas
    lambdas=model.lambdas

    res=np.zeros((train_xs.shape[0],args.cluster_num))
    if not args.tied:
        for iter in range(args.iterations):
            for j in range(train_xs.shape[0]):
                de = 0
                for i in range(args.cluster_num):
                    # print(model.sigmas[i],i)
                    de += (model.lambdas[i] * multivariate_normal(mean=model.mus[i], cov=model.sigmas[i]). \
                           pdf(train_xs[j]))
                for z in range(args.cluster_num):
                    res[j][z] = ((model.lambdas[z] * multivariate_normal(mean=model.mus[z], cov=model.sigmas[z]).pdf(
                        train_xs[j])) / de)
            # M
            for i in range(args.cluster_num):
                NK = sum([res[x][i] for x in range(train_xs.shape[0])])
                model.mus[i] = sum([res[x][i] * train_xs[x] for x in range(train_xs.shape[0])]) / NK

                model.sigmas[i] = sum(
                    [res[x][i] / NK * (
                                np.array([train_xs[x] - model.mus[i]]) * np.array([train_xs[x] - model.mus[i]]).T)
                     for x in range(train_xs.shape[0])])
                # print( ((np.array([train_xs[2] - model.mus[i]]) ).shape))
                model.lambdas[i] = NK / train_xs.shape[0]
    else:
        for iter in range(args.iterations):
            for j in range(train_xs.shape[0]):
                de = 0
                for i in range(args.cluster_num):
                    # print(model.sigmas[i],i)
                    de += (model.lambdas[i] * multivariate_normal(mean=model.mus[i], cov=model.sigmas). \
                           pdf(train_xs[j]))
                for z in range(args.cluster_num):
                    res[j][z] = ((model.lambdas[z] * multivariate_normal(mean=model.mus[z], cov=model.sigmas).pdf(
                        train_xs[j])) / de)
            # M
            for i in range(args.cluster_num):
                NK = sum([res[x][i] for x in range(train_xs.shape[0])])
                model.mus[i] = sum([res[x][i] * train_xs[x] for x in range(train_xs.shape[0])]) / NK

                model.sigmas = sum(
                    [res[x][i] / NK * (
                                np.array([train_xs[x] - model.mus[i]]) * np.array([train_xs[x] - model.mus[i]]).T)
                     for x in range(train_xs.shape[0])])
                # print( ((np.array([train_xs[2] - model.mus[i]]) ).shape))
                model.lambdas[i] = NK / train_xs.shape[0]

        # cur_ll=average_log_likelihood(model,train_xs)

    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    #raise NotImplementedError #remove when model training is implemented

    return model

def average_log_likelihood(model, data):
    from math import log
    from scipy.stats import multivariate_normal
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    for line in data:
        count=0
        for i in range(len(model.lambdas)):
            if len(model.sigmas.shape)==3:
                count+=model.lambdas[i]*multivariate_normal(mean=model.mus[i],cov=model.sigmas[i]).pdf(line)
            else:
                count += model.lambdas[i] * multivariate_normal(mean=model.mus[i], cov=model.sigmas).pdf(line)
        ll+=log(count)
    ll=ll/data.shape[0]
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model.lambdas
    mus = model.mus
    sigmas = model.sigmas
    # raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, mus, sigmas

def main():
    import argparse
    import decimal
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()

