import itertools
import random
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm
import os
import userstudy_sampling.ckl as ckl

random.seed(1234)
torch.random.manual_seed(1234)
np.random.seed(1234)

torch.set_default_tensor_type(torch.DoubleTensor)

import json


def json_readfile(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def json_savefile(filename, data):
    # store the new json file
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True)


def triplet_in_matrix(triplet, matrix):
    # check that the reference has not been previously sampled
    tensor_triplet = torch.LongTensor(triplet)
    matrix = torch.LongTensor(matrix)
    br = matrix[:, 0] == tensor_triplet[0]
    ba = matrix[:, 1] == tensor_triplet[1]
    bb = matrix[:, 2] == tensor_triplet[2]
    idxs1 = ((br + ba + bb) == 3).nonzero()

    br = matrix[:, 0] == tensor_triplet[0]
    ba = matrix[:, 1] == tensor_triplet[2]
    bb = matrix[:, 2] == tensor_triplet[1]
    idxs2 = ((br + ba + bb) == 3).nonzero()

    idxs = torch.cat((idxs1, idxs2), dim=0)

    return idxs


def entropy(dist, dim=-1):
    return -(dist * dist.log()).sum(dim=dim)


def build_triplets_from_pairs(nelem, pairs, answered_triplets):
    def _chunks(l, n):
        """yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # build the triplets by selecting the reference query from the uniform distribution (as the prior)
    triplets = []
    ref_list = list(range(nelem))
    pairs_batch = _chunks(pairs, nelem)
    for j, pairs_hit in enumerate(pairs_batch):
        shuffle(ref_list)

        for ix, pair in enumerate(pairs_hit):
            ref = ref_list[ix]
            pair = pair.numpy()

            # check that if the ref is in the comparison
            ref_in_pair = ref in pair
            old_ref_in_pair = False

            # if it has been already sampled in previous batches
            ref_already_sampled1 = len(
                triplet_in_matrix([ref, *pair], answered_triplets)) > 0
            old_ref_already_sampled = False

            # if it has been already sampled in that batch (should not happen because all the
            # pairs of comparisons are different)
            already_sampled = [ref, *pair] in triplets
            old_already_sampled = False

            new_ref = ix
            while ref_in_pair or ref_already_sampled1 or already_sampled or \
                    old_ref_in_pair or old_ref_already_sampled or old_already_sampled:
                new_ref = (new_ref + 1) % nelem

                ref = ref_list[new_ref]

                ref_in_pair = ref in pair
                ref_already_sampled1 = len(
                    triplet_in_matrix([ref, *pair], answered_triplets)) > 0
                already_sampled = [ref, *pair] in triplets

                # this means we are starting to loop from the beginning of the array already
                # we will need to find a triplet already sampled where the reference we are
                # currently evaluating fits with the comparison
                if ix > new_ref and not (
                        ref_in_pair or ref_already_sampled1 or already_sampled):
                    # get the reference we started in the current iteration
                    old_ref = ref_list[ix]
                    # get the pair that has been already sampled
                    prev_pair = [triplets[j * nelem + new_ref][1],
                                 triplets[j * nelem + new_ref][2]]

                    old_ref_in_pair = old_ref in prev_pair
                    old_ref_already_sampled = len(
                        triplet_in_matrix([old_ref, *prev_pair],
                                          answered_triplets)) > 0
                    old_already_sampled = [old_ref, *prev_pair] in triplets

            if ix > new_ref:
                triplets[j * nelem + new_ref][0] = old_ref

            # if the reference was in the pair swap positions
            if new_ref != ix:
                aux = ref_list[ix]
                ref_list[ix] = ref_list[new_ref]
                ref_list[new_ref] = aux

            triplets.append([ref, *pair])

        # see how the reference sampling was
        count = torch.zeros(nelem)
        for triplet in triplets:
            count[triplet[0]] += 1
        tqdm.write('check uniform sampling - max = %d | min = %d' %
                   (count.max(), count.min()))

    triplets = torch.LongTensor(triplets)

    # see how the reference sampling was
    count = torch.zeros(nelem)
    for triplet in triplets:
        count[triplet[0]] += 1
    tqdm.write('check uniform sampling - max = %d | min = %d' %
               (count.max(), count.min()))
    return triplets


def format_triplets(nelem, nworkers, triplets):
    # get triplets with same ref
    triplet_by_ref = []
    for i in range(nelem):
        triplet_by_ref.append(triplets[triplets[:, 0] == i])

    for i, triplets in enumerate(triplet_by_ref):
        rnd_idxs = torch.randperm(len(triplets))
        triplet_by_ref[i] = triplets[rnd_idxs]

    # build questions for each worker so the reference never repeats
    hit_triplets = []
    for i in range(nworkers):
        hit_question = []
        for triplet in triplet_by_ref:
            hit_question.append(triplet[i])
        hit_triplets.append(hit_question)

    # adapt to the user-study launch script format
    hits_data = []
    for hit_triplet in hit_triplets:
        shuffle(hit_triplet)
        hit_data = {}

        for triplet in hit_triplet:
            assert triplet[0].item() != triplet[1].item() != triplet[2].item(), \
                'bad sampling, elements in triplet do not have all different class'

        hit_data['R_input_urls_idx'] = [triplet[0].item() for triplet in
                                        hit_triplet]
        hit_data['A_input_urls_idx'] = [triplet[1].item() for triplet in
                                        hit_triplet]
        hit_data['B_input_urls_idx'] = [triplet[2].item() for triplet in
                                        hit_triplet]

        hits_data.append(hit_data)

        assert sorted(hit_data['R_input_urls_idx']) == list(range(nelem)), \
            'Error, R_imgs_urls_idx does not contain a sample of each material'

    return hits_data


def get_all_pairs(number_of_workers, input_data_urls):
    # store the idxs to index the
    nelem = len(input_data_urls)
    idxs = range(nelem)

    # get all posible pairs of combinations
    pair_combinations = list(itertools.combinations(idxs, 2))
    triplets = []
    for elem in idxs:
        for pair in pair_combinations:
            if elem != pair[0] and elem != pair[1]:
                if random.uniform(0, 1) > 0.5:
                    triplets.append([elem, pair[0], pair[1]])
                else:
                    triplets.append([elem, pair[1], pair[0]])

    triplets = np.array(triplets)
    tqdm.write(len(triplets))
    triplets_by_reference_img = [list(triplets[triplets[:, 0] == idx]) for idx
                                 in range(nelem)]
    [random.shuffle(elem) for elem in triplets_by_reference_img]
    triplets_by_reference_img = [elem[:nworkers] for elem in
                                 triplets_by_reference_img]

    items_in_hit = len(triplets_by_reference_img)

    hits_data = []
    for idx_worker in range(number_of_workers):
        hit_data = {}

        triplets_user = []
        for idx_item in range(len(triplets_by_reference_img)):
            triplets_user.append(
                [triplets_by_reference_img[idx_item][idx_worker][0],
                 triplets_by_reference_img[idx_item][idx_worker][2],
                 triplets_by_reference_img[idx_item][idx_worker][1],
                 ])

        shuffle(triplets_user)
        hit_data['R_input_urls_idx'] = [int(triplet[0]) for triplet in
                                        triplets_user]
        hit_data['A_input_urls_idx'] = [int(triplet[1]) for triplet in
                                        triplets_user]
        hit_data['B_input_urls_idx'] = [int(triplet[2]) for triplet in
                                        triplets_user]

        assert sorted(hit_data['R_input_urls_idx']) == list(
            range(items_in_hit)), \
            'Error, R_imgs_urls_idx does not contain a sample of each material'

        hits_data.append(hit_data)
    return hits_data


def get_all_triplets(nelem, answered_triplets=[]):
    """

    :param input_data_urls: list with the urls of the images for the triplets
    :param answered_pairs: matrix with shape [A x 3] where A is the number of answers and the second dimension is arrenged
            such as the first element is the reference and the second element is the closer to the reference.
    :return: returns a list of 3 element lists containing all the possible triplets based on the ones previously answered
    """

    tqdm.write('generating rest of candidate triplets')

    import itertools
    # store the idxs to index the
    idxs = range(nelem)

    # get all posible pairs of combinations
    pair_combinations = list(itertools.combinations(idxs, 2))
    triplets = []
    count_double = 0
    already_sampled = 0
    sampling = torch.zeros(len(answered_triplets))
    for elem in tqdm(idxs):
        if len(answered_triplets) == 0:
            answered_pairs = []
        else:
            answered_pairs = [(triplet[1].item(), triplet[2].item()) for triplet
                              in
                              answered_triplets[
                                  np.where(answered_triplets[:, 0] == elem)]]
        for pair in pair_combinations:
            # avoid that the pair in the triplet have the same class as the reference
            if elem in pair:
                continue
            # we do not want to sample triplets that have been sampled already
            c1 = answered_pairs.count(pair)
            c2 = answered_pairs.count((pair[1], pair[0]))

            c = c1 + c2
            if c > 0:
                already_sampled += 1
                if c > 1:
                    count_double += 1

                idx_answered = triplet_in_matrix([elem, *pair],
                                                 answered_triplets)
                sampling[idx_answered] += 1
            else:
                triplets.append([elem, *pair])
    tqdm.write('count double %d ' % count_double)
    tqdm.write('constructed %d possible triplets' % len(triplets))

    # ixx = (sampling == 0).nonzero()
    # tqdm.write (answered_triplets[ixx])
    # tqdm.write(ixx)
    return torch.LongTensor(triplets)


def maximal_information_pairs(x, mu, answers, possible_triplets, npairs,
                              eps=1e-8):
    def _get_p(x, triplets, mu=1):
        sum_x = (x ** 2).sum(dim=1).expand(x.shape[0], x.shape[0])

        # get the gram matrix difference between pairs
        d_x = (-2 * (x @ x.t()) + sum_x + sum_x.t())

        # get probability
        if mu == -1:
            numer = d_x[triplets[:, 0], triplets[:, 2]]
            denom = d_x[triplets[:, 0], triplets[:, 1]] + d_x[
                triplets[:, 0], triplets[:, 2]]
        else:
            numer = mu + d_x[triplets[:, 0], triplets[:, 2]]
            denom = 2 * mu + d_x[triplets[:, 0], triplets[:, 1]] + d_x[
                triplets[:, 0], triplets[:, 2]]

        # numerical stability
        numer = torch.clamp(numer, min=eps)
        denom = torch.clamp(denom, min=eps)

        return numer / denom

    tqdm.write('getting pairs that maximize information')

    # get also the inverted possible triplets. (i.e. 'c' is more similar to the reference than 'b')
    inv_possible_triplets = torch.zeros_like(possible_triplets)
    inv_possible_triplets[:, 0] = possible_triplets[:, 0]
    inv_possible_triplets[:, 1] = possible_triplets[:, 2]
    inv_possible_triplets[:, 2] = possible_triplets[:, 1]

    # n will be the number of different images
    n = (torch.max(torch.cat((answers, possible_triplets), dim=0)) + 1).item()

    # get idx of triplets with same reference question
    tqdm.write('\t-getting idxs')
    idxs = [(answers[:, 0] == idx).nonzero() for idx in range(n)]
    # # get idx of possible triplets with different reference but same comparison
    same_ref_comp = []
    memoization = {}
    for triplet in tqdm(possible_triplets):
        key = str(triplet[1]) + '-' + str(triplet[2])
        if key in memoization:
            pass
            # same_ref_comp.append(memoization[key])
        else:
            value = ((possible_triplets[:, 1] == triplet[1]) & (
                    possible_triplets[:, 2] == triplet[2]) |
                     ((possible_triplets[:, 2] == triplet[1]) & (
                             possible_triplets[:, 1] == triplet[
                         2]))).nonzero().squeeze()
            memoization[key] = value
            same_ref_comp.append(value)

    # get the similarity matrices for the answered and to-be-answered triplets
    tqdm.write('\t-getting similarities')
    p_xbc = _get_p(x, answers, mu=mu)  # get p^x_{ab} for every answered element
    sim = _get_p(x, possible_triplets,
                 mu=-1)  # get the similarity for all candidate triplets
    inv_sim = _get_p(x, inv_possible_triplets,
                     mu=-1)  # get the similarity of inverted triplets (used later)

    # the prior is uniform between the number of samples
    prior_prob = 1 / n

    # get tau, using the already answered triplets
    tqdm.write('\t-getting tau')
    prod_p_xbc = torch.zeros(n)
    for i, x_idx in enumerate(
            idxs):  # product is computed only on the comparisons that have same reference
        prod_p_xbc[i] = p_xbc[
            x_idx].prod()  # get the product of each triplet with same reference image
    tau = prior_prob * prod_p_xbc  # multiply by the prior

    # solve the integral
    tqdm.write('\t-solving integral for p')
    p = torch.zeros(len(same_ref_comp))
    for i, idx in enumerate(
            same_ref_comp):  # solve the integral for each element 'a' and 'b' in the database
        p[i] = (tau[possible_triplets[idx][:, 0]] * sim[
            idx]).sum()  # do the integral over the reference with fixed 'a' and 'b'
    assert bool(torch.all(p < 1).item()), 'value of p bigger than 1, p: ' + str(
        p)

    # get tau_b and tau_c
    tqdm.write('\t-computing tau_b and tau_c')
    Hs_tau_b = torch.zeros(len(
        same_ref_comp))  # use python lists instead, they do not have a fixed dimension
    Hs_tau_c = torch.zeros(len(same_ref_comp))
    for i, idx in enumerate(same_ref_comp):
        Hs_tau_b[i] = entropy(tau[possible_triplets[idx][:, 0]] * sim[idx])
        Hs_tau_c[i] = entropy(tau[possible_triplets[idx][:, 0]] * inv_sim[idx])

    # calculate the information gain
    tqdm.write('\t-getting information gain')
    prev_entropy = entropy(tau)

    # sort it by value and get the elements that maximize it
    info_gain = torch.zeros(len(Hs_tau_b))
    for i, (h_tau_b, h_tau_c) in enumerate(zip(Hs_tau_b, Hs_tau_c)):
        info_gain[i] = (prev_entropy - p[i] * h_tau_b - (1 - p[i]) * h_tau_c)

    # get the new candidate pairs according to the number of pairs we want to obtain. Get the ones that gives us the bigger
    # amount of information
    pair_infogain = []
    for triplet, gain in zip(same_ref_comp, info_gain):
        pair_infogain.append((possible_triplets[triplet[0]][1:3], gain.item()))
    selected_pairs = [pair_infogain[idx][0] for idx in
                      info_gain.topk(k=npairs)[1]]
    pairs_infogain = [pair_infogain[idx][1] for idx in
                      info_gain.topk(k=npairs)[1]]
    return torch.stack(selected_pairs), torch.tensor(pairs_infogain)


if __name__ == '__main__':

    answers_path = None  # path where answers are stored
    out_path = None  # path to store the sampling ('data/sampling_iter_10.json')
    nworkers = 10  # number of people doing the user study in each iteration
    nquestions = 100  # number of triplets to be answer for each participant
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # read the input data of the test (here I used urls pointing to the images)
    """ fill in 
    """
    input_data = None  # list with each possible stimuli to sample.
    nimages = len(input_data)

    # get all the answered triplets in the user study until now
    """ fill in 
    """
    # answers is a np.array of integers with size (N, 3) containing all the
    # triplets answered until now in the user studies (N). The triplets are
    # stored as first the reference, and then the pair that the participants
    # choose from. The array stores, the class of each stimuli for each triplet.
    # Note that the class is an integer with maximum value len(input_dat)
    # agreement is a np.array of size (N, 2) that stores the number of users
    # that answered each pair. agreement[x, 0] corresponds to the number of
    # users that have answered the stimuli answers[x, 1].
    answers, agreement = None
    # here we split the answers that have equal and different agreement.
    # Equal agreement means that the same number of participants have choosen
    # both pairs. agreement[x,0] == agreement[x,1]. Both keep the same format
    # as the answers array.
    answers_equal, answers_diff = None

    # check if we have no answers and therefore is the first iteration.
    if len(answers) == 0:
        # if it is the first iteration we generate random triplets for the
        # user study
        hit_data = get_all_pairs(nworkers, input_data)
        tqdm.write(len(hit_data))
    else:
        # get all the candidate triplets excluding the answered ones
        rest_triplets = get_all_triplets(nimages, answered_triplets=answers)

        # generate the matrix of the embeddings with the current answers
        mu = 0.05
        solver = ckl.cklXSolver(mu=mu)
        X = solver.fit(answers_diff, ndims=2)

        # generate the new pairs that maximize the information gain
        max_info_pairs, pairs_infogain = maximal_information_pairs(X, mu,
                                                                   answers_diff,
                                                                   rest_triplets,
                                                                   npairs=nquestions * nworkers)
        tqdm.write('infogain max %f | min %f' %
                   (pairs_infogain[0], pairs_infogain[-1]))

        # save data in case it is needed later for their analysis
        add_data = out_dir + '/additional_data/'
        torch.save(X, add_data + 'X_matrix.pt')
        torch.save(max_info_pairs, add_data + 'max_info_pairs.pt')
        torch.save(pairs_infogain, add_data + 'pairs_infogain_matrix.pt')

        # obtain triplets using a uniform prior over the samples
        triplets = build_triplets_from_pairs(nimages, max_info_pairs, answers)

        # format the triplets to be suitable to launch the user-studies
        hit_data = format_triplets(nimages, nworkers, triplets)

    # store json object in the out path
    sampling_data = {'input_urls': input_data, 'hits_input_data': hit_data}
    json_savefile(out_path, sampling_data)

    tqdm.write('done')
