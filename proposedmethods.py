from algobase import AlgoBase
from prediction import PredictionImpossible
import numpy as np
from six import iteritems
import heapq

class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.
    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. all the algorithms in this module.
    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

class OurMethod(SymmetricAlgo):


    def __init__(self,dataset,base_line=False, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):


        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.list_of_cats = dataset.item_weight
        self.taste_score_data = dataset.user_weight
        self.base_line = base_line
        self.user_mean = dataset.all_user_means
        



    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):
        details = {}
        details['was_impossible'] = False
        est = self.means[u]
        raw_uid = self.trainset.to_raw_uid(u)
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            details['was_impossible'] = True
            # details['default predict1'] = self.default_prediction()
            details['mean predict'] = self.user_mean[raw_uid]
            item_weight = self.list_of_cats[i]
            est = self.user_mean[raw_uid]
        else:
           item_weight = self.list_of_cats[self.trainset.to_raw_iid(i)]

        result = np.dot(item_weight, self.taste_score_data[raw_uid])

        # if (est < self.trainset.global_mean) :
        #     est = est + est * (result - 0.05)
        # else :
        est = est + est * (result + 0.05)
        
        details['total neighbor weigth_score'] = result
        #add list: error, result, user's taste ,movie's cate
        return est, details