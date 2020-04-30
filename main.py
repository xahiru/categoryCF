import util
import knnwithmeans
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise.model_selection import KFold
import copy
import numpy as np
import pandas as pd

import random

my_seed = 100
random.seed(my_seed)
np.random.seed(my_seed)

from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy



file_path_save_data = util.file_path_save_data
datasetname = util.datasetname
data1 = util.data1
list_of_cats = util.list_of_cats


print(len(data1.user_weight))
print(len(data1.all_user_means))
print(len(data1.item_weight))

# trainset, testset = train_test_split(data1, test_size=.25)

# # # # Run 5-fold cross-validation and print results

user_based = True  # changed to False to do item-absed CF
sim_options = {'name': 'pearson', 'user_based': user_based}
algo = knnwithmeans.KNNWithMeans(data1, sim_options=sim_options)
# algo = KNNWithMeans(sim_options=sim_options)
cross_validate(algo, data1, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# # # Train the algorithm on the trainset, and predict ratings for the testset
# algo.fit(trainset)
# predictions = algo.test(testset)

# # # # Then compute RMSE
# accuracy.rmse(predictions)
# accuracy.mae(predictions)


# df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'ratings_ui', 'estimated_Ratings', 'details'])
# print(df)
# df.to_csv(file_path_save_data+'Predictions_'+datasetname+'.csv', sep='\t', encoding='utf-8')
       

