import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import als_mf as al
import sgd_mf as sg

def create_train_test(ratings):
    """
    split into training and test sets,
    remove 10 ratings from each user
    and assign them to the test set
    """
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_index = np.random.choice(
            np.flatnonzero(ratings[user]), size = 10, replace = False)

        train[user, test_index] = 0.0
        test[user, test_index] = ratings[user, test_index]
        
    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)
    return train, test
    
    
def plot_learning_curve(model):
    """visualize the training/testing loss"""
    print("Plot")
    linewidth = 3
    plt.plot(model.test_mse_record, label = 'Test', linewidth = linewidth)
    plt.plot(model.train_mse_record, label = 'Train', linewidth = linewidth)
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.legend(loc = 'best')
    plt.show()
    
    
df = pd.read_csv("data/u.data",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
print(df.shape)
df.head()

n_users = df['user_id'].unique().shape[0]
n_items = df['item_id'].unique().shape[0]
ratings = np.zeros((n_users, n_items))
for row in df.itertuples(index = False):
    ratings[row.user_id - 1, row.item_id - 1] = row.rating

matrix_size = np.prod(ratings.shape)
interaction = np.flatnonzero(ratings).shape[0]
sparsity = 100 * (interaction / matrix_size)

print('dimension: ', ratings.shape)
print('sparsity: {:.1f}%'.format(sparsity))

plt.rcParams['figure.figsize'] = 8, 6 
plt.rcParams['font.size'] = 12

plt.hist(np.sum(ratings != 0, axis = 1), histtype = 'stepfilled', bins = 30,
         alpha = 0.85, label = '# of ratings', color = '#7A68A6', normed = True)
plt.axvline(x = 10, color = 'black', linestyle = '--')
plt.legend(loc = "upper right")
#plt.show()

train, test = create_train_test(ratings)


als = al.ALS_MF(n_iters = 100, n_factors = 40, reg = 0.01)
als.fit(train, test)


sgd = sg.SGD_MF()
sgd.fit(train, test)
plot_learning_curve(als)
    

    

