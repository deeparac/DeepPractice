import os
import urllib
import zipfile
import pandas as pd
import mxnet as mx

ctx = [mx.gpu(0)]

def simpleMF():
    user = mx.sym.Variable("user")
    movie = mx.sym.Variable("movie")

    y = mx.sym.Variable("softmax_label")

    # embedding
    user = mx.sym.Embedding(user, input_dim=n_users, output_dim=25)
    movie = mx.sym.Embedding(movie, input_dim=n_movies, output_dim=25)

    # network
    nn = mx.sym.sum_axis((user * movie), axis=1)
    nn = mx.sym.flatten(nn)
    yhat = mx.sym.LinearRegressionOutput(nn, y)

    model = mx.module.Module(
                symbol = yhat,
                context=ctx, 
                data_names=["user", "movie"],
                label_names=["softmax_label"]
            )

    model.fit(
        X_train,
        eval_data=X_eval,
        eval_metric='rmse',
        num_epoch=10, 
        optimizer='adam', 
        optimizer_params={'learning_rate': 1e-3},
        batch_end_callback=mx.callback.Speedometer(batch_size, 250)
    )
    
def deepMF():
    user = mx.sym.Variable("user")
    movie = mx.sym.Variable("movie")

    y = mx.sym.Variable("softmax_label")

    # embedding
    user = mx.sym.Embedding(user, input_dim=n_users, output_dim=50)
    movie = mx.sym.Embedding(movie, input_dim=n_movies, output_dim=25)
    
    nn = mx.sym.concat(user, movie)
    nn = mx.sym.flatten(nn)
    
    # network
    nn = mx.symbol.FullyConnected(data=nn, num_hidden=64)
    nn = mx.symbol.BatchNorm(nn)
    nn = mx.symbol.Activation(data=nn, act_type='relu')
    nn = mx.symbol.FullyConnected(data=nn, num_hidden=64)
    nn = mx.symbol.BatchNorm(nn)
    nn = mx.symbol.Activation(data=nn, act_type='relu')
    nn = mx.symbol.FullyConnected(data=nn, num_hidden=1)
    
    yhat = mx.sym.LinearRegressionOutput(nn, y)

    model = mx.module.Module(
                symbol = yhat,
                context=ctx, 
                data_names=["user", "movie"],
                label_names=["softmax_label"]
            )

    model.fit(
        X_train,
        eval_data=X_eval,
        eval_metric='rmse',
        num_epoch=10, 
        optimizer='adam', 
        optimizer_params={'learning_rate': 1e-3},
        batch_end_callback=mx.callback.Speedometer(batch_size, 250)
    )
    
def make_sure_data():
    if not os.path.exists('./ml-20m/'):
        url, name = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip', 'ml-20m.zip'
        if not os.path.exists(name):
            urllib.urlretrieve(url, name)
        with zipfile.ZipFile(name, 'r') as f:
            f.extractall('./')

make_sure_data()
data = pd.read_csv('./ml-20m/ratings.csv', sep=',', usecols=(0, 1, 2))

n_users, n_movies = data['userId'].max(), data['movieId'].max()
batch_size = 4096 * 8

# train test split
ntrain = data.shape[0] / 20 * 19
data = data.sample(frac=1).reset_index(drop=True)

train_users = data['userId'].values[:ntrain] - 1
train_movies = data['movieId'].values[:ntrain] - 1
train_ratings = data['rating'].values[:ntrain]

valid_users = data['userId'].values[ntrain:] - 1
valid_movies = data['movieId'].values[ntrain:] - 1
valid_ratings = data['rating'].values[ntrain:]

train_iter = {
    'user': train_users, 
    'movie': train_movies 
}

valid_iter = {
    'user': valid_users, 
    'movie': valid_movies 
}

X_train = mx.io.NDArrayIter(train_iter, 
                            label=train_ratings, 
                            batch_size=batch_size)
X_eval  = mx.io.NDArrayIter(valid_iter, 
                            label=valid_ratings, 
                            batch_size=batch_size)

deepMF()