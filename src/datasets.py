import os

from ast import literal_eval
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MultiLabelBinarizer


class Dataset(object):
    def __init__(self, data, skip_testdataset_bc=False, **side_info):
        data = pd.DataFrame(data, copy=True)
        if 'is_test' not in data.columns:
            data['is_test'] = False
        if 'is_tune' not in data.columns:
            data['is_tune'] = False
        data = data[['user_id', 'item_id', 'rating', 'is_test', 'is_tune']].astype({
            'user_id': int, 'item_id': int, 'rating': float, 'is_test': bool, 'is_tune': bool,
        })
        data['is_train'] = (data.is_test == False) & (data.is_tune == False)
        
        self.train = data[data.is_train == True].sample(frac=1)
        self.tune = data[data.is_tune == True].sample(frac=1)
        self.test = data[data.is_test == True].sample(frac=1)    
        self.index = {'train': 0, 'tune': 0, 'test': 0}
        
        self.n_data = len(data)
        self.min = data['rating'].min()
        self.max = data['rating'].max()
        self.range = self.max - self.min
        self.data = data
        
        self.name = side_info['name'] = side_info.get('name', 'dataset')
        self.n_user = side_info['n_user'] = side_info.get('n_user', self.data['user_id'].max() + 1)
        self.n_item = side_info['n_item'] = side_info.get('n_item', self.data['item_id'].max() + 1)
        # check
        self.diff_tag_n_user = self.diff_tag_n_item = 0
        if side_info['n_user'] != len(set(self.data['user_id'])):
            self.diff_tag_n_user = 1
            print("\tWarning!!! diff_tag_n_user, meta info: {}, dataset: {}".format(\
                                          side_info['n_user'], len(set(self.data['user_id']))))
        if side_info['n_item'] != len(set(self.data['item_id'])):
            self.diff_tag_n_item = 1
            print("\tWarning!!! diff_tag_n_item, meta info: {}, dataset: {}".format(\
                                          side_info['n_item'], len(set(self.data['item_id']))))
        self.shape = (self.n_user, self.n_item)
        self.side_info = side_info
        try:
            print("user feature shape: {}; item feature shape: {}".format(
                    side_info.get('user_features', None).shape, side_info.get('item_features', None).shape))
        except:
            print("no side feature~")
        
        # For each user in the test set, we take all the items that the user has not interacted with as negative samples. 
        # Each method outputs the preference score of users over all the items in the graph, except the positive samples in the training and validation set. 
        if not skip_testdataset_bc:
            if not os.path.exists(f"../data/datasets/{self.name}/test_dataset_bc.pkl"):
                print(f"build: ../data/datasets/{self.name}/test_dataset_bc.pkl")
                # here we need to predict the score of one user corr to all items
                # therefore we need to rebuild the test dataset        
                self.test['is_positive'] = 1  # indicate the positive data
                all_items_list = list(set(self.data['item_id']))
                print(f"all_items_list num : {len(all_items_list)}")
                test_df = self.test.drop_duplicates(['user_id'])[['user_id']].reset_index(drop=True)  # get all test user
                def f(group):
                    row = group['user_id'].iloc[0]
                    return pd.DataFrame({'user_id': [row] * len(all_items_list), 'item_id':all_items_list})
                test_df_bc = test_df.groupby('user_id', group_keys=False).apply(f)
                self.test = test_df_bc.merge(self.test, on=['user_id', 'item_id'], how='left')
                self.test['is_test'] = True
                print(f"\ttest dataframe info: userNum: {len(test_df)}; itemNum: {len(all_items_list)}")
                print(f"\ttest dataframe size before filter: {len(self.test)}")
                # remove data in train and tune
                self.test = self.test.append(self.train)
                self.test = self.test.append(self.tune)
                self.test = self.test.drop_duplicates(subset=['user_id', 'item_id'], keep=False)
                self.test = self.test.query('is_test==True').reset_index(drop=True)
                print(f"\ttest dataframe size after train&valid filter: {len(self.test)}")
                self.test['is_positive'] = self.test['is_positive'].fillna(0)
                self.test.to_pickle(f"../data/datasets/{self.name}/test_dataset_bc.pkl")
                print(f"succ save to ../data/datasets/{self.name}/test_dataset_bc.pkl")
            else:
                self.test = pd.read_pickle(f"../data/datasets/{self.name}/test_dataset_bc.pkl")
                print(f"fast load ../data/datasets/{self.name}/test_dataset_bc.csv")
        else:
            print("build_dataset_stage, skip test dataset bc")
        
        print(f"DATASET INFO: \n\tname: {self.name}\n\tnumber of user: {self.n_user}; number of item: {self.n_item}\n\tRATEING: min: {self.min}; max: {self.max}")

    def get_batch(self, mode='train', size=None):
        dataset = getattr(self, mode)
        if size is None:
            return dataset[:]
        i = self.index[mode]
        self.index[mode] = i + size if i + size < len(dataset) else 0
        return dataset[i:(i + size)]
    
    def save(self, path):
        folder = '{}/{}'.format(path, self.name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.data.to_csv(folder + '/data.csv', index=False)
        
        metadata = pd.DataFrame({'name': [self.name], 'n_user': [self.n_user], 'n_item': [self.n_item]})
        metadata.to_csv(folder + '/metadata.csv', index=False)
        
        for k, v in self.side_info.items():
            if sp.issparse(v):
                sp.save_npz('{}/{}.npz'.format(folder, k), v)
            elif isinstance(v, np.ndarray):
                np.save('{}/{}.npy'.format(folder, k), v)
            elif isinstance(v, pd.DataFrame):
                v.to_csv('{}/{}.csv'.format(folder, k), index=False)

    @staticmethod
    def load(path, skip_testdataset_bc=False, **kwargs):
        print("dataset load")
        kwargs['name'] = os.path.basename(path)
        for f in os.listdir(path):
            if f.startswith('.'):
                continue
            f_path = '{}/{}'.format(path, f)
            print("process file: {}".format(f))
            if f.endswith('metadata.csv'):
                s = pd.read_csv(f_path).iloc[0]
                kwargs.update(s.to_dict())
            elif f.endswith('.npz'):
                kwargs[f[:-4]] = sp.load_npz(f_path)
            elif f.endswith('.npy'):
                kwargs[f[:-4]] = np.load(f_path)
            elif f.endswith('.csv'):
                kwargs[f[:-4]] = pd.read_csv(f_path)
            else:
                print('File not loaded: ' + f)
        return Dataset(skip_testdataset_bc = skip_testdataset_bc, **kwargs)


class Monti(Dataset):
    def __init__(self, path, **kwargs):
        matrix = sp.coo_matrix(self._load_matlab_file(path, 'M'))
        data = pd.DataFrame({'user_id': matrix.row, 'item_id': matrix.col, 'rating': matrix.data})
        mask_test = sp.dok_matrix(self._load_matlab_file(path, 'Otest'))
        data['is_test'] = [mask_test[i, j] for i, j in zip(matrix.row, matrix.col)]
        
        kwargs['data'] = data
        kwargs['name'] = kwargs.get('name', 'monti')
        kwargs['n_user'] = kwargs.get('n_user', 3000)
        kwargs['n_item'] = kwargs.get('n_item', 3000)
        super(Monti, self).__init__(**kwargs)

    @classmethod
    def _load_matlab_file(cls, path_file, name_field):
        # https://github.com/fmonti/mgcnn
        """
        load '.mat' files
        inputs:
            path_file, string containing the file path
            name_field, string containig the field name (default='shape')
        warning:
            '.mat' files should be saved in the '-v7.3' format
        """
        db = h5py.File(path_file, 'r')
        ds = db[name_field]
        try:
            if 'ir' in ds.keys():
                data = np.asarray(ds['data'])
                ir   = np.asarray(ds['ir'])
                jc   = np.asarray(ds['jc'])
                out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        except AttributeError:
            # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
            out = np.asarray(ds).astype(np.float32).T
        db.close()
        return out


class Flixster(Monti):
    def __init__(self, path='data/raw_data/mgcnn/flixster/training_test_dataset_10_NNs.mat', **kwargs):
        kwargs['name'] = kwargs.get('name', 'Flixster')
        kwargs['user_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_users'))
        kwargs['item_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_movies'))
        super(Flixster, self).__init__(path, **kwargs)


class Movielens(Monti):
    def __init__(self, path='data/raw_data/mgcnn/movielens/split_1.mat', **kwargs):
        kwargs['name'] = kwargs.get('name', 'Movielens')
        kwargs['user_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_users'))
        kwargs['item_graph'] = sp.coo_matrix(self._load_matlab_file(path, 'W_movies'))
        kwargs['n_user'] = kwargs['user_graph'].shape[0]
        kwargs['n_item'] = kwargs['item_graph'].shape[0]
        super(Movielens, self).__init__(path, **kwargs)

class MovieLens100K(Movielens):
    def __init__(self, path='data/raw_data/ml-100k', **kwargs):        
        user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        user = pd.read_csv(path + '/u.user', engine='python', encoding='ISO-8859-1', sep='|', names=user_columns)
        user.age = user.age.astype(float)/user.age.max()
        user.gender = user.gender.astype('category').cat.codes
        user = pd.get_dummies(user, columns=['occupation'])
        user.drop(columns=['user_id', 'zip_code'], inplace=True)
        kwargs['user_features'] = user.astype(float)

        item_columns = [
            'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
            'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western',
        ]
        item = pd.read_csv(path + '/u.item', engine='python', encoding='ISO-8859-1', sep='|', names=item_columns)
        item.drop(columns=item_columns[:6], inplace=True)
        kwargs['item_features'] = item.astype(float)

        kwargs['name'] = 'MovieLens100K'
        super(MovieLens100K, self).__init__(**kwargs)
        
        
class MovieLens1M(Dataset):
    """
    https://grouplens.org/datasets/movielens/1m/
    """
    def __init__(self, path='data/raw_data/ml-1m/', **kwargs): 
        kwargs['name'] = 'MovieLens1M'
        
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
        user = pd.read_table(path+'users.dat', sep='::', header=None, names=unames, engine='python')
        # uid2id
        uid2id_map = {uid:idx for idx, uid in enumerate(user['user_id'])}
        user.age = user.age.astype(float)/user.age.max()
        user.gender = user.gender.astype('category').cat.codes
        user = pd.get_dummies(user, columns=['occupation'])
        user.drop(columns=['user_id', 'zip_code'], inplace=True)
        kwargs['user_features'] = user.astype(float)
        kwargs['n_user'] = len(user)
        
        mnames = ['movie_id', 'title', 'genres']
        item = pd.read_table(path+'movies.dat', sep='::', header=None, names=mnames, engine='python')
        # item = pd.concat([item, item['genres'].str.get_dummies(sep='|')], axis=1)
        kwargs['item_features'] = item['genres'].str.get_dummies(sep='|').astype(float)
        # itemid2id
        iid2id_map = {uid:idx for idx, uid in enumerate(item['movie_id'])}
        kwargs['n_item'] = len(item)
        
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        data = pd.read_table(path+'ratings.dat', sep='::', header=None, names=rnames, engine='python')
        # 这里因为uid和itemid不是连续的，所以会有潜在问题，因此将要remap
        data = pd.DataFrame(
            {'user_id': data.user_id.map(uid2id_map), 'item_id': data.movie_id.map(iid2id_map), 'rating': data.rating}
        )
        # random mask 20% dataset
        data['is_test']=False
        data.loc[data.sample(frac=0.2).index, 'is_test']=True
        kwargs['data'] = data
        
        kwargs['user_graph'] = kneighbors_graph(kwargs['user_features'].values, 10)
        kwargs['item_graph'] = kneighbors_graph(kwargs['item_features'].values, 10)
        
        super(MovieLens1M, self).__init__(skip_testdataset_bc=True, **kwargs)

        
        
class KuaiRandPure(Dataset):
    def __init__(self, path='data/raw_data/KuaiRand-Pure/', **kwargs): 
        kwargs['name'] = 'KuaiRandPure'

        path='data/raw_data/KuaiRand-Pure/'
        log1 = pd.read_csv(path + "./data/log_random_4_22_to_5_08_pure.csv")
        log2 = pd.read_csv(path + "./data/log_standard_4_08_to_4_21_pure.csv")
        log3 = pd.read_csv(path + "./data/log_standard_4_22_to_5_08_pure.csv")
        data = pd.concat([log1, log2, log3])[["user_id", "video_id", "is_click", "time_ms"]]
        data = data.drop_duplicates().sort_values(by='time_ms').reset_index(drop=True)
        data.columns = ['user_id', 'item_id', 'rating', 'time_ms']
        # data = data.query("user_id<6000 & item_id<6000")  

        train = data.query("time_ms<1651593600000")  # 20220504
        test = data.query("time_ms>1651593600000")
        print(f"user num: {len(set(data['user_id']))} item num: {len(set(data['item_id']))}")
        print( train.shape, test.shape, "user num of test:", len(set(test['user_id'])) )
        
        
        train["rating"] = train["rating"] * 5
        test["rating"] = test["rating"] * 5

        uid2id_map = {uid:idx for idx, uid in enumerate(set(train['user_id']).union( set(test['user_id']) ))}
        iid2id_map = {uid:idx for idx, uid in enumerate(set(train['item_id']).union( set(test['item_id']) ))}

        user_feature = pd.read_csv(path + "./data/user_features_pure.csv")
        user_feature = user_feature[user_feature['user_id'].isin(uid2id_map.keys())]
        user_feature['user_id'] = user_feature['user_id'].map(uid2id_map)
        for cc in [k for k in user_feature.columns if user_feature[k].dtype=='object']:
            user_feature[cc] = user_feature[cc].astype('category').cat.codes
        user_feature = user_feature.drop("user_id", axis=1)

        item_feature1 = pd.read_csv(path + "./data/video_features_basic_pure.csv")
        item_feature2 = pd.read_csv(path + "./data/video_features_statistic_pure.csv")
        item_feature = item_feature1.merge(item_feature2, on='video_id')
        
        item_feature = item_feature[item_feature['video_id'].isin(iid2id_map.keys())]
        item_feature = item_feature[[x for x in item_feature.columns if x not in ['upload_dt', 'visible_status']]]
        item_feature['video_id'] = item_feature['video_id'].map(iid2id_map)
        item_feature = item_feature.drop("video_id", axis=1)
        item_feature
        for cc in item_feature[[k for k in item_feature.columns if item_feature[k].dtype=='object']]:
            item_feature[cc] = item_feature[cc].astype('category').cat.codes

        from sklearn.preprocessing import MinMaxScaler
        kwargs['user_features'] = user_feature.fillna(0.0).astype(float)
        kwargs['user_features'] = pd.DataFrame( MinMaxScaler().fit_transform(kwargs['user_features']) )
        kwargs['n_user'] = len(user_feature)

        kwargs['item_features'] = item_feature.fillna(0.0).astype(float)
        kwargs['item_features'] = pd.DataFrame( MinMaxScaler().fit_transform(kwargs['item_features']) )
        kwargs['n_item'] = len(item_feature)

        
        train['is_test']=False
        test['is_test']=True
        # random mask 20% dataset
        data = pd.concat([train, test]).reset_index(drop=True)
        data['user_id'] = data['user_id'].map(uid2id_map)
        data['item_id'] = data['item_id'].map(iid2id_map)
        kwargs['data'] = data
        print("check:==> data.shape", data.shape, "--", len(set(data['user_id'])), "--", kwargs['n_user'])
        
        kwargs['user_graph'] = kneighbors_graph(kwargs['user_features'].values, 10)
        kwargs['item_graph'] = kneighbors_graph(kwargs['item_features'].values, 10)
        
        super(KuaiRandPure, self).__init__(skip_testdataset_bc=True, **kwargs)


# if __name__ == '__main__':
#     dataset = MovieLens100K()
#     dataset.save('data/datasets')
    # print("=="*20)
#     dataset = Dataset.load('data/datasets/MovieLens100K')
#     print(dataset.name)
#     print(dataset.n_user)
#     print(dataset.n_item)
#     print(dataset.n_data)
#     print(dataset.min)
#     print(dataset.max)
#     print(dataset.n_data/float(dataset.n_user*dataset.n_item))
#     print(len(dataset.train))
#     print(len(dataset.tune))
#     print(len(dataset.test))
#     print(dataset.side_info)

# if __name__ == '__main__':
#     dataset = MovieLens1M()
#     dataset.save('data/datasets')
#     print("=="*20)
#     dataset = Dataset.load('data/datasets/MovieLens1M')
#     print(dataset.name)
#     print(dataset.n_user)
#     print(dataset.n_item)
#     print(dataset.n_data)
#     print(dataset.min)
#     print(dataset.max)
#     print(dataset.n_data/float(dataset.n_user*dataset.n_item))
#     print(len(dataset.train))
#     print(len(dataset.tune))
#     print(len(dataset.test))
#     print(dataset.side_info)

    
    
# if __name__ == '__main__':
#     dataset = KuaiRandPure()
#     dataset.save('data/datasets')
#     print("=="*20)
#     dataset = Dataset.load('data/datasets/KuaiRandPure')
#     print(dataset.name)
#     print(dataset.n_user)
#     print(dataset.n_item)
#     print(dataset.n_data)
#     print(dataset.min)
#     print(dataset.max)
#     print(dataset.n_data/float(dataset.n_user*dataset.n_item))
#     print(len(dataset.train))
#     print(len(dataset.tune))
#     print(len(dataset.test))
#     print(dataset.side_info)
