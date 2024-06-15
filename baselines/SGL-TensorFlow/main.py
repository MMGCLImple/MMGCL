import os
import sys
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool
import ipdb

# np.random.seed(2018)
# random.seed(2018)
# tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    root_folder ='./'
    conf = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")
    seed = conf["seed"]
    print('seed=', seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    print(f"recommender: {recommender}")

    dataset = Dataset(conf)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    with tf.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        r_pred = []
        from tqdm import tqdm

        r_pred = model.predict(dataset.test_dataset_bc["user_id"], dataset.test_dataset_bc["item_id"])
        r_pred = np.array(r_pred)
            
        import sys
        sys.path.append("../MMGCL/")
        import metrics
        import time
        import pandas as pd
        print("\n\nEVALUATING...")
        time_start=time.time()
        r_true, user_ids, item_ids = dataset.test_dataset_bc[['rating', 'user_id', 'item_id']].values.T

        ddd1 = metrics.metric_at_once(r_pred, r_true, user_ids, item_ids, k=10)
        ddd2 = metrics.metric_at_once(r_pred, r_true, user_ids, item_ids, k=20)
        dddall = {**ddd1, **ddd2}

        
        return_res = {
            "P@10":dddall['Precision@10'],"P@20":dddall['Precision@20'],
            "R@10":dddall['Recall@10'],"R@20":dddall['Recall@20'],
            "H@10":dddall['Hitrate@10'],"H@20":dddall['Hitrate@20'],
            "N@10":dddall['NDCG@10'],"N@20":dddall['NDCG@20']
                     }
        return_res = {k:round(v,4) for k,v in return_res.items()}
        
        print('\ttime cost',time.time()-time_start,'s')
        print(pd.DataFrame({k:[v] for k,v in return_res.items()}))