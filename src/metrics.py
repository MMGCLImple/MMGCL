"""
we refer some code from:
https://github.com/microsoft/recommenders/blob/main/recommenders/evaluation/python_evaluation.py
"""
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from six import iteritems

def bootstrap(metric, r_pred, r_true, user_ids, item_ids, steps = 1):
    n = len(r_true)
    values = []
    print(f"bootstrap steps: {steps}")
    for i in range(steps):
        x = np.random.choice(n, size=n)
        values.append(metric(r_pred[x], r_true[x], user_ids[x], item_ids[x]))
    return values

def rmse(r_pred, r_true, user_ids, item_ids):
    """
    here we need to remove nan values
    """
    df = pd.DataFrame({'r_pred': r_pred, 'r_true': r_true, 'user_ids': user_ids, 'item_ids': item_ids})
    df = df.dropna()   # drop nan values 
    return np.mean((df['r_pred'] - df['r_true'])**2)**0.5

def get_top_k_items(
    dataframe,
    col_user="user_id",
    col_rating="rating_predict",
    k=5
):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.
    Note:
        If it is implicit rating, just append a column of constants to be
        ratings.
    Args:
        dataframe (pandas.DataFrame): DataFrame of rating data (in the format
        customerID-itemID-rating)
        col_user (str): column name for user
        col_rating (str): column name for rating
        k (int or None): number of items for each user; None means that the input has already been
        filtered out top k items and sorted by ratings and there is no need to do that again.
    Returns:
        pandas.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    if k is None:
        top_k_items = dataframe
    else:
        top_k_items = (
            dataframe.sort_values([col_user, col_rating], ascending=[True, False])
            .groupby(col_user, as_index=False)
            .head(k)
            .reset_index(drop=True)
        )
    # Add ranks
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items

def merge_ranking_true_pred(
    rating_true,
    rating_pred,
    col_user="user_id",
    col_item="item_id",
    col_rating="rating_true",
    col_prediction="rating_predict",
    relevancy_method="top_k",
    k=5):
    """Filter truth and prediction data frames on common users
    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user (optional)
    Returns:
        pandas.DataFrame, pandas.DataFrame, int: DataFrame of recommendation hits, sorted by `col_user` and `rank`
        DataFrame of hit counts vs actual relevant items per user number of unique user ids
    """

    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user])) 
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)] 
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    n_users = len(common_users)

    # Return hit items in prediction data frame with ranking information. This is used for calculating NDCG and MAP.
    # Use first to generate unique ranking values for each item. This is to align with the implementation in
    # Spark evaluation metrics, where index of each recommended items (the indices are unique to items) is used
    # to calculate penalized precision of the ordered items.
    if relevancy_method == "top_k":
        top_k = k
    elif relevancy_method is None:
        top_k = None
    else:
        raise NotImplementedError("Invalid relevancy_method")
    df_hit = get_top_k_items( 
        dataframe=rating_pred_common,
        col_user=col_user,
        col_rating=col_prediction,
        k=top_k,
    )
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge( 
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
    )

    return df_hit, df_hit_count, n_users


#https://github.com/microsoft/recommenders/blob/main/recommenders/evaluation/python_evaluation.py
def recall_at_k_metric(r_pred, r_true, user_ids, item_ids, k=5):
    print(f"\tRECALL at top {k}")
    rating_pred = zip(r_pred, user_ids, item_ids)
    column_names_pred = ["rating_predict", "user_id", "item_id"]
    pred_df = pd.DataFrame(rating_pred, columns=column_names_pred)
    
    rating_true = zip(r_true, user_ids, item_ids)
    column_names_true = ["rating_true", "user_id", "item_id"]
    true_df = pd.DataFrame(rating_true, columns=column_names_true)
    true_df = true_df.dropna().reset_index(drop=True)
    
    return _recall_at_k(true_df, pred_df, k=k)

def precision_at_k_metric(r_pred, r_true, user_ids, item_ids, k=5):
    print(f"\tPRECISION at top {k}")
    rating_pred = zip(r_pred, user_ids, item_ids)
    column_names_pred = ["rating_predict", "user_id", "item_id"]
    pred_df = pd.DataFrame(rating_pred, columns=column_names_pred)
    
    rating_true = zip(r_true, user_ids, item_ids)
    column_names_true = ["rating_true", "user_id", "item_id"]
    true_df = pd.DataFrame(rating_true, columns=column_names_true)
    true_df = true_df.dropna().reset_index(drop=True)
    
    return _precision_at_k(true_df, pred_df, k=k)

def ndcg_at_k_metric(r_pred, r_true, user_ids, item_ids, k=5):
    print(f"\tPRECISION at top {k}")
    rating_pred = zip(r_pred, user_ids, item_ids)
    column_names_pred = ["rating_predict", "user_id", "item_id"]
    pred_df = pd.DataFrame(rating_pred, columns=column_names_pred)
    
    rating_true = zip(r_true, user_ids, item_ids)
    column_names_true = ["rating_true", "user_id", "item_id"]
    true_df = pd.DataFrame(rating_true, columns=column_names_true)
    true_df = true_df.dropna().reset_index(drop=True)
    
    return _ndcg_at_k(true_df, pred_df, k=k)

def hiteRatio_at_k_metric(r_pred, r_true, user_ids, item_ids, k=5):
    """
    see: http://t.csdn.cn/gVHga
    """
    print(f"\tPRECISION at top {k}")
    rating_pred = zip(r_pred, user_ids, item_ids)
    column_names_pred = ["rating_predict", "user_id", "item_id"]
    pred_df = pd.DataFrame(rating_pred, columns=column_names_pred)
    
    rating_true = zip(r_true, user_ids, item_ids)
    column_names_true = ["rating_true", "user_id", "item_id"]
    true_df = pd.DataFrame(rating_true, columns=column_names_true)
    true_df = true_df.dropna().reset_index(drop=True)
    
    
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=true_df,
        rating_pred=pred_df,
        col_user="user_id", col_item="item_id", col_prediction="rating_predict", col_rating="rating_true",
        relevancy_method="top_k",
        k=k,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return df_hit_count['hit'].sum() / df_hit_count['actual'].sum()



def metric_at_once(r_pred, r_true, user_ids, item_ids, k=5):
    print(f"\tmetric_at_once at top {k}")
    rating_pred = zip(r_pred, user_ids, item_ids)
    column_names_pred = ["rating_predict", "user_id", "item_id"]
    pred_df = pd.DataFrame(rating_pred, columns=column_names_pred)
    
    rating_true = zip(r_true, user_ids, item_ids)
    column_names_true = ["rating_true", "user_id", "item_id"]
    true_df = pd.DataFrame(rating_true, columns=column_names_true)
    true_df = true_df.dropna().reset_index(drop=True)
    
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=true_df,
        rating_pred=pred_df,
        col_user="user_id",
        col_item="item_id",
        col_rating="rating_true",
        col_prediction="rating_predict",
        relevancy_method="top_k",
        k=k,
    )
    
    ################################################precision
    if df_hit.shape[0] == 0:
        _val_precision = 0.0
    else:
        _val_precision = (df_hit_count["hit"] / k).sum() / n_users
    ################################################recall
    if df_hit.shape[0] == 0:
        _val_recall = 0.0
    else:
        _val_recall = (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users
    ################################################hitrate
    if df_hit.shape[0] == 0:
        _val_hitrate = 0.0
    else:
        _val_hitrate = df_hit_count['hit'].sum() / df_hit_count['actual'].sum()
    ################################################ndcg
    if df_hit.shape[0] == 0:
        _val_ndcg = 0.0
    else:
        df_dcg = df_hit.merge(pred_df, on=["user_id", "item_id", ]).merge(
            true_df, on=["user_id", "item_id", ], how="outer", suffixes=("_left", None)
        )
        df_dcg["rel"] = 1
        discfun = np.log
        # Calculate the actual discounted gain for each record
        df_dcg["dcg"] = df_dcg["rel"] / discfun(1 + df_dcg["rank"])

        # Calculate the ideal discounted gain for each record
        df_idcg = df_dcg.sort_values(["user_id", "item_id",], ascending=False)
        df_idcg["irank"] = df_idcg.groupby("user_id", as_index=False, sort=False)["rating_true"].rank("first", ascending=False)
        df_idcg["idcg"] = df_idcg["rel"] / discfun(1 + df_idcg["irank"])

        # Calculate the actual DCG for each user
        df_user = df_dcg.groupby("user_id", as_index=False, sort=False).agg({"dcg": "sum"})

        # Calculate the ideal DCG for each user
        df_user = df_user.merge(
            df_idcg.groupby("user_id", as_index=False, sort=False)
            .head(k)
            .groupby("user_id", as_index=False, sort=False)
            .agg({"idcg": "sum"}),
            on="user_id",
        )

        # DCG over IDCG is the normalized DCG
        df_user["ndcg"] = df_user["dcg"] / df_user["idcg"]
        _val_ndcg = df_user["ndcg"].mean()

    return {f"Precision@{k}":_val_precision, 
            f"Recall@{k}":_val_recall, 
            f"Hitrate@{k}":_val_hitrate, 
            f"NDCG@{k}":_val_ndcg}

def _ndcg_at_k(rating_true, rating_pred, 
              col_user="user_id", col_item="item_id", col_prediction="rating_predict", col_rating="rating_true",
    relevancy_method="top_k", k=5, score_type="binary", discfun_type="loge",
    **kwargs
):
    df_hit, _, _ = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_dcg = df_hit.merge(rating_pred, on=[col_user, col_item]).merge(
        rating_true, on=[col_user, col_item], how="outer", suffixes=("_left", None)
    )

    if score_type == "binary":
        df_dcg["rel"] = 1
    elif score_type == "raw":
        df_dcg["rel"] = df_dcg[col_rating]
    elif score_type == "exp":
        df_dcg["rel"] = 2 ** df_dcg[col_rating] - 1
    else:
        raise ValueError("score_type must be one of 'binary', 'raw', 'exp'")

    if discfun_type == "loge":
        discfun = np.log
    elif discfun_type == "log2":
        discfun = np.log2
    else:
        raise ValueError("discfun_type must be one of 'loge', 'log2'")

    # Calculate the actual discounted gain for each record
    df_dcg["dcg"] = df_dcg["rel"] / discfun(1 + df_dcg["rank"])

    # Calculate the ideal discounted gain for each record
    df_idcg = df_dcg.sort_values([col_user, col_rating], ascending=False)
    df_idcg["irank"] = df_idcg.groupby(col_user, as_index=False, sort=False)[
        col_rating
    ].rank("first", ascending=False)
    df_idcg["idcg"] = df_idcg["rel"] / discfun(1 + df_idcg["irank"])

    # Calculate the actual DCG for each user
    df_user = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})

    # Calculate the ideal DCG for each user
    df_user = df_user.merge(
        df_idcg.groupby(col_user, as_index=False, sort=False)
        .head(k)
        .groupby(col_user, as_index=False, sort=False)
        .agg({"idcg": "sum"}),
        on=col_user,
    )

    # DCG over IDCG is the normalized DCG
    df_user["ndcg"] = df_user["dcg"] / df_user["idcg"]
    return df_user["ndcg"].mean()


def _recall_at_k(
    rating_true,
    rating_pred,
    col_user="user_id",
    col_item="item_id",
    col_rating="rating_true",
    col_prediction="rating_predict",
    relevancy_method="top_k",
    k=5,
    **kwargs
):
    """Recall at K.
    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
    Returns:
        float: recall at k (min=0, max=1). The maximum value is 1 even when fewer than
        k items exist for a user in rating_true.
    """
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users


def _precision_at_k(
    rating_true,
    rating_pred,
    col_user="user_id",
    col_item="item_id",
    col_rating="rating_true",
    col_prediction="rating_predict",
    relevancy_method="top_k",
    k=5,
    **kwargs
):
    """Precision at K.
    Note:
        We use the same formula to calculate precision@k as that in Spark.
        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.precisionAt
        In particular, the maximum achievable precision may be < 1, if the number of items for a
        user in rating_pred is less than k.
    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
    Returns:
        float: precision at k (min=0, max=1)
    """
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users
