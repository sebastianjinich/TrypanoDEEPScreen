import plotly.express as px
from sklearn.metrics import roc_curve, auc, roc_auc_score
from find_max_similiarity import find_max_similiarity
import pandas as pd

def roc(true,pred,width=500, height=500):
    fpr, tpr, thresholds = roc_curve(true, pred)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'AUC={auc(fpr, tpr):.4f} AUC15={roc_auc_score(true, pred,max_fpr=0.15):.4f} AUC30={roc_auc_score(true, pred,max_fpr=0.30):.4f}',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=width, height=height
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(title_x=0.5)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig

def scores_label_histogram(prediction_df,score_coumn_name,label,nbins=50,width=700,height=500):
    predictions_hist = px.histogram(prediction_df,x=score_coumn_name,color=label,nbins=nbins,width=width,height=height)
    return predictions_hist

def tanimoto_datasets_comparison(reference,to):
    return px.histogram(find_max_similiarity(reference,to),"best_tanimoto",width=600,height=500,range_x=(0,1))

def tanimoto_similarity_2D_plot(train,test,validation,width=500,height=500):
    train_tanimoto_test_validation = train[["comp_id","bioactivity"]]

    test_train = find_max_similiarity(train,test)
    test_train = test_train.rename(columns={"best_tanimoto":"test"})

    val_train = find_max_similiarity(train,validation)
    val_train = val_train.rename(columns={"best_tanimoto":"validation"})

    train_tanimoto_test_validation = pd.merge(train_tanimoto_test_validation,test_train[["comp_id","test"]],on="comp_id")
    train_tanimoto_test_validation = pd.merge(train_tanimoto_test_validation,val_train[["comp_id","validation"]],on="comp_id")
    train_tanimoto_test_validation["bioactivity"] = train_tanimoto_test_validation["bioactivity"].astype(str)

    return px.scatter(train_tanimoto_test_validation,x="test",y="validation",color="bioactivity",width=width,height=height,range_x=(0,1),range_y=(0,1))
