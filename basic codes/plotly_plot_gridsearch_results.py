import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

def plot_gridsearch_results(dt, name, by_train = True):
    train_metric_f1 = 'mean_train_f1' 
    test_metric_f1 = 'mean_test_f1'
    train_metric_precision = 'mean_train_precision' 
    test_metric_precision = 'mean_test_precision'
    train_metric_recall = 'mean_train_recall' 
    test_metric_recall = 'mean_test_recall'
    train_metric_accuracy = 'mean_train_accuracy'
    test_metric_accuracy = 'mean_test_accuracy'


    y1 = np.array(dt[test_metric_f1]).reshape(1,-1)[0]
    y2 = np.array(dt[train_metric_f1]).reshape(1,-1)[0]
    y3 = np.array(dt[test_metric_precision]).reshape(1,-1)[0]
    y4 = np.array(dt[test_metric_recall]).reshape(1,-1)[0]
    y5 = np.array(dt[train_metric_precision]).reshape(1,-1)[0]
    y6 = np.array(dt[train_metric_recall]).reshape(1,-1)[0]
    y7 = np.array(dt[test_metric_accuracy]).reshape(1,-1)[0]
    y8 = np.array(dt[train_metric_accuracy]).reshape(1,-1)[0]
    
    indexes = np.argsort(y2) if by_train else np.argsort(y1)
    params_pairs = np.array([[i,j,k] for i,j,k in zip(dt['param_max_depth'].tolist(),
                                                  dt['param_max_features'].tolist(),
                                                  dt['param_min_samples_leaf'].tolist())])
    params_pairs = params_pairs[indexes]

    x = list(range(len(params_pairs)))
    y1 = y1[indexes]
    y2 = y2[indexes]
    y3 = y3[indexes]
    y4 = y4[indexes]
    y5 = y5[indexes]
    y6 = y6[indexes]
    y7 = y7[indexes]
    y8 = y8[indexes]
    #y1 = np.sort(np.array(dt[test_metric]).reshape(1,-1)[0])
    #y2 = np.array(dt[train_metric]).reshape(1,-1)[0][indexes]



    trace_high = go.Scatter(
                    x=x,
                    y=y1,
                    name = "F1 test",
                    line = dict(color = 'teal'),
                    opacity = 1)

    trace_low = go.Scatter(
                    x=x,
                    y=y2,
                    name = "F1 train",
                    line = dict(color = '#7F7F7F'),
                    opacity = 1)
    trace_precision_test = go.Scatter(
                    x=x,
                    y=y3,
                    name = "Precision test",
                    line = dict(color = 'paleturquoise'),
                    opacity = 0.6)
    trace_recall_test = go.Scatter(
                    x=x,
                    y=y4,
                    name = "Recall test",
                    line = dict(color = 'lightblue'),#17BECF
                    opacity = 0.6)
    trace_precision_train = go.Scatter(
                    x=x,
                    y=y5,
                    name = "Precision train",
                    line = dict(color = 'khaki'),
                    opacity = 0.8)
    trace_recall_train = go.Scatter(
                    x=x,
                    y=y6,
                    name = "Recall train",
                    line = dict(color = 'peachpuff'),
                    opacity = 0.8)
    trace_accuracy_test = go.Scatter(
                    x=x,
                    y=y7,
                    name = "Accuracy test",
                    line = dict(color = 'lightcoral'),
                    opacity = 0.5)
    trace_accuracy_train = go.Scatter(
                    x=x,
                    y=y8,
                    name = "Accuracy train",
                    line = dict(color = 'palevioletred'),
                    opacity = 0.5)


    data = [trace_low, trace_precision_train, trace_recall_train, trace_accuracy_train,
            trace_high, trace_precision_test, trace_recall_test,trace_accuracy_test]

    layout = dict(
        title = "Gridsearch results. F1 metrics. %s."%(name),
        xaxis=dict(
        tickvals=x,
        ticktext=params_pairs
        )
    )
    plotly.offline.init_notebook_mode(connected=True)
    fig = dict(data=data, layout=layout)
    plotly.offline.iplot(fig, filename = "Gridsearch results, %s. %s "%(name, 'By_test' if by_train else 'By_test'))