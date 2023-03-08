import darts
from darts.metrics import mape, smape, mse, mae, rmse
import sktime
from sktime.performance_metrics.forecasting import MeanSquaredScaledError


def multiple_metrics(validation, pred):
    mape_ = mape(validation, pred)
    smape_ = smape(validation, pred)
    mse_ = mse(validation, pred)
    mae_ = mae(validation, pred)
    rmse_ = rmse(validation, pred)

    return [mape_, smape_, mse_, mae_, rmse_]


def multiple_metrics_global(validation, pred, y_train):
    rmsse = MeanSquaredScaledError(square_root=True)

    mape_ = mape(validation, pred)
    smape_ = smape(validation, pred)
    mse_ = mse(validation, pred)
    mae_ = mae(validation, pred)
    rmse_ = rmse(validation, pred)
    rmsse_ = rmsse(validation.values()[:len(pred)], pred.values(), y_train=y_train.values())

    return [mape_, smape_, mse_, mae_, rmse_, rmsse_]
