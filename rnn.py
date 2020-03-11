from statsmodels.tsa.arima_model import ARIMA
from plot_hit_rates import _read_hit_rates

if __name__ == '__main__':
    basedir = "C:/Users/havar/Home/cache_simulation_results/"
    _disc_time, _hit_rates = _read_hit_rates(basedir + "res_01_r.csv", None)
    train = len(_hit_rates) * 0.8
    model = ARIMA(_hit_rates, 1)
    model_fit = model.fit(disp=False)
    pred = model_fit.predict()


    '''
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(_hit_rates.shape[1], 2)))
    model.add(LSTM(128, input_shape=(_hit_rates.shape[1], 2)))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(_hit_rates, trainy, epochs=2000, batch_size=10, verbose=2, shuffle=False)
    model.save_weights('LSTMBasic1.h5')
    '''