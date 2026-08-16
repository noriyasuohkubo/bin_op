from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model_dir_lstm = "/app/model/bin_op/"
save_dir = "/app/model/h5/"

models =[
    ['MN887-39', 'MN887-39',],
    ['MN885-6','MN885-6',],
    ['USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.25_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub1_MN715-40','MN715-40'],
    ['USDJPY_LT1_M7_LSTM1_B1_T4_I1-5-30_IL300-300-240_LU30-30-24_DU48-24-12_BNL2_BDIV0.5_201701_202303_L-RATE0.0005_LT1_ADAM_DA4_RA8_RRA9_d1_1_d1_ehd1-1_eld1-1_23-SEP_OT-d_OD-c_BS5120_SD0_SHU1_EL20-21-22_ub3_MN714-36','MN714-36'],



]

for model in models:
    m , name = model
    model_tmp = load_model(model_dir_lstm + m,
                           custom_objects={"root_mean_squared_error": root_mean_squared_error, })
    # Check its architecture
    model_tmp.summary()

    #Save the model into h5 format
    model_tmp.save(save_dir + name + '.h5')