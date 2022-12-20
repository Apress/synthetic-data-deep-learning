import numpy as npy
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn.model_selection as sms
from sklearn import ensemble
from sklearn import metrics
from tensorflow import keras


wine_data = pnd.read_csv('C:/Users/Esma/Desktop/Necmi Hoca Kitap/WineQTNew.csv')
x_var_name=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residua_ sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
y_var_name=['quality']
x_wine =wine_data[x_var_name]
y_wine =wine_data[y_var_name]
x_r_trn, x_r_tst, y_r_trn, y_r_tst=sms.train_test_split(x_wine, y_wine,random_state=40)
random_forest=ensemble.RandomForestClassifier(n_estimators=200)
random_forest.fit(x_r_trn, y_r_trn.values.ravel())
y_r_prd=random_forest.predict(x_r_tst)

print ("Acurracy: ", metrics.accuracy_score(y_r_tst, y_r_prd))
print ("Classification Result : ",metrics.classification_report(y_r_tst, y_r_prd))


def hidden_genset(hidden_size, number_s):
    ent_x = npy.random.randn(hidden_size * number_s)
    ent_x = ent_x .reshape(number_s, hidden_size)
    return ent_x 

def fake_generate(genset, hidden_size, number_s):
    ent_x = hidden_genset(hidden_size, number_s)
    fake_x = genset.predict(ent_x)
    fake_y = npy.zeros((number_s, 1))
    
    return fake_x, fake_y

def real_generate(s_number):
    real_x = wine_data.sample(s_number)
    real_y = npy.ones((s_number, 1))
    return real_x, real_y


def gan_genset(hidden_size, n_outputs=12):
    gan_mdl = keras.models.Sequential()
    gan_mdl.add(keras.layers.Dense(16, activation='selu',  kernel_initializer='random_normal', input_dim=hidden_size))
    gan_mdl.add(keras.layers.Dense(32, activation='selu'))
    gan_mdl.add(keras.layers.Dense(n_outputs, activation='softmax'))
    return gan_mdl


genset1 = gan_genset(13, 12)
genset1.summary()


def gan_sorter(in_number=12):
    gan_mdl = keras.models.Sequential()
    gan_mdl.add(keras.layers.Dense(30, activation='selu', kernel_initializer='random_normal', input_dim=in_number))
    gan_mdl.add(keras.layers.Dense(60, activation='selu'))
    gan_mdl.add(keras.layers.Dense(1, activation='sigmoid'))
    gan_mdl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return gan_mdl


sorter1 = gan_sorter(12)
sorter1.summary()

def gan_model(genset, sorter):
    sorter.trainable = False
    gan_mdl = keras.models.Sequential()
    gan_mdl.add(genset)
    gan_mdl.add(sorter)
    gan_mdl.compile(loss='binary_crossentropy', optimizer='adam')
    return gan_mdl

def plot_history(sorter_graph, genset_graph):
    plt.plot(sorter_graph, label='Sorter')
    plt.plot(genset_graph, label='Genset')
    plt.show()
    plt.close()


def data_train(genset_mdl,sorter_model, model_gan, hidden_dim, n_epochs=1000, n_batch=140, n_eval=250):
    b_size = int(n_batch / 2)
    sorter_hist= []
    genset_hist= []

    for i in range(n_epochs):
        x_r, y_r = real_generate(b_size)
        x_f, y_f = fake_generate(genset_mdl, hidden_dim,b_size)
        r_loss_d, acc_real_d= sorter_model.train_on_batch(x_r, y_r)
        f_loss_d, acc_fake_d= sorter_model.train_on_batch(x_f, y_f)
        loss_value_d = 0.5 * npy.add(r_loss_d,f_loss_d)
        x_g_values = hidden_genset(hidden_dim, n_batch)
        y_g_values = npy.ones((n_batch, 1))
        fake_gloss = model_gan.train_on_batch(x_g_values, y_g_values)
        print('>%d_value, d1_value=%.4f, d2_value=%.4f d_value=%.4f g_value=%.4f' % (i+1, r_loss_d, f_loss_d, loss_value_d, fake_gloss))
        sorter_hist.append(loss_value_d)
        genset_hist.append(fake_gloss )
    plot_history(sorter_hist, genset_hist)
    genset_mdl.save('generated model of trained data.h5')



hidden_dim = 13
sorter = gan_sorter()
genset = gan_genset(hidden_dim)
new_g_model = gan_model(genset, sorter)
data_train(genset, sorter, new_g_model, hidden_dim)

gan_model_trained =keras.models.load_model('C:/Users/Esma/Desktop/Necmi Hoca Kitap/generated model of trained data.h5')
hidden_dots = hidden_genset(13, 800)
x_predict = gan_model_trained.predict(hidden_dots)
trained_fake_data= pnd.DataFrame(data=x_predict,  columns=['fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residua_ sugar','chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol','quality' ])
trained_fake_data .head()

quality_mean = trained_fake_data .quality.mean()
trained_fake_data['quality'] = trained_fake_data ['quality'] > quality_mean
trained_fake_data["quality"] = trained_fake_data ["quality"].astype(int)

features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residua_ sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
label = ['quality']
x_f_predicted =trained_fake_data [features]
y_f_predicted =trained_fake_data [label]

x_f_trn, x_f_tst, y_f_trn, y_f_tst = sms.train_test_split(x_f_predicted, y_f_predicted, random_state=99)
random_forest_fake = ensemble.RandomForestClassifier(n_estimators=200)
random_forest_fake.fit(x_f_trn,y_f_trn.values.ravel())
y_f_pred=random_forest_fake.predict(x_f_tst)
print("Fake data Accuracy ",metrics.accuracy_score(y_f_tst, y_f_pred))
print("Fake data Classification Result:",metrics.classification_report(y_f_tst, y_f_pred))



from table_evaluator import load_data, TableEvaluator
evaluation_table = TableEvaluator(wine_data,trained_fake_data)
evaluation_table.evaluate(target_col='quality')
evaluation_table.visual_evaluation()

