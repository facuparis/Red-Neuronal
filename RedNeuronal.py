import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import json
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


archivos = 96 # contador de archivos n_dev y n_resultados

#Especificaciones de la interpolacion
inicio = 2.0
final = 28.0
intervalo = 1

#Numero de x_n
num_xn = int((final - inicio) / intervalo)

def Abrime_resultados(argumento):
    """Devuelve un diccionario con Lf,Temperatura,viscosidad,k,n,Patm,Pcap (desordenado)"""
    fname = './Datos_curvasOK/{0}_resultados.log'.format(argumento)
    fi = open(fname, 'r')  # Apertura del archivo
    lines = fi.read()  # Lectura: el resultado es una lista
    dic_result = json.loads(lines)
    fi.close()
    return dic_result




# Uso los archivos _dev
def lectura(nombre):
    t,x = np.loadtxt('./Datos_curvasOK/{0}_dev.dat'.format(nombre),unpack=True)
    # Interpolacion de orden cubica
    t_inter = np.arange(inicio, final, intervalo) #Utilizo desde los 2 a los 20 segundos
    interpolacion =  interpolate.interp1d(t,x,kind='cubic')
    x_inter = interpolacion(t_inter)
    return x_inter

#Leo todos los archivos _dev, 96 en total

lista_dev = []


#lista en donde cada elemento es una lista n_dev
for i in range(1,archivos+1):
    lista_dev.append(lectura(i))


lista_xn = []
for n in range(0,num_xn +1):
    lista_xn.append([])

#Creo la lista [x_1:[], x_2: [], ..] (solo pondre las listas en cada elemento para despues usarlas con dic_Inputs)
for i in range(0,num_xn):
    for m in range(0,archivos):
        lista_xn[i].append(lista_dev[m][i])


# print(lista_xn)

# Creo el diccionario de inputs {x_1:[], x_2: [], ..} x_n , donde n es el numero total de puntos interpolados
dic_Inputs = {}
#de 0 a numero total de puntos interpolados
for i in range(0, num_xn):
        dic_Inputs['x_' + str(i)] = lista_xn[i]


#print(dic_Inputs)


#Lista de nombre de features [x1,x2,x3,..] para pasarle columns a pandas
features = []
for i in range(0,num_xn):
    features.append('x_{}'.format(i))

#Agrego las columnas a mano Patm y k
features.append('x_Patm')
features.append('k')

#Leo k, Patm y n.
lista_k = []
lista_Patm = []
lista_exponente_n = []
for i in range(1,archivos+1):
    lista_k.append(Abrime_resultados(i)['k'])
    lista_Patm.append(Abrime_resultados(i)['Patm'])
    lista_exponente_n.append(Abrime_resultados(i)['n'])

#Abro con pandas, primero agrego a dic_Inputs k,n,Patm..

dic_Inputs['x_Patm'] = np.array(lista_Patm)/1000000 #Normalizo la presion atmosferica
dic_Inputs['k'] = np.array(lista_k)

"""Como por ahora todos los casos son con n = 1, dejo afuera n del dic_Inputs"""

df = 1000*pd.DataFrame(dic_Inputs, columns = features)

print(df.head())
"""******************************************************************************************************"""


"""Neural Network regression"""
#Defino las variables

X = df.as_matrix()[:,:-1]
Y = 1000*dic_Inputs['k']

#Separo las variables en training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

#Artificial Neural Network
learning_rate = 0.001
training_epochs = 100000
cost_history = np.empty(shape=[1],dtype = float)
n_dim = X.shape[1]
n_class = 1 # NO estoy seguro...  numero de output..
model_path = "C:\\Users\\Facundo Paris\\PycharmProjects\\TensorFlowww\\RedNeuronal.py"

total_len = X_train.shape[0]
batch_size = 10
display_step = 1
dropout_rate = 0.9


print('n_dim', n_dim)

n_hidden_1 = 10
n_hidden_2 = 10
n_hidden_3 = 10
n_hidden_4 = 10

x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros(n_class))
y = tf.placeholder(tf.float32,[None])

def multilayer_perceptron(x, weights, biases):
    #Hidden Layer
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)


    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    #Output layer
    out_layer = tf.matmul(layer_4,weights['out']) + biases['out']
#    out_layer = tf.reduce_sum(out_layer) #Sumo los elementos de lo de arriba

    return out_layer

#Defino los weights and biases

weights = {

    'h1': tf.Variable(tf.random_normal([n_dim, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_class], 0, 0.1))
#     'h1': tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
#     'h2': tf.Variable(tf.truncated_normal([n_hidden_1],n_hidden_2)),
#     'h3': tf.Variable(tf.truncated_normal([n_hidden_2],n_hidden_3)),
#     'h4': tf.Variable(tf.truncated_normal([n_hidden_3], n_hidden_4)),
#     'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
 }

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

#Inicializo todas las variables

# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()

#Llamo al modelo
pred = multilayer_perceptron(x,weights,biases)

#Defino la funcion de costo y el optimizador
cost = tf.reduce_mean(tf.square(tf.transpose(pred)-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# sess = tf.Session()
# sess.run(init)

#Calculo la funcion de costo y la accuracy para cada epoch

mse_history = []
accuracy_history = []
cost_history = []
prediccion = []


# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            #batch_x = X_train[i*batch_size:(i+1)*batch_size]
            #batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: X_train,
                                                          y: Y_train})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = Y_train
        estimate = p
        err = label_value-estimate
        print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in range(10):
                print ("label value:", label_value[i], \
                    "estimated value:", estimate[i])
            print ("[*]============================")

    print ("Optimization Finished!")

    print('cost:', c)
    #Test model
    # correct_prediction = tf.equal(tf.argmax(pred), tf.argmax(y))
    #Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))





