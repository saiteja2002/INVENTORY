from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
import os
import warnings
import pandas as pd
#parameters of the model to create (compile and .fit method parameters)
base_model={
    "architecture_lenght":4,
    "optimizer":"sgd",
    "learning_rate":0.001,
    "decay_rate":0.0,
    "momentum":0.002,
    "dropout":0.25,
    "loss_function":"binary_crossentropy",
    "target_classes":2

}

class Generate_Sample_Space_and_Create_and_train_model(object):

    def __init__(self):
        self.basemodel_optimizer = base_model["optimizer"]
        self.basemodel_target_classes = base_model["target_classes"]
        self.basemodel_lr = base_model["learning_rate"]
        self.basemodel_decay = base_model["decay_rate"]
        self.basmodel_dropout = base_model["dropout"]
        self.basemodel_momentum = base_model["momentum"]

        self.basemodel_target_classes = base_model["target_classes"]
        self.id_layerconf = self.id_layerconfig()
        self.metrics = ['accuracy']
        self.basemodel_loss_func = base_model["loss_function"]

        self.trainedlayer_weights_file = 'trainedlayer_shared_weights.pkl'
        self.trainedlayer_weights = pd.DataFrame({'layer_id': [], 'weights': []})
        if not os.path.exists(self.trainedlayer_weights_file):
            self.trainedlayer_weights.to_pickle(self.trainedlayer_weights_file)

    #creates a sample space in the form of dictionary with key as layer_id(encoding) and value as layer configuration
    #contains every possible layer configuration with the given parameters
    def id_layerconfig(self):
        # stores whether the encoding is a conv layer or a neural layer
        conv_or_neural={}
        #possible nodes and activations of conv layer
        nodes = [8,16, 32]
        act_funcs = ['tanh', 'relu', 'elu']
        #stores every possible layer configurations
        possible_layer_config_conv = []
        #stores the layer id of every possible layer
        layer_id_conv = []
        #fills the conv_or_neural,layer_id_conv,possible_layer_cpnfig
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                possible_layer_config_conv.append((nodes[i], act_funcs[j],"conv2d"))
                layer_id_conv.append(len(act_funcs) * i + j + 1)
                conv_or_neural[len(act_funcs) * i + j + 1]="conv"

        max_til=max(layer_id_conv)#stores last id given to conv layers
        # stores every possible layer configurations
        shallow_node=[8,32,64]
        shallow_act_funcs=[ 'tanh', 'relu', 'elu']
        # stores every possible shallow layer configurations
        layer_params_shallow = []
        # stores the layer id of every possible shallow layer
        layer_id_shallow = []
        # fills the conv_or_neural,layer_id_shallow,layer_params_shallow
        for i in range(len(shallow_node)):
            for j in range(len(shallow_act_funcs)):
                layer_params_shallow.append((shallow_node[i], shallow_act_funcs[j],"shallowneural"))
                layer_id_shallow.append(len(shallow_act_funcs) * i + j + 1+max_til)
                conv_or_neural[len(act_funcs) * i + j + 1+max_til] = "neural"
        possible_layer_config_conv=possible_layer_config_conv+layer_params_shallow#combines every possible layers of conv and shallow
        layer_id_conv=layer_id_conv+layer_id_shallow#same as the above line but for shallow
        id_layerconfigg = dict(zip(layer_id_conv, possible_layer_config_conv))#creates a dictionary with key as layerid and value as layer configuration
        id_layerconfigg[len(id_layerconfigg) + 1] = (('dropout'))#we want to give the model a dropuout layer as well in the final model architeture
        #the final layer depeends on binary or categorical classification
        if self.basemodel_target_classes == 2:
            id_layerconfigg[len(id_layerconfigg) + 1] = (self.basemodel_target_classes - 1, 'sigmoid', "shallowneural")
        else:
            id_layerconfigg[len(id_layerconfigg) + 1] = (self.basemodel_target_classes, 'softmax', "shallowneural")
        #add the dropu out and final layer to conv_or_shallow dict ,shallow and neural mean the same
        conv_or_neural[len(id_layerconfigg)+1]="droppout"
        conv_or_neural[len(id_layerconfigg)+2]="neural"
        return id_layerconfigg

    # given a sequence of ids converts it into a seqquence of their layers
    def id_to_config(self, sequence):
        id = list(self.id_layerconf.keys())
        layer_configg = list(self.id_layerconf.values())
        id_to_configg = []
        for k in sequence:
            id_to_configg.append(layer_configg[id.index(k)])
        return id_to_configg
    #given a sequence of encoded models(layerids) convert into a model and return it
    def sequence_to_model(self, sequence, basemodel_input_shape):
        basemodel_layer_configs = self.id_to_config(sequence)
        model = Sequential()
        for i, layer in enumerate(basemodel_layer_configs):
                if i == 0:
                    model.add(Conv2D(filters=layer[0], kernel_size=(11,11), strides=(4,4), activation=layer[1], input_shape=basemodel_input_shape))
                    model.add(MaxPooling2D(pool_size=(9, 9), strides=(3, 3)))
                elif layer =='dropout':
                    model.add(Dropout(self.basmodel_dropout))
                else:
                    if layer[-1]=="conv2d":
                        model.add(Conv2D(filters=layer[0], kernel_size=(3,3), activation=layer[1]))
                        model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))
                    else:
                        model.add(Dense(units=layer[0], activation=layer[1]))

                if i==1:
                    model.add(Flatten(name='flatten'))
        # model.summary()
        return model

# {1: (16, 'tanh', 'conv2d'), 2: (16, 'relu', 'conv2d'), 3: (16, 'elu', 'conv2d'), 4: (32, 'tanh', 'conv2d'), 5: (32, 'relu', 'conv2d'), 6: (32, 'elu', 'conv2d'), 7: (32, 'tanh', 'shallowne
# ural'), 8: (32, 'relu', 'shallowneural'), 9: (32, 'elu', 'shallowneural'), 10: (64, 'tanh', 'shallowneural'), 11: (64, 'relu', 'shallowneural'), 12: (64, 'elu', 'shallowneural'), 13: 'dro
# pout', 14: (1, 'sigmoid', 'shallowneural')}
    def model_to_sharedweightsfile(self, model):
        """
            one-shot approach. If the layers in the current model weights are not stored then then future models it's been stored.
            input params:
        """
        #noting down differnt layers whihc can have weights(exclude dropuout and pooling cause no weights)
        basemodel_layer_configs = [tuple(['input'])]
        for layer in model.layers:
            if 'flatten' in layer.name:
                basemodel_layer_configs.append(('flatten'+str(layer.output_shape)))
            elif 'dropout' not in layer.name and "max_pooling2d" not in layer.name:
                if "conv2d" in layer.name:
                    basemodel_layer_configs.append((layer.get_config()['filters'], layer.get_config()['activation'],"conv2d"))
                else:
                    basemodel_layer_configs.append((layer.get_config()['units'], layer.get_config()['activation'],"shallowneural"))
        baseModel_config_ids = []
        #making pairs
        #each pair represnt pair[0]==previous layer shaep pair[1]=current layer shape
        for i in range(1, len(basemodel_layer_configs)):
            if type(basemodel_layer_configs[i])!=str:
                baseModel_config_ids.append((basemodel_layer_configs[i - 1], basemodel_layer_configs[i]))
        # print(baseModel_config_ids)

        #now if there exists a layer in the shared train file with same configurarions as the current one ,the weights
        # are updated if not the weights are uploaded
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name and "max_pooling2d" not in layer.name and "flatten" not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                layer_ids = self.trainedlayer_weights['layer_id'].values
                layers_found = []
                for i in range(len(layer_ids)):
                    if baseModel_config_ids[j] == layer_ids[i]:
                        layers_found.append(i)
                #if no weights found upload them to file
                if len(layers_found) == 0:
                    print(baseModel_config_ids[j])
                    print("uploading weights to shared weights file")
                    self.trainedlayer_weights = self.trainedlayer_weights.append({'layer_id': baseModel_config_ids[j],
                                                                      'weights': layer.get_weights()},

                                                                                 ignore_index=True)
                #else update the found weights
                else:
                        self.trainedlayer_weights.at[layers_found[0], 'weights'] = layer.get_weights()
                j += 1
        self.trainedlayer_weights.to_pickle(self.trainedlayer_weights_file)



    def sharedweightsfile_to_model(self, model):
        """
            one-shot approach. If the layers in the current model weights are already trained by some other model then
            instead of training from scratch those trained weights are used.
        """
        # noting down differnt layers whihc can have weights(exclude dropuout and pooling cause no weights)
        basemodel_layer_configs = [tuple(['input'])]
        for layer in model.layers:
            if 'flatten' in layer.name:
                basemodel_layer_configs.append('flatten'+str(layer.output_shape))
            elif 'dropout' not in layer.name and "max_pooling2d" not in layer.name:
                if "conv2d" in layer.name:
                    basemodel_layer_configs.append((layer.get_config()['filters'], layer.get_config()['activation'],"conv2d"))
                else:
                    basemodel_layer_configs.append((layer.get_config()['units'], layer.get_config()['activation'],"shallowneural"))
        basemodel_config_ids = []
        #making pairs
        #each pair represnt pair[0]==previous layer shaep pair[1]=current layer shape
        for i in range(1, len(basemodel_layer_configs)):
            if type(basemodel_layer_configs[i])!=str:
                basemodel_config_ids.append((basemodel_layer_configs[i - 1], basemodel_layer_configs[i]))
        # print(basemodel_config_ids)
        j = 0
        # now if there exists a layer in the shared train file with same configurarions as the current one ,the weights
        # are transfered to the layer to be trained
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name and "max_pooling2d" not in layer.name and "flatten" not in layer.name:
                # print(layer.name)
                warnings.simplefilter(action='ignore', category=FutureWarning)
                layer_ids = self.trainedlayer_weights['layer_id'].values
                found_index = []
                for i in range(len(layer_ids)):
                    if basemodel_config_ids[j] == layer_ids[i]:
                        found_index.append(i)
                #if found then the layers weights are transfered
                if len(found_index) > 0:
                        print("Transferring weights from shared weights file for layer:", basemodel_config_ids[j])
                        layer.set_weights(self.trainedlayer_weights['weights'].values[found_index[0]])
                j += 1
    #trained the compiled model with given inputs
    def train_basemodel(self, model, x_data, y_data, epochs, callbacks=None):
            # if sgd use this as sgd requires momentum value
            if self.basemodel_optimizer == 'sgd':
                optimizer = optimizers.SGD(learning_rate=self.basemodel_lr, decay=self.basemodel_decay,
                                           momentum=self.basemodel_momentum)
            else:
                # only way to pass optimizer as a string with custom parameters
                optimizer = getattr(optimizers, self.basemodel_optimizer)(learning_rate=self.basemodel_lr,
                                                                          decay=self.basemodel_decay)
            model.compile(loss=self.basemodel_loss_func, optimizer=optimizer, metrics=self.metrics)
            print('TRAINING THE BASE MODEL')
            self.sharedweightsfile_to_model(model)
            history = model.fit(x_data,
                                epochs=epochs,
                                validation_data=y_data,
                                callbacks=callbacks,
                                verbose=1)
            self.model_to_sharedweightsfile(model)
            return history

