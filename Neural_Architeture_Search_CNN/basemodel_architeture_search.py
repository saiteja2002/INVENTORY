from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from rnn_model_createtraindata_andfit import RNN_model_createtraindata_andfit
import keras.backend as backend


loop={
    "epochs_for_sampling":1,
    "number_of_samples_genrate":4,
    "epochs_to_train_base_model":1,
    "epoch_for_NAS":20,
    "alpha_rewardfunction":0.9
}
target_classes=2
architecture_len=4


class basemodel_architeture_search(RNN_model_createtraindata_andfit):
    #initializing required paramter
    def __init__(self, x, y):
        super().__init__()
        self.x_data = x
        self.y_data = y
        self.basemodel_target_classes = target_classes
        self.number_of_search_loops = loop["epochs_for_sampling"]
        self.number_of_seqences_generate_per_epoch = loop["number_of_samples_genrate"]
        self.rnnmodel_training_epochs = loop["epoch_for_NAS"]
        self.basemodel_training_epochs =loop["epochs_to_train_base_model"]
        self.rnn_loss_function_alpha = loop["alpha_rewardfunction"]
        self.basemodel_data = []
        self.rnnmodel_data = 'model_data_generated_sofar.pkl'
        #create rnnmodel
        self.rnn_model_inputshape = (1, architecture_len - 1)
        self.rnn_model = self.RNNModel_genration(self.rnn_model_inputshape)
    #function for training and returning history of current generated sequence
    def basemodel_training(self, sequence):
        model = self.sequence_to_model(sequence, (150,150,3))
        history = self.train_basemodel(model, self.x_data, self.y_data, self.basemodel_training_epochs)
        return history
    #storing the data from a generated sequence after training
    def basemodel_seq_and_valacc(self, sequence, history, rnnmodel_pred_valacc):
            val_acc = np.ma.average(history.history['val_accuracy'],
                                    weights=np.arange(1, len(history.history['val_accuracy']) + 1),
                                    axis=-1)

            self.basemodel_data.append([sequence, val_acc, rnnmodel_pred_valacc])
            print('validation accuracy of current architeture of base model: ', val_acc)
    #create training trainnig data from genrrated sequneces for rnn model
    def rnnmodel_training_data(self, seq):
        generated_seq = pad_sequences(seq, maxlen=self.max_len, padding='post')#padding them to max len
        prev_layer = generated_seq[:, :-1].reshape(len(generated_seq), 1, self.max_len - 1)#taking x as all the layers except last one
        last_layer_one_hot_form = to_categorical(generated_seq[:, -1], self.rnnmodel_predictclasses)#y as the last layer
        val_acc_target = [item[1] for item in self.basemodel_data]#actual val_acc of models
        return prev_layer, last_layer_one_hot_form, val_acc_target


    def rnnmodel_loss_function(self,target, output):
        punish_accuracy = 0.7
        reward=[]
        for i in self.basemodel_data[-self.number_of_seqences_generate_per_epoch:]:
            reward.append(i[1]-punish_accuracy)
        reward=np.array(reward)
        reward=reward.reshape(self.number_of_seqences_generate_per_epoch, 1)
        discounted_reward = np.zeros_like(reward, dtype=np.float32)
        # print(self.data)
        # print(reward)
        for t in range(len(reward)):
            tem = 0.
            gamma_exp = 0.
            for r in reward[t:]:
                tem += self.rnn_loss_function_alpha ** gamma_exp * r
                gamma_exp += 1
            discounted_reward[t] = tem
        discounted_reward = (discounted_reward - discounted_reward.mean()) / discounted_reward.std()
        # print(output)
        loss = - backend.log(output) * discounted_reward[:, None]
        # print(loss)
        # gvbhjk
        return loss


    def basemodelsearch(self):
        """
        This function is the main overflow of entire NAS. This functino do the following thigs.
            1) rnnmodel generates sample architectures.
            2) rnnmodel predicts accuracies for generated samples.
            3) Decodes the sequences to created model from architecture.
            4) create model and training model.
            5) storing model accuraces for training rnnmodel.
            6) creating data for training controller and training it.
        """
        for search_epoch in range(self.number_of_search_loops):
            sequences = self.basemodel_seq_generation(self.rnn_model, self.number_of_seqences_generate_per_epoch)
            pred_accuracies = self.basemodel_valacc_prediction_from_rnn(self.rnn_model, sequences)
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.id_to_config(sequence))
                history = self.basemodel_training(sequence)
                self.basemodel_seq_and_valacc(sequence, history, pred_accuracies[i])
                print('#####################################################################################')
            x__, y__, val_acc_target = self.rnnmodel_training_data(sequences)
            self.RNNModel_Train(self.rnn_model,
                                x__,
                                y__,
                                val_acc_target[-self.number_of_seqences_generate_per_epoch:],
                                self.rnnmodel_loss_function,
                                len(self.basemodel_data),
                                self.rnnmodel_training_epochs)
        with open(self.rnnmodel_data, 'wb') as f:
            pickle.dump(self.basemodel_data, f)
        return self.basemodel_data
