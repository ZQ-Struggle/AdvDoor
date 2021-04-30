import os

from poison_detection import ActivationDefence


class Activations:
    def __init__(self, model, para):
        # self.poison = model.get_train_poison()
        # defence = ActivationDefence(model.classifier, data.x_train, data.y_train,
        #                             data_path=os.path.join(data.data_path, 'train'), batch_size=data.batch_size)
        # self.activations = self._get_activations(defence)
        self.activations = model
        self.para = para

    def _get_activations(self, defences):
        nb_layers = len(defences.classifier.layer_names)
        activations_by_layers = []
        '''
        for i in range(nb_layers):
            activations_by_layers.append(
                defences.classifier.get_activations(defences.x_train, layer=i, data_path=defences.data_path,
                                                    batch_size=defences.batch_size))
        '''

        activations_by_layers.append(
            defences.classifier.get_activations(defences.x_train, layer=nb_layers - 2, data_path=defences.data_path,
                                                batch_size=defences.batch_size))
        nb_layers = 1
        activations = [[] for i in range(len(defences.x_train))]
        for i in range(nb_layers):
            for j in range(len(defences.x_train)):
                activations[j].append(activations_by_layers[i][j])
        # print(len(activations[0]))
        return activations

    def restore_data(self, data):
        data = data(self.para)
        data.load_data();
        data.restore_train_backdoor(self.poison)
        # self.shuffle_activations(data.shuffled_indices)
        data.gen_test_backdoor()
        return data

    def shuffle_activations(self, shuffled_index):
        self.activations = [self.activations[i] for i in shuffled_index]
