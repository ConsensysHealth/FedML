import numpy as np
import torch
import torch.nn as nn

from fedml_api.model.finance.vfl_models_standalone import DenseModel


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class VFLGuestModel(object):

    def __init__(self, local_model):
        super(VFLGuestModel, self).__init__()
        self.localModel = local_model
        self.is_debug = False

        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.parties_grad_component_list = []
        self.current_global_step = None
        self.X = None
        self.y = None

    def _fit(self, X):
        self.temp_K_Z = self.localModel.forward(X)
        return self.temp_K_Z

    def send_components(self):
        return self._fit(self.X)

    def send_predict(self,X):
        return self._fit(X)

    def set_batch(self, X, y, global_step):
        self.X = X
        self.y = y
        self.current_global_step = global_step

    def receive_activations(self, activations_server):
        self.K_U = activations_server

    def predict(self, U, component_list):
        for comp in component_list:
            U = U + comp
        return sigmoid(np.sum(U, axis=1))

    def receive_components(self, component_list):
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component)

    def fit(self):
        self._compute_common_gradient_and_loss(self.y)
        self.parties_grad_component_list = []

    def _compute_common_gradient_and_loss(self, y):
        U = self.K_U
        for grad_comp in self.parties_grad_component_list:
            U = U + grad_comp

        U = torch.tensor(U, requires_grad=True).float()
        y = torch.tensor(y)
        y = y.type_as(U)
        class_loss = self.classifier_criterion(U, y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=U)
        self.top_grads = grads[0].numpy()
        self.loss = class_loss.item()

    def _fit_back(self, X,back_grad):
        self.localModel.backward(X, back_grad)

    def receive_gradients(self,backprop):
        self._fit_back(self.X, backprop)

    def send_loss(self):
        return self.loss

    def send_gradients(self):
        return self.top_grads

class VFLHostModel(object):

    def __init__(self, local_model):
        super(VFLHostModel, self).__init__()
        self.localModel = local_model
        self.is_debug = False

        self.common_grad = None
        self.current_global_step = None
        self.X = None

    def set_batch(self, X, global_step):
        self.X = X
        self.current_global_step = global_step

    def _fit(self, X):
        self.A_Z = self.localModel.forward(X)
        return self.A_Z

    def _fit_back(self, X, back_grad):
        self.localModel.backward(X, back_grad)

    def send_components(self):
        return self._fit(self.X)

    def receive_gradients(self, back_grad):
        self._fit_back(self.X, back_grad)

    def send_predict(self, X):
        return self._fit(X)


class VFLServerModel(object):

    def __init__(self, part_a_out, part_b_out):
        self.dense_model_host = DenseModel(input_dim=part_b_out, output_dim=1, bias=False)
        self.dense_model_guest = DenseModel(input_dim=part_a_out, output_dim=1, bias=True)
        self.A_host = None
        self.A_guest = None

    def set_dense_model(self, dense_model,name):
        if name == "host":
            self.dense_model_host = dense_model
        elif name == "guest":
            self.dense_model_guest = dense_model

    def receive_activations(self,activations,name):
        if name == "host":
            self.A_host = activations
        elif name == "guest":
            self.A_guest = activations

        return self._fit(name)

    def receive_gradients(self, gradients,name):
        self.common_grad = gradients
        back_grad = self._fit_back(name)
        return back_grad

    def _fit(self,name):
        if name == "host":
            activations = self.A_host
            A_U = self.dense_model_host.forward(activations)
        elif name == "guest":
            activations = self.A_guest
            A_U = self.dense_model_guest.forward(activations)
        return A_U


    def _fit_back(self,name):
        if name == "host":
            back_grad = self.dense_model_host.backward(self.A_host, self.common_grad)
        elif name == "guest":
            back_grad = self.dense_model_guest.backward(self.A_guest, self.common_grad)
        return back_grad

