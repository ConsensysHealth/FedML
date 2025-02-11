import numpy
import numpy as np
import torch
import torch.nn as nn
import logging
from fedml_api.model.finance.vfl_models_standalone import DenseModel, LocalModel
from fedml_api.utils.function_managment import sigmoid

logger = logging.getLogger(__name__)

class VFLGuestModel(object):
    """
    The Guest allows to add additional layers after receiving the final activations from the facilitator.
    It calculates the loss from the activations received by the facilitator and performs backpropagation
    """
    def __init__(self, local_model):
        super(VFLGuestModel, self).__init__()
        self.localModel = local_model
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.current_global_step = None
        self.y = None

    def _fit(self, inter_act: numpy.ndarray) -> numpy.ndarray:
        """
        Sets activations and in case there are additional layers on the guest side performs forward propagation

        Args:
            inter_act: the intermediate activatiation received from the facilitator

        Returns: the final activations

        """
        logger.debug("Step 5:receives the activation")
        final_act = inter_act

        if self.localModel != None:
            logger.debug("Step 5a: performs forward propagation")
            final_act = self.localModel.forward(inter_act)

        return final_act

    def receive_concatination(self, concat_act: numpy.ndarray):
        """
        Receives the activations from the facilitator and calculated the loss
        """
        self.final_act = self._fit(concat_act)
        self._compute_loss(self.y)

    def set_batch(self, y:numpy.ndarray, global_step:int):
        """
        Sets the batches for the guest
        """
        self.y = y
        self.current_global_step = global_step

    def predict(self,final_act: numpy.ndarray):
        """
        Predicts the output based on the received activations from the facilitator
        """
        logger.debug("Step 5:receives the activation")
        self.final_act = self._fit(final_act)

        return sigmoid(np.sum(final_act, axis=1))

    def _compute_loss(self, y: numpy.ndarray):
        logger.debug("Step 6:computes loss ")
        U = torch.tensor(self.final_act, requires_grad=True).float()
        y = torch.tensor(y)
        y = y.type_as(U)
        class_loss = self.classifier_criterion(U, y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=U)
        self.top_grads = grads[0].numpy()
        if self.localModel != None:
            self.top_grads = self._fit_back()
        self.loss = class_loss.item()

    def _fit_back(self):
        """
        Performs backward propagations in the case a model is present
        """
        self.localModel.backward(self.final_act, self.top_grads)

    def send_loss(self) -> float:
        """
        Sends the loss back to the facilitator
        """
        logger.debug("Step 7:sends loss")
        return self.loss

    def send_gradients(self) -> numpy.ndarray:
        """
        Sends the gradient back to the facilitator
        """
        logger.debug("Step 8:send gradients")
        return self.top_grads

class VFLHostModel(object):
    """
    The Host model applies the first layers to the data hold locally by each of the clients
    """
    def __init__(self, local_model: LocalModel):
        super(VFLHostModel, self).__init__()
        self.localModel = local_model
        self.common_grad = None
        self.current_global_step = None
        self.X = None

    def set_batch(self, X: numpy.ndarray, global_step: int):
        """
        Sets the batch for the hosts
        """
        logger.debug("Step 1: Set batch for Host")
        self.X = X
        self.current_global_step = global_step

    def _fit(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Performs forward propagation on the local model and returns actiavtions
        """
        logger.debug("Successfully entered _fit")
        self.A_C2F = self.localModel.forward(X)
        return self.A_C2F

    def _fit_back(self, X: numpy.ndarray, back_grad:numpy.ndarray):
        """
        Performs backward propagation
        """
        logger.debug("backprop succesfull")
        self.localModel.backward(X, back_grad)

    def send_components(self) -> numpy.ndarray:
        logger.debug("Step 2: performs forward on the local model")
        return self._fit(self.X)

    def receive_gradients(self, back_grad:numpy.ndarray):
        """
        Receives the gradients and initializes backward prop
        """
        logger.debug("Step 11: performs backprop on host")
        self._fit_back(self.X, back_grad)

    def send_predict(self, X):
        logger.debug("Step 12: Fits prediction")
        return self._fit(X)

class VFLFacilitator(object):
    """
    The Facilitator concatinates the results from the host model
    """

    def __init__(self, part_a_out: int, part_b_out: int):
        self.dense_model_c1 = DenseModel(input_dim=part_b_out, output_dim=1, bias=False)
        self.dense_model_c2 = DenseModel(input_dim=part_a_out, output_dim=1, bias=True)
        # activations of the local model
        self.A_C1 = None
        self.A_C2 = None
        # activations after the dense model
        self.A_C1_F = None
        self.A_C2_F = None

    def set_dense_model(self, dense_model: DenseModel,name: str):
        """
        Sets the dense model to Client 1 and 2
        """
        if name == "Client 1":
            self.dense_model_c1 = dense_model
        elif name == "Client 2":
            self.dense_model_c2 = dense_model

    def receive_activations(self,activations: numpy.ndarray, name: str) -> numpy.ndarray:
        """
        Receives the activations from the Host and uses the activations to fits it on the dense model
        """
        if name == "Client 1":
            self.A_C1 = activations
        elif name == "Client 2":
            self.A_C2 = activations
        return self._fit(name)

    def _fit(self,name: str) -> numpy.ndarray:
        """
        Performs forward propagation on the Activations received by the Client
        """
        if name == "Client 1":
            self.A_C1_F = self.dense_model_c1.forward(self.A_C1)
            logger.debug("Step 3a: performs forward on the dense model for Client 1")
            A_U = self.A_C1_F
        elif name == "Client 2":
            self.A_C2_F = self.dense_model_c2.forward(self.A_C2)
            logger.debug("Step 3b: performs forward on the dense model for Client 2")
            A_U = self.A_C2_F
        return A_U

    def _compute_common_gradient(self)-> numpy.ndarray:
        """
        Adds the activations of both hosts after applying the dense model
        """
        logger.debug("Step 4: adds the activation of both parties")
        A_F2H = self.A_C1_F + self.A_C2_F
        return A_F2H

    def receive_concatination(self) -> numpy.ndarray:
        """
        Receives the activations of the concatenated activations
        """
        return self._compute_common_gradient()

    def receive_gradients(self, gradients:numpy.ndarray) -> numpy.ndarray:
        """
        Facilitator receives gradients from Guest
        """
        logger.debug("Step 9: receives gradients")
        self.common_grad = gradients

    def perform_back(self,name:str):
        """
        Performs backward propagation for the specified client
        """
        back_grad = self._fit_back(name)
        return back_grad

    def _fit_back(self,name:str)-> numpy.ndarray:
        """
        Performs the backprop for the corresponding client
        """
        if name == "Client 1":
            logger.debug("Step 10a: performs backprop Client 1")
            back_grad = self.dense_model_c1.backward(self.A_C1, self.common_grad)
        elif name == "Client 2":
            logger.debug("Step 10b: performs backprop Client 2")
            back_grad = self.dense_model_c2.backward(self.A_C2, self.common_grad)
        return back_grad