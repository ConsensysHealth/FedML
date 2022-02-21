import logging
from fedml_api.standalone.classical_vertical_fl.party_models_facilitator import VFLHostModel, VFLGuestModel, VFLFacilitator
import numpy

logger = logging.getLogger(__name__)

class VerticalMultiplePartyLogisticRegressionFederatedLearning(object):

    def __init__(self, party_A: VFLGuestModel, fascilator: VFLFacilitator, main_party_id="_main"):
        super(VerticalMultiplePartyLogisticRegressionFederatedLearning, self).__init__()
        self.facilitator = fascilator
        self.main_party_id = main_party_id
        # party A is the parity with labels
        self.party_a = party_A
        # the party dictionary stores other parties that have no labels
        self.party_dict = dict()

    def get_main_party_id(self) -> str:
        """
        Retreives the main party ID
        """
        return self.main_party_id

    def add_party(self, *, id: str, party_model: VFLHostModel):
        """
        Adds the ID and the Host model to a dictionary
        """
        self.party_dict[id] = party_model

    def fit(self, y: numpy.ndarray, party_X_dict: dict, global_step: int) -> float:
        """
        Performs the forward for each batch on the host model, forwards it to the facilitator, and the loss is computed
        at the guest. Vice versa holds for the backpropagation
        """
        logger.info("==> start fit")

        logger.info("==> Set Batch for all hosts")
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, global_step)

        logger.info("==> Facilitator receives intermediate computing results from hosts")
        for party_id, party in self.party_dict.items():
            activations = party.send_components()
            self.facilitator.receive_activations(activations, party_id)

        logger.info("==> Facilitator concatenates results")
        concatenation = self.facilitator.receive_concatination()

        logger.info("==> Set Batch for guest")
        self.party_a.set_batch(y, global_step)

        logger.info("==> Guest computes loss")
        self.party_a.receive_concatination(concatenation)
        loss = self.party_a.send_loss()
        grad_result = self.party_a.send_gradients()
        self.facilitator.receive_gradients(grad_result)

        logger.info("==> Hosts receive common grad from facilitator and perform training")
        for party_id, party in self.party_dict.items():
            back_prop = self.facilitator.perform_back(party_id)
            party.receive_gradients(back_prop)

        return loss

    def predict(self,party_X_dict: dict) -> numpy.ndarray:
        """
        Performs the final prediction
        """
        for id, party_X in party_X_dict.items():
            activations = self.party_dict[id].send_predict(party_X)
            self.facilitator.receive_activations(activations, id)
        self.facilitator.receive_concatination()
        return self.party_a.predict(activations)