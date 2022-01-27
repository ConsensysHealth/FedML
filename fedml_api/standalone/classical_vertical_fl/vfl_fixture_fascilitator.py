import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from fedml_api.utils.function_managment import compute_correct_prediction
from fedml_api.standalone.classical_vertical_fl.vfl_fascilitator import VerticalMultiplePartyLogisticRegressionFederatedLearning
import numpy
logger = logging.getLogger(__name__)
log_output = logging.getLogger('training_output')
fileHandler = logging.FileHandler('training_output.log', mode='w')
log_output.setLevel(logging.INFO)
log_output.addHandler(fileHandler)

class FederatedLearningFixture(object):

    def __init__(self, federated_learning: VerticalMultiplePartyLogisticRegressionFederatedLearning):
        self.federated_learning = federated_learning
        logger.info("Step 7: initialize FederatedLearningFixture")

    def fit(self, train_data: dict, test_data: dict, epochs=50, batch_size=-1):
        """
        Performs the fitting of the federated learning environment
        """
        logger.info("Step 9: start to fit, split data and refer back to VerticalMultiplePartyLogisticRegressionFederatedLearning")
        main_party_id = self.federated_learning.get_main_party_id()

        # receives the train and the test data from guest
        y_train = train_data[main_party_id]["Y"]
        y_test = test_data[main_party_id]["Y"]

        N = y_train.shape[0]
        residual = N % batch_size

        # received the batches
        if residual == 0:
            n_batches = N // batch_size
        else:
            n_batches = N // batch_size + 1

        log_output.info("number of samples: {}".format(N))
        log_output.info("batch size: {}".format(batch_size))
        log_output.info("number of batches: {}".format(n_batches))

        global_step = -1
        recording_period = 30
        recording_step = -1
        threshold = 0.5

        loss_list = []
        for ep in range(epochs):
            for batch_idx in range(n_batches):
                global_step += 1
                # Splits the data!
                # prepare batch data for guest, which has y.
                Y_batch = y_train[batch_idx * batch_size: batch_idx * batch_size + batch_size]

                # prepare batch data for all Hosts parties
                party_X_train_batch_dict = dict()
                for party_id, party_X in train_data["party_list"].items():
                    party_X_train_batch_dict[party_id] = party_X[
                                                         batch_idx * batch_size: batch_idx * batch_size + batch_size]

                loss = self.federated_learning.fit(Y_batch,party_X_train_batch_dict, global_step)
                loss_list.append(loss)

                if (global_step + 1) % recording_period == 0:
                    recording_step += 1
                    ave_loss = np.mean(loss_list)
                    loss_list = list()
                    party_X_test_dict = dict()
                    for party_id, party_X in test_data["party_list"].items():
                        party_X_test_dict[party_id] = party_X
                    y_prob_preds = self.federated_learning.predict(party_X_test_dict)
                    y_hat_lbls, statistics = compute_correct_prediction(y_targets=y_test,
                                                                        y_prob_preds=y_prob_preds,
                                                                        threshold=threshold)
                    acc = accuracy_score(y_test, y_hat_lbls)
                    auc = 0#roc_auc_score(y_test, y_prob_preds)
                    log_output.info("--- epoch: {0}, batch: {1}, loss: {2}, acc: {3}, auc: {4}"
                          .format(ep, batch_idx, ave_loss, acc, auc))
                    log_output.info("--- {}".format(precision_recall_fscore_support(y_test, y_hat_lbls, average="macro", warn_for=tuple())))
