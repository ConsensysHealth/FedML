import os
import sys
import logging
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.data_preprocessing.lending_club_loan.lending_club_dataset import loan_load_two_party_data
from fedml_api.standalone.classical_vertical_fl.vfl_fixture_fascilitator import fed_learning_fixture_fit
from fedml_api.standalone.classical_vertical_fl.party_models_facilitator import VFLGuestModel, VFLHostModel, VFLFacilitator
from fedml_api.model.finance.vfl_models_standalone import LocalModel, DenseModel
from fedml_api.standalone.classical_vertical_fl.vfl_facilitator import VerticalMultiplePartyLogisticRegressionFederatedLearning


def run_experiment(train_data: list, test_data: list, batch_size: int, learning_rate: float, epoch:int):

    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    print("################################ Wire Federated Models ############################")

    # create local models for both Client 1 and Client 2
    party_a_local_model = LocalModel(input_dim=Xa_train.shape[1], output_dim=10, learning_rate=learning_rate)
    party_b_local_model = LocalModel(input_dim=Xb_train.shape[1], output_dim=20, learning_rate=learning_rate)

    # Instantiates the Guest, Host and Facilitator
    partyA = VFLGuestModel(local_model=None)
    client1 = VFLHostModel(local_model=party_a_local_model)
    client2 = VFLHostModel(local_model=party_b_local_model)
    facilitator_vfl = VFLFacilitator(party_a_local_model.get_output_dim(), party_b_local_model.get_output_dim())



    # Sets the dense model for the host at the facilitator
    facilitator_dense_model_c2 = DenseModel(party_b_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=False)
    facilitator_dense_model_c1 = DenseModel(party_a_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=False)
    facilitator_vfl.set_dense_model(facilitator_dense_model_c1, "Client 1")
    facilitator_vfl.set_dense_model(facilitator_dense_model_c2,"Client 2")

    federatedLearning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA,facilitator_vfl)
    federatedLearning.add_party(id="Client 1", party_model=client1)
    federatedLearning.add_party(id="Client 2", party_model=client2)

    print("################################ Train Federated Models ############################")

    train_data = {federatedLearning.get_main_party_id(): {"Y": y_train},
                  "party_list": {"Client 1": Xa_train, "Client 2": Xb_train}}
    test_data = {federatedLearning.get_main_party_id(): {"Y": y_test},
                 "party_list": {"Client 1": Xa_test, "Client 2": Xb_test}}

    fed_learning_fixture_fit(federatedLearning,train_data=train_data, test_data=test_data, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    logging.basicConfig(filename='fasciliator_info.log', level=logging.INFO, format = '%(asctime)s:%(name)s:%(message)s')
    logger = logging.getLogger("main")
    logger.info('Master runs')

    print("################################ Prepare Data ############################")
    data_dir = "../../../data/lending_club_loan/"

    train, test = loan_load_two_party_data(data_dir)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    batch_size = 256
    epoch = 10
    lr = 0.001

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]
    run_experiment(train_data=train, test_data=test, batch_size=batch_size, learning_rate=lr, epoch=epoch)