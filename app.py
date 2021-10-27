from dataAcquisition.dataAcquisition import DataAcquisition


if __name__ == '__main__':
    data_acquirer = DataAcquisition()

    train_X, test_X, train_y, test_y = data_acquirer.acquire_data()