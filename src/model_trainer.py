# local packages:
from data_loader import create_dataset, load_nyudv2


def train_model_to_data(model,dataset='user',epochs=1):
    if dataset == 'user':
        ds = create_dataset(path="../data/",
                        shape=(224, 224))
    elif dataset == 'nyudv2':
        ds = load_nyudv2()

    print(ds)
    model.fit(ds, epochs=epochs)
    return model
