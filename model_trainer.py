# local packages:
from data_loader import create_dataset


def train_model_to_data(model):
    ds = create_dataset(path="./data/",
                        shape=(224, 224))
    model.fit(ds, epochs=10)
    return model
