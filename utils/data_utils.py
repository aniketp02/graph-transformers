

PatternNet_label_classes = {
    'Airfield': 0,
    'Anchorage': 1,
    'Beach': 2,
    'DenseResidential': 3,
    'Farm': 4,
    'Flyover': 5,
    'Forest': 6,
    'GameSpace': 7, 
    'ParkingSpace': 8, 
    'River': 9, 
    'SparseResidential': 10,
    'StorageCisterns': 11,
}


EuroSAT_label_classes = {
    'AnnualCrop': 0,
    'Forest': 1,
    'HerbaceousVegetation': 2,
    'Highway': 3,
    'Industrial': 4,
    'Pasture': 5,
    'PermanentCrop': 6,
    'Residential': 7, 
    'River': 8, 
    'SeaLake': 9
    }



def get_label_patternet(label):
    labels = [0] * 12

    index = PatternNet_label_classes[label]
    labels[index] = 1

    return labels

def get_label_eurosat(label):
    labels = [0] * 10

    index = EuroSAT_label_classes[label]
    labels[index] = 1

    return labels


# print(get_label('Farm'))
