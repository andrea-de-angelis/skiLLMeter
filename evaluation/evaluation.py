def compute_accuracy(dataset):
    return round( len(dataset[dataset['pred'] == dataset['label']]) / len(dataset), 3 )