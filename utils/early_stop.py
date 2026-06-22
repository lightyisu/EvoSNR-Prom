class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_mcc = -float('inf')
        self.early_stop = False

    def __call__(self, mcc):
        if mcc > self.best_mcc:
            self.best_mcc = mcc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True