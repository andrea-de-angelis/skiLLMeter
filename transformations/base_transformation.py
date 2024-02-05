class BaseTransformation:
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.transformed_data = None
        
    def transform(self):
        '''
        Transform the benchmark in a Pandas DataFrame.

        This method should be overridden by specific transformation classes.
        '''
        raise NotImplementedError("transform method must be implemented in the derived class.")
    
    
    def predict(self, model, tokenizer, labels=None):
        '''
        Do the inference on the transformed data.

        This method should be overridden by specific transformation classes.
        '''
        raise NotImplementedError("inference method must be implemented in the derived class.")
        

    def get_transformed_data(self):
        if self.transformed_data is None:
            raise ValueError("Dataset has not been transformed. Call transform first.")
        return self.transformed_data