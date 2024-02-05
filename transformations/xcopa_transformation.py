import pandas as pd
from .base_transformation import BaseTransformation


class XCOPATransformation(BaseTransformation):
    def transform(self):
        premises = [line for split in ["test", "validation"] for line in self.benchmark[split]['premise']]
        choices1 = [line for split in ["test", "validation"] for line in self.benchmark[split]['choice1']]
        choices2 = [line for split in ["test", "validation"] for line in self.benchmark[split]['choice2']]
        labels = [line for split in ["test", "validation"] for line in self.benchmark[split]['label']]

        transformed_df = pd.DataFrame({
            'premise': premises, 
            'choice1': choices1, 'choice2': choices2, 
            'label': labels
        })
        self.transformed_data = transformed_df
        
        
    def predict(self, model, tokenizer, labels=None):
        for index, row in self.transformed_data.iterrows():
            if labels:
                input1 = f"{labels} | {row.premise} {row.choice1}"
                input2 = f"{labels} | {row.premise} {row.choice2}"
            else:
                input1 = f"{row.premise} {row.choice1}"
                input2 = f"{row.premise} {row.choice2}"
                
            enc_input1 = tokenizer.encode(input1, return_tensors="pt").to('cuda')
            enc_input2 = tokenizer.encode(input2, return_tensors="pt").to('cuda')
            
            res1 = model(enc_input1, labels=enc_input1)
            res2 = model(enc_input2, labels=enc_input2)
            
            loss1 = res1.loss.item()
            loss2 = res2.loss.item()
            
            self.transformed_data.at[index, 'loss_choice1'] = loss1
            self.transformed_data.at[index, 'loss_choice2'] = loss2
            self.transformed_data.at[index, 'pred'] = 0 if loss1 < loss2 else 1
            
        return self.transformed_data
