import pandas as pd
from .base_transformation import BaseTransformation
import torch 

class MMLUTransformation(BaseTransformation):
    def __init__(self, benchmark):
        super().__init__(benchmark)
        self.answer_map = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def transform(self):
        questions = [line for line in self.benchmark['question']]
        choices1 = [line for line in self.benchmark['choice1']]
        choices2 = [line for line in self.benchmark['choice2']]
        choices3 = [line for line in self.benchmark['choice3']]
        choices4 = [line for line in self.benchmark['choice4']]

        answers = [line for line in self.benchmark['answer']]

        transformed_df = pd.DataFrame({
            'question': questions,
            'choice1': choices1, 'choice2': choices2,
            'choice3': choices3, 'choice4': choices4,
            'label': answers
        })
        self.transformed_data = transformed_df
        
        
    def predict(self, model, tokenizer, labels=None):
        for index, row in self.transformed_data.iterrows():
            if labels:
                input1 = f"{labels} | {row.question} {row.choice1}"
                input2 = f"{labels} | {row.question} {row.choice2}"
                input3 = f"{labels} | {row.question} {row.choice3}"
                input4 = f"{labels} | {row.question} {row.choice4}"
            else:
                input1 = f"{row.question} {row.choice1}"
                input2 = f"{row.question} {row.choice2}"
                input3 = f"{row.question} {row.choice3}"
                input4 = f"{row.question} {row.choice4}"
            
            enc_input1 = tokenizer.encode(input1, return_tensors="pt").to('cuda')
            enc_input2 = tokenizer.encode(input2, return_tensors="pt").to('cuda')
            enc_input3 = tokenizer.encode(input3, return_tensors="pt").to('cuda')
            enc_input4 = tokenizer.encode(input4, return_tensors="pt").to('cuda')

            with torch.no_grad():
                res1 = model(enc_input1, labels=enc_input1)
                res2 = model(enc_input2, labels=enc_input2)
                res3 = model(enc_input3, labels=enc_input3)
                res4 = model(enc_input4, labels=enc_input4)


            loss1 = res1.loss.item()
            loss2 = res2.loss.item()
            loss3 = res3.loss.item()
            loss4 = res4.loss.item()

            losses = [loss1, loss2, loss3, loss4]

            self.transformed_data.at[index, 'loss_choice1'] = loss1
            self.transformed_data.at[index, 'loss_choice2'] = loss2
            self.transformed_data.at[index, 'loss_choice3'] = loss3
            self.transformed_data.at[index, 'loss_choice4'] = loss4
            
            self.transformed_data.at[index, 'pred'] = self.answer_map[losses.index(min(losses))]
        return self.transformed_data