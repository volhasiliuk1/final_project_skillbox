import json
import pandas as pd
import dill

def main():
    with open('sber_auto_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    with open('data/example.json') as file:
        input_file = json.load(file)
        df = pd.DataFrame.from_dict([input_file])
        output = model['model'].predict(df)
        print(f'Посетитель {input_file["client_id"]}: {output[0]}')


if __name__=='__main__':
    main()
