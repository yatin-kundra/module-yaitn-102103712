import argparse
import os

import pandas as pd
import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context


# df = pd.read_csv("https://raw.githubusercontent.com/NoOne03/topsis/master/data.csv")
# print(df)
# df = df.iloc[:, 1:]
# print("============")
# print(df)




def normalizing_data(df):
    squareSums = df.map(lambda x:x**2).sum(axis=0)     # applymap is used to apply a function on each element
    # axis = 0 means on each column
    # now normalizing each value of each column
    normalized_df = df / np.sqrt(squareSums)
    # print(normalized_df)
    return normalized_df


def multiply_weights(df, weights):
    df_vals = df.values
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr = weights_arr / sum(weights_arr)

    result = df_vals * weights_arr
    result_df = pd.DataFrame(result, columns=df.columns, index=df.index)

    return result_df


def deciding_impacts(df, impacts_):

    S_minus = np.zeros(df.shape[1])
    S_plus = np.zeros(df.shape[1])
    for i , sign in enumerate(impacts_):
        if sign == '+':
            S_plus[i] = max(df.iloc[:, i])
            S_minus[i] = min(df.iloc[:, i])
        elif sign == '-':
            S_plus[i] = min(df.iloc[:, i])
            S_minus[i] = max(df.iloc[:, i])
    return S_plus, S_minus


def finding_Rank(df, S_plus, S_minus):
    ecudi_posi = ((df - S_plus) ** 2).sum(axis=1).apply(np.sqrt)
    ecudi_neg = ((df - S_minus) ** 2).sum(axis=1).apply(np.sqrt)

    df['Topsis Score'] = ecudi_posi / (ecudi_posi + ecudi_neg)
    df['Rank'] = df['Topsis Score'].rank(ascending=False)

    return df

# if the user is an idiot
def checking_input(values):
    for value in values:
        if ' ' in value:
            return False
    return True


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()

    argumentParser.add_argument(
        'InputDataFile',
        type=str,
        help='give path to the input file'
    )
    argumentParser.add_argument(
        'Weights',
        type=str,
        help='give weights , they should be comma seprated'
    )
    argumentParser.add_argument(
        'Impacts_',
        type=str,
        help='give impacts i.e (+ or -) , they should be comma seprated'
    )
    argumentParser.add_argument(
        'ResultFileName',
        type=str,
        help='give name of the output file.'
    )
    arguments = argumentParser.parse_args()

    if len(vars(arguments)) != 4:
        argumentParser.error(f'number of parameters provided not 4, provided: {len(vars(arguments))}')

    weights = arguments.Weights.split(',')
    impacts_ = arguments.Impacts_.split(',')

    assert (checking_input(weights) == True and checking_input(
        impacts_) == True), "Weights and Impacts must be present in comma seprated format"

    try:
        df = pd.read_csv(arguments.InputDataFile)
    except FileNotFoundError:
        print(f'Error: File {arguments.InputDataFile} not found')
        exit()
    except Exception as e:
        print(f'An unexpected error occured: {e}')
        exit()

    if (df.shape[1] < 3):
        print('data does not have enough parameters for topsis')
        exit()

    assert (len(weights) == len(impacts_) and (
                len(impacts_) == df.shape[1] - 1)), "Length of impacts, weights and columns must be same"

    allowed_signs = ['+', '-']
    flag = all(value in allowed_signs for value in impacts_)

    assert (flag == True), "Impacts can only be negative or positive"

    df_products = df.iloc[:, 0]
    df = df.drop(columns=df.columns[0])
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    df = normalizing_data(df)
    df = multiply_weights(df, weights)
    Splus, Sminus = deciding_impacts(df, impacts_)

    df = finding_Rank(df, Splus, Sminus)

    output_path = os.path.join(output_dir, arguments.ResultFileName)
    df = pd.concat([df_products, df], axis=1)
    df.to_csv(output_path, index=False)

    print(f'Results saved to: {output_path}')

# print(normalizing_data(df))
# print("------------")
#
# df = multiply_weights(df,[2,2,0,0])
# print(df)