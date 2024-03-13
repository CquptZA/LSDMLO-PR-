
# Requirements 
numpy 1.21.5

pytorch 1.11.0

pandas 1.3.5

scikit-learn 1.0.2

scikit-multilearn 0.2.0 notice: skmultilearn is a multi label code package that we have fixed its bugs. It is recommended to download and install it from this code library into the Conda environment



# Usage
"Corel5k.arff","yahoo-Business1","yahoo-Arts1.arff" are demo dataset for giving.

run Main.ipynb

parameter:

c_dx: classifier method index (give MLkNN for example)

OptmParameter is used to Get_W
paracombine: is a list, with each element being a tuple. The first element is 'alpha' and the second element is 'beta' in a tuple.

sp:sampling rate

Result printing format(Output the average result and standard deviation of 5 *2 folds, in the following order:
    np.mean(Macro_F),
    np.mean(Micro_F),
    np.mean(Macro_AUC),
    np.mean(Ranking_loss),
    np.mean(Hamming_loss),
    np.mean(One_error),
    np.std(Macro_F),
    np.std(Micro_F),
    np.std(Macro_AUC),
    np.std(Ranking_loss),
    np.std(Hamming_loss),
    np.std(One_error)

# Other

If there are any issues, please contact email zacqupt@gamil.com
