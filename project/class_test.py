import sklearn
from sklearn.model_selection import train_test_split

class easy:

    def __init__(self):
        pass

    def split_data(self, df):
        self.df = df
        """
        function to split data into train, test, validate data

        The dataframe is split into 15% test, then 85% train. 
        The train data set is then further split into 20% validation
        and 80% train dataset. 

        So the final result is 15% of original data = test
        about 15% of original data is validation,
        and 70% of original data is train. 
    
        Split dataframe into 70% train, 15% val, 15% test
        """
        self.train_1, self.test = train_test_split(self.df, test_size=0.15, train_size=0.85, random_state=42)

        self.train, self.val = train_test_split(self.train_1, test_size=0.15, train_size=0.85, random_state=42)

        return self.train, self.test, self.val
