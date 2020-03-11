
import sklearn
from sklearn.model_selection import train_test_split

class Easy:
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

    def target_features(self, train, val, test, target):
        self.train = train
        self.val = val
        self.test = test
        '''
        Target = target from the dataframe. 

        '''

        self.target = target
        features = self.train.columns.drop(self.target)

        self.x_train = self.train[features]
        self.y_train = self.train[target]
        self.x_val = self.val[features]
        self.y_val = self.val[target]
        self.x_test = self.test[features]
        self.y_test = self.test[target]

        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

        # function to perform data cleaning on all datasets. 

    def wrangle(self, x):
        self.x = x
        x = x.copy()
        
        x.columns = [col.lower()         # make column names lowercase
                    .strip('_')         # strip leading/trailing underscores
                    .replace('_', ' ')  # replace remaining punctuation with spaces
                    .replace('-', ' ') 
                    .replace('/', ' ')
                    for col in x.columns]
        
        # dropping columns with leakage
        x = x.drop(columns = ['country', 'year', 'income composition of resources',
                            'percentage expenditure', 'infant deaths', 
                            'under five deaths', 'adult mortality'])
        
        # filling nan values with 0 because I can't do imputation and the missing
        # values can be assumed to be 0.
        
        x = x.fillna(value=0)
        
        return self.x
