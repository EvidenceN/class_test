import sklearn
from sklearn.model_selection import train_test_split

# function that does train, test, val split

def split_data(df):
    """
    Split dataframe into 70% train, 15% val, 15% test
    """
    train, test = train_test_split(df, test_size=0.15, train_size=0.85, random_state=42)

    train, val = train_test_split(train, test_size=0.15, train_size=0.85, random_state=42)

    return train, test, val

def target_features(target):
    '''
    Target = target from the dataframe. 

    '''

    target = target
    features = train.columns.drop(target)

    x_train = train[features]
    y_train = train[target]
    x_val = val[features]
    y_val = val[target]
    x_test = test[features]
    y_test = test[target]

    return x_train, y_train, x_val, y_val, x_test, y_test

# function to perform data cleaning on all datasets. 

def wrangle(x):
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
    
    return x