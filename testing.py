from class_magnesium_not_drop_na import *

def test_file(file_, fold):
    print('WITH GROUPS, RANDOMFOREST') 
    m = Magnesium(file_, fold = fold, with_groups = True)
    print('Portion of sites = ', round(np.sum(m.y==1)/m.y.shape[0], 2))
    m.compute(n_splits = 15)
    print('\nWITHOUT GROUPS, RANDOMFOREST')
    m = Magnesium(file_, fold = fold, with_groups = False)
    m.compute(n_splits = 15)

def test(files, fold):
    for file_ in files:
        print(file_)
        test_file(file_, fold)