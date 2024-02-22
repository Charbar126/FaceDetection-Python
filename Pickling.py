import pickle
class PickleHelper:
    def save_to(self, location, obj):
        with open(location, "wb") as f:
            pickle.dump(obj, f)

    def load_back(self, location):
        with open(location, 'rb') as f:
            return pickle.load(f)


### demo on how to pickle and unpickle
if __name__ == '__main__':
    class User:
        def __init__(self, name, email, contact):
            self.name = name
            self.email = email
            self.contact = contact

    ##
    User1 = User('Kumar', 'rajesh.kumar@bucknell.edu', '5705771234')
    print('Before pickling: ')
    print('User1.name: ', User1.name)
    print('User1.email:', User1.email)
    print('User1.contact:', User1.contact)
    Pickler = PickleHelper()
    Pickler.save_to('User1.pkl', User1)
    User1Loaded = Pickler.load_back('User1.pkl')
    print('After pickling and loading: ')
    print('User1Loaded.name: ', User1Loaded.name)
    print('User1Loaded.email:', User1Loaded.email)
    print('User1Loaded.contact:', User1Loaded.contact)