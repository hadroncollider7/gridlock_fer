import os

os.system('cls')

# The AffectNet key
key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt', 8:'Ambiguous', 9:'Not a Face'}
mylist = [0,1,2,3,4,5,6,7,8,9]
print(f'key type: {type(key)}')
print(f'list type: {type(mylist)}')
print(mylist)

for i in range(len(mylist)):
    mylist[i] = key[mylist[i]]

print(f'list type: {type(mylist)}')
print(mylist)
