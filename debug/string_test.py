mystring = "John_Smith_1.jpg"
thename = mystring.split('_')[-1].split('.')[0]

print(thename)