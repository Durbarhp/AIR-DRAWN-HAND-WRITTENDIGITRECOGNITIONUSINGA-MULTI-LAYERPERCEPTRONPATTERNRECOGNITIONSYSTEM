function C = digit_classify(testdata)
     testdata = py.numpy.array(testdata);
     C = pyrunfile("main.py", "C", testdata = testdata);
     C = double(C);
 end