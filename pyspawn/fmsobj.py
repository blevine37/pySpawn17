# A class from which all fms classes should be derived.
# Includes methods for output of classes to json format.
# The ability to read/dump data from/to json is essential to the
# restartability that we intend.
# nested python dictionaries serve as an intermediate between json
# and the native python class
import types
import numpy as np
import json

class fmsobj(object):
    # Convert fmsobj structure to python dict structure
    def to_dict(self):
        tempdict=(self.__dict__).copy()
        for key in tempdict:
            # numpy objects
            if type(tempdict[key]).__module__ == np.__name__ :
                tempdict[key] = tempdict[key].tolist()
                for i in range(len(tempdict[key])):
                    # complex elements of 1d arrays are encoded here
                    if isinstance(tempdict[key][i], complex):
                        tempdict[key][i] = "^complex(" + str(tempdict[key][i].real) + "," + str(tempdict[key][i].imag) + ")"
                    else:
                        # and complex 2d arrays here
                        if isinstance(tempdict[key][i], types.ListType):
                            for j in range(len(tempdict[key][i])):
                                if isinstance(tempdict[key][i][j], complex):
                                    tempdict[key][i][j] = "^complex(" + str(tempdict[key][i][j].real) + "," + str(tempdict[key][i][j].imag) + ")"
            # fms objects here
            if (type(tempdict[key]).__module__)[0:7] == __name__[0:7] :
                fmsobjlabel = type(tempdict[key]).__module__
                tempdict[key] = tempdict[key].to_dict()
                (tempdict[key])["fmsobjlabel"] = fmsobjlabel
            # dictionaries here
            if isinstance(tempdict[key],types.DictType) :
                tempdict2 = (tempdict[key]).copy()
                tempdict[key] = tempdict2
                for key2 in tempdict2:
                    if (type(tempdict2[key2]).__module__)[0:7] == __name__[0:7] :
                        fmsobjlabel = type(tempdict2[key2]).__module__
                        tempdict2[key2] = tempdict2[key2].to_dict()
                        (tempdict2[key2])["fmsobjlabel"] = fmsobjlabel
                
        return tempdict

    # Convert dict structure to fmsobj structure
    def from_dict(self,**tempdict):
        for key in tempdict:
            if isinstance(tempdict[key],types.UnicodeType) :
                tempdict[key] = str(tempdict[key])
            if isinstance(tempdict[key],types.ListType) :
                if isinstance((tempdict[key])[0],types.FloatType) :
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key],dtype=np.float64)
                if isinstance((tempdict[key])[0],types.StringTypes) :
                    if (tempdict[key])[0][0] == "^":
                        for i in range(len(tempdict[key])):
                            tempdict[key][i] = eval(tempdict[key][i])
                        tempdict[key] = np.asarray(tempdict[key],dtype=np.complex128)
#new
                    if isinstance((tempdict[key])[0],types.UnicodeType) :
                        for i in range(len(tempdict[key])):
                            tempdict[key][i]= str(tempdict[key][i])
#end new
                            
                            
                else:
                    if isinstance((tempdict[key])[0],types.ListType):
                        if isinstance((tempdict[key])[0][0],types.FloatType) :
                            # convert 2d float lists to np arrays
                            tempdict[key] = np.asarray(tempdict[key],dtype=np.float64)
                        if isinstance((tempdict[key])[0][0],types.StringTypes) :
                            if (tempdict[key])[0][0][0] == "^":
                                for i in range(len(tempdict[key])):
                                    for j in range(len(tempdict[key][i])):
                                        tempdict[key][i][j] = eval(tempdict[key][i][j][1:])
                                tempdict[key] = np.asarray(tempdict[key],dtype=np.complex128)
        self.__dict__.update(tempdict)

    # Write fmsobj structure to disk in json format
    def write_to_file(self,outfilename):
        tempdict = self.to_dict()
        with open(outfilename,'w') as outputfile:
            json.dump(tempdict,outputfile,sort_keys=True, indent=4, separators=(',', ': '))

    # Read fmsobj structure from json file
    def read_from_file(self,infilename):
        with open(infilename,'r') as inputfile:
            tempdict = json.load(inputfile)

        # replace unicode keys (as read by json) with python strings
        # unicode causes problems other places in the code
        for key in tempdict:
            if isinstance(tempdict[key],types.DictType):
                for key2 in tempdict[key]:
                    if isinstance(key2, types.UnicodeType):
                        tempdict[key][str(key2)] = tempdict[key].pop(key2)
            
        self.from_dict(**tempdict)

    def set_parameters(self,params):
        print "### Setting " + self.__class__.__name__ + " parameters"
        for key in params:
            print key + " = " + str(params[key]) 
            method = "set_" + key
            if hasattr(self,method):
                exec("self.set_" + key + "(params[key])")
            else:
                print "### Parameter " + key + " not found in " + self.__class__.__name__ + ", exiting"
                quit()
                
