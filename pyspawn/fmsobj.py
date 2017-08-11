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
            if type(tempdict[key]).__module__ == np.__name__ :
                tempdict[key] = tempdict[key].tolist()
            if (type(tempdict[key]).__module__)[0:7] == __name__[0:7] :
                print "I'm here"
                fmsobjlabel = type(tempdict[key]).__module__
                tempdict[key] = tempdict[key].to_dict()
                (tempdict[key])["fmsobjlabel"] = fmsobjlabel
            if isinstance(tempdict[key],types.DictType) :
                tempdict2 = (tempdict[key]).copy()
                tempdict[key] = tempdict2
                for key2 in tempdict2:
                    if (type(tempdict2[key2]).__module__)[0:7] == __name__[0:7] :
                        print "I'm there"
                        fmsobjlabel = type(tempdict2[key2]).__module__
                        tempdict2[key2] = tempdict2[key2].to_dict()
                        (tempdict2[key2])["fmsobjlabel"] = fmsobjlabel
                
        return tempdict

    # Convert dict structure to fmsobj structure
    def from_dict(self,**tempdict):
        for key in tempdict:
            print key
            if isinstance(tempdict[key],types.ListType) :
                if isinstance((tempdict[key])[0],types.FloatType) :
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key])
                else:
                    if isinstance((tempdict[key])[0],types.ListType):
                        if isinstance((tempdict[key])[0][0],types.FloatType) :
                            # convert 2d float lists to np arrays
                            tempdict[key] = np.asarray(tempdict[key])
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
            
        self.from_dict(**tempdict)
