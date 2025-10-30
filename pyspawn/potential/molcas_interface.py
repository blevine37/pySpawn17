import shutil
import os
import sys
import numpy as np
import subprocess

class Input:

    def __init__(self,project,labels, coords, basis, charge, mult, nactel, inactive, actorb,
    nstates,cbackprop, castarget, method, pt2,imaginary=0.2,ipea=0.0):
        self.labels=labels
        self.coords=coords
        self.basis=basis
        self.charge=charge
        self.mult=mult
        self.nactel=nactel
        self.inactive=inactive
        self.actorb=actorb
        self.nstates=nstates
        self.imaginary=imaginary
        self.ipea=ipea
        self.castarget=castarget+1
        self.cbackprop=cbackprop
        self.method = method
        self.pt2=pt2
        self.input =""
        self.input_name=project+'.inp'


    def write_gateway(self):
        gateway="&GATEWAY\nCOORD\n{}\n\n".format(len(self.labels))
        for index, (name, x, y, z) in enumerate(zip(self.labels, self.coords[::3], self.coords[1::3], self.coords[2::3])):
            atom="{0}{1} {2:12.8f} {3:12.8f} {4:12.8f}\n".format(name, index+1, x*0.52918,y*0.52918,z*0.52918)
            gateway += atom

        gateway += "\nGroup = nosym\nTitle = pySpawn\n"
        gateway += "Basis = " + self.basis + "\n\n"
        gateway += "RICD\n"
        self.input += gateway

    def write_seward(self):
        seward = "&SEWARD\nEXPERT\n\n"#&SCF\n\n"
        seward+= "\n >> COPY $CurrDir/INPORB INPORB\n\n"
        self.input +=seward
    
    def write_rasscf(self):
        rasscf = "&RASSCF\nSpin = {0}\nNactel = {1}\nInact = {2}\nRas2 = {3}\nCIRoot = {4} {4} 1\nLumorb\nPRWF = 1e-15".format(self.mult, self.nactel, self.inactive, self.actorb, self.nstates)
        rasscf += "\n >> SAVE $Project.JobIph $CurrDir/JobIph\n\n".format(self.cbackprop)
        if self.method == "casscf":
            rasscf += "\n >> SAVE $Project.JobIph JOB002\n\n"
        rasscf += "\n >> SAVE $Project.RasOrb $CurrDir/RasOrb\n\n".format(self.cbackprop)
        self.input+=rasscf
    
    def write_caspt2(self):
        caspt2 = "&CASPT2\nimag={0}\nipea={1}\nmaxiter=200\n".format(self.imaginary, self.ipea)
        if self.pt2 == "xms":
            caspt2 += "x"
        elif self.pt2=="ms":
            caspt2 += ""
        elif self.pt2=="rms":
            caspt2 += "r"
        elif self.pt2=='xdw':
            caspt2 += 'x'
        caspt2 +="multistate={0} {1}\n".format(self.nstates,[int(state) for state in range(1, self.nstates+1)]).replace("[", "").replace("]", "").replace(",", "")
        if self.pt2 =='xdw':
            caspt2 += "DWTYpe=3\nDWMS=1\n"
        caspt2 +="GRDT\nRLXROOT={}\nNOPROP\nPRWF = 1e-15\n\n".format(self.castarget)
        caspt2 += "\n >> SAVE $Project.JobMix $CurrDir/JobMix\n\n".format(self.cbackprop)
        caspt2 += "\n >> SAVE $Project.JobMix JOB002\n\n"
        self.input += caspt2

    def write_alaska(self):
        alaska="&ALASKA\nROOT = {}\n\n".format(self.castarget)
        self.input+=alaska
    
    def write_rassi(self):
        if self.method == "caspt2":
            rassi = " >> COPY $CurrDir/JobMix.old JOB001\n\n" 
        else:
            rassi = " >> COPY $CurrDir/JobIph.old JOB001\n\n" 
        rassi+= "&RASSI\nNROF = 2 {0} {0}\n".format(self.nstates)
        rassi+= ' '.join([str(index) for index in range(1, self.nstates+1)])
        rassi+="\n"
        rassi+= ' '.join([str(index) for index in range(1, self.nstates+1)])
        rassi+="\nSTOVERLAP\n"
        self.input+=rassi

    def write_input(self):

        with open(self.input_name, "w") as inp:
            inp.write(self.input)

##############################################################################################################
    
class ReadOutput:
    
    def __init__(self, project,atoms, nstates,method,pt2="xms"): #,nstates):
        self.atoms = atoms
        self.output = project+".out"
        self.nstates= nstates
        self.energies = []
        self.gradients = []
        self.method = method
        self.pt2 = pt2
        self.civectors = []
        self.overlap = []
        self.status=project+".status"
    
    def results(self):
        return {"energy": self.energies, "gradient": self.gradients, "ci_overlap": self.overlap}#, "civectors": self.civectors}

    def check_happy_landing(self):
        # Check if the calculation was successful by looking for the string 
        # "Happy landing starting from the bottom of the file, otherwise return error and exit
        with open(self.status, "r") as out:
            for line in out:
                if "Happy landing" in line:
                    pass
                else:
                    error = "Error: Molcas calculation failed. Check {} for more information".format(self.output)
                    print(error)
                    sys.exit(1)
    
    def get_energy(self):
        energies = []
        with open(self.output, "r") as out:
            for line in out:
                if self.method == "casscf":
                    if "RASSCF root number" in line:
                        energies.append(float(line.split()[-1]))
                elif self.method == "caspt2":
                    if self.pt2 == "xms":
                        if "XMS-CASPT2 Root" in line:
                            energies.append(float(line.split()[-1]))
                    elif self.pt2 == 'ms':
                        if "MS-CASPT2 Root" in line:
                            energies.append(float(line.split()[-1]))
                    elif self.pt2 == 'rms':
                        if "RMS-CASPT2 Root" in line:
                            energies.append(float(line.split()[-1]))
                    elif self.pt2 == 'xdw':
                        if "XDW-CASPT2 Root" in line:
                            energies.append(float(line.split()[-1]))
                        
        self.energies=np.array(energies)

    def get_gradients(self):
        # Get the gradients from the output file
        gradients=[]
        with open(self.output, "r") as out:
            for line in out:
                if "Molecular gradients" in line:
                    #go to seven line after
                    for _ in range(8):
                        current_line = next(out)
                    for atom in range(len(self.atoms)):
                        grad = []
                        grad.append(float(current_line.split()[-3]))
                        grad.append(float(current_line.split()[-2]))
                        grad.append(float(current_line.split()[-1]))
                        current_line = next(out)
                        grad = np.array(grad)
                        gradients.append(grad)
        self.gradients=np.array(gradients)

    def readci(self):
        #actually never called, but once I wrote it it can stay for any future application
        if self.method == "casscf":
            for state in range(1, self.nstates+1):
                vectorfile="molcas.VecDet.{}".format(state)
                with open(vectorfile, "r") as out:
                    lines = out.readlines()
                    for vect in range(1, len(lines)):
                        self.civectors.append(float(lines[vect].split()[0]))
                    
        elif self.method == "caspt2":
            with open(self.output, "r") as out:
                for line in out:
                    if "Conf  SGUGA info        Occupation    Coefficient         Weight" in line:
                        newci=True
                        vec = next(out)
                        self.civectors.append(float(vec.split()[-2]))
                        vec = next(out)
                        while newci:
                            try:
                                if vec.split()[0].isdigit():
                                    self.civectors.append(float(vec.split()[-2]))
                                    vec = next(out)
                            except:
                                newci=False

    def get_overlap(self): #, output='rassi.out'):
        overlap = []
        with open(self.output, "r") as out:

            for line in out:
                if "OVERLAP MATRIX FOR THE ORIGINAL STATES:" in line:
                    current_line = next(out)
                    current_line = next(out)
                    for state in range(self.nstates):
                        if state < 5:
                            current_line = next(out)
                        if state >= 5 and state < 10:
                            current_line = next(out)
                            current_line = next(out)

                    if self.nstates < 5:
                        for state in range(self.nstates):
                            overlap.extend([float(x) for x in (current_line.split()[:self.nstates])])
                            current_line = next(out)
                            full_indexes = self.nstates+state+1
                            if full_indexes > 5:
                                current_line = next(out)
                    if self.nstates >= 5 and self.nstates < 10:
                        for state in range(self.nstates):
                            overlap.extend([float(x) for x in (current_line.split()[:self.nstates])])
                            current_line = next(out)
                            overlap.extend([float(x) for x in (current_line.split()[:self.nstates-5])])
                            current_line = next(out)
                            full_indexes = self.nstates+state+1
                            if full_indexes > 10:
                                current_line = next(out)
        
        ## WARNING: UP Tp 10 states now, needs to be adpted for more
        if self.nstates > 10:
            print('Error: Currently, the interface works with up to 10 singlet states. get_overlap in molcas_interface needs to be updated.')
            sys.exit(1)

        
        matrix = np.array(overlap).reshape(self.nstates,self.nstates)
        #print(matrix)
        self.overlap = matrix
        #matrix = np.zeros((self.nstates,self.nstates))
        #element=0
        #print(overlap)
        #for i in range(self.nstates):
        #    for j in range(i,self.nstates):
        #        matrix[i][j] = matrix[j][i] = overlap[element]
        #        element+=1
        #self.overlap = matrix
        return matrix

                        
        


##############################################################################################################

class Environment():

    def __init__(self, project, qmdir, orbfile,molcasdir,python3,tmpdir="pyspawn/molcas"):
        self.qmdir = qmdir
        self.tmpdir = tmpdir
        self.orbfile = orbfile
        self.python3 = python3
        self.molcasdir = molcasdir
        self.project= project
    
    def setup_molcas(self):
        #if not os.path.exists(self.qmdir):
        #    os.makedirs(self.qmdir)
        #try:
        #    shutil.copy(self.orbfile, self.qmdir)
        #except IOError:
        #    print("\nError: Initial Orbital file not found. Please provide one\n")
        #    sys.exit(1)

        os.chdir(self.qmdir)
        os.system("export MOLCAS_KEEP_WORKDIR=NO")
        workdir = "WorkDir={}\nexport WorkDir".format(self.tmpdir)
        os.system(workdir)

    #def backup_orb(self):
    #    shutil.copy(self.orbfile, self.qmdir)

    def update_inporb(self):
        shutil.copy("JOBIPHPrev", self.orbfile)
    
    #def cleantmp(self):
    #    shutil.rmtree(self.tmpdir)
    
        #os.chdir(self.tmpdir)
        
    def run_molcas(self):
        
        try:
            os.system(self.python3+' '+self.molcasdir+"/pymolcas "+self.project+".inp > "+self.project+".out")
        except IOError:
            print("Error: pymolcas not found in PATH")
            sys.exit(1)     
    
    def run_rassi(self):
        #command = "export Project=molcas"
        #process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            os.system(self.python3+' '+self.molcasdir+"/pymolcas rassi.inp > rassi.out")
        except IOError:
            print("Error: pymolcas not found in PATH")
            sys.exit(1)


            



