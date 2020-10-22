from __future__ import division
import ROOT
import os 
#from __future__ import division

#credit to: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def chunkify(myList, numChunks):
    numChunks = min (numChunks, len(myList))
    if numChunks == 0:
        print "Why are you trying to make 0 chunks or trying to chunkify an empty list? That doesn't make sense. Exiting..."
        return 
    if len(myList)/numChunks == 1:
        print "Why are you trying to put all the files into one chunk? That doesn't make sense, if you wanted this, you wouldn't need to use this function to accomplish that. Exiting..."
        return 
    chunkSize, remainder = divmod (len(myList), numChunks)
    return (myList[i*chunkSize + min(i, remainder):(i+1)* chunkSize + min(i+1, remainder)] for i in range(numChunks))
    
    


#Proof of Concept, can comment out after testing 
# testList = range(10)
# #print testList
# choppedUpList =  list(chunkify(testList,3)) #remember to do "list(chunkify(myList, numChunks))" to get the output as something useful!
# print "choppedUpList is:", choppedUpList


chunksPerDir = 2 #Set this to whatever you think is appropriate based on how many files in the directory 

outDir = 'scratchOut_22Oct2020'
if not os.path.exists(outDir):
    os.makedirs(outDir)

#Remember that SAFEPathToFiles must be written like this: '  root://cmseos.fnal.gov//store/user/mhadley/15June2020_lowpTTauPreprocessedSFlatNtuples/' 
#Those 2 spaces at the front are important!! Well, could be 1 space but I think 2 makes easier reading. Remember also the / at the end and the beginning are important
#Thanks Marguerite and David for reminding me of the SAFE path syntax for LPC EOS

def loopChunkAndHaddFilesInTheDir(pathToDir, SAFEPathToFiles):
    filesInTheDirList = [myF for myF in os.listdir(pathToDir) if os.path.isfile(os.path.join(pathToDir, myF))]
    print "filesInTheDirList:", filesInTheDirList
    
    choppedUpList = list(chunkify(filesInTheDirList, chunksPerDir))
    print "choppedUpList:", choppedUpList
    
    counter = 0
    for i in choppedUpList:
        haddCommand = 'hadd  '  + outDir + '/bigFile_' #Not the spaces at the end of the 'hadd   ' string, they are important
#        SAFEpathToFiles = '  /scatch2/'
        counter += 1
        haddCommand += str(counter) + '.root'
        print haddCommand
        chunkInChoppedUpListList = list(i)
        for el in chunkInChoppedUpListList: #el for element 
            print el
            haddCommand += SAFEPathToFiles + el 
        print haddCommand
        os.system(haddCommand)
    return 
    
loopChunkAndHaddFilesInTheDir('scratch2', '  scratch2/')