#include <fstream>
#include <iostream>

{
std::string getFileForCondorToProcess() {
std::ifstream inFile("1.txt");
//std::cout << inFile; //Ok this makes things go crazy 

std::string myStr;
std::string fileInQuestion;

int counter = 0;
int fileNumber= 0; //fix me

while (std::getline(inFile, myStr))
{
  if (counter == fileNumber) {fileInQuestion = myStr;}
  else if (counter > fileNumber) {break;}
  counter = counter + 1; //I think could also write +=
//  std::cout << "fileInQuestion: \n" << fileInQuestion;
}

std::cout << "fileInQuestion: " << fileInQuestion << std::endl;
return fileInQuestion;
}

std::string fileForCondorToProcess = getFileForCondorToProcess();

{	
  gROOT->LoadMacro("xAna_mc.C+");
  
  
  

   
 
  const char* inpaths[] = {
//      "file:/cms/heindl/2018/HF_calibration/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/crab_DYJetsToLL_M-50_Fall18/181127_103024/ggTree.root"
 
    //   "file:/eos/cms/store/group/dpg_hcal/comm_hcal/mhadley/2017ReRecoBigSubmitForHFZtoe/secondTry/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/200501_223602/0000/ggTree_11.root"
    fileForCondorToProcess
 };

  xAna_mc(inpaths, 1);
}

}
