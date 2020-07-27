#include <fstream>
#include <iostream>
#include <string.h>
#include "xAna_mc.C+"
//#include "xAna_mc_C.so" //bad bad bad, causes craziness, instead use the .C+ notation above

///Instructions to Run ///
// run as, for example:
//root -l 'run_mc.C(1,0)'
//where the first argument (1 in this example) is the int associated with the textFile to open
//and the second argument (0 here) is the line in that textFile we want to pick out 
//(remember line counting starts at 0!)
//So in words, type into the command line root -l, then single quote, run_mc.C(firstArg, secondArg) then close single quote
//where firstArg and secondArg are ints

void run_mc ( int txtFileNum, int lineToGet) {

std::ifstream inFile;

if (txtFileNum == 1) {inFile.open("1.txt");}
//std::ifstream inFile("1.txt");
std::string myStr;
std::string fileInQuestion;
int counter = 0;
int lineInFile;
lineInFile = lineToGet;

//if (txtFileNum == 1) {inFile = "1.txt"}
if (inFile.is_open()){
while (std::getline(inFile, myStr))
{ if (counter == lineInFile) {fileInQuestion = myStr;}
  else if (counter > lineInFile) {break;}
  counter = counter + 1;
    
    }
}
std::cout << "fileInQuestion: \n" << fileInQuestion << std::endl;

//Example of how to do this from: https://www.geeksforgeeks.org/convert-string-char-array-cpp/
int n = fileInQuestion.length();
std::cout << "n is: \n" << n << std::endl;

char char_array[n+1]; //no idea why you need n+1, but did it based on the example

strcpy (char_array, fileInQuestion.c_str());



std::cout << "char_array: \n" << char_array << std::endl;

const char* ptr_to_the_string_that_is_char_array = char_array;

std::cout << "ptr_to_the_string_that_is_char_array: " << ptr_to_the_string_that_is_char_array << std::endl;
std::cout << "value pointed to by ptr_to_the_string_that_is_char_array:" << *ptr_to_the_string_that_is_char_array << std::endl;
//gROOT->LoadMacro("xAna_mc.C+");

const char* inpaths[] = {ptr_to_the_string_that_is_char_array};

xAna_mc(inpaths, 1);

}

