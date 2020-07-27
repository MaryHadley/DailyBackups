#include <fstream>
#include <iostream>
#include <string.h>
#include "xAna_mc.C+"
//#include "xAna_mc_C.so" //bad bad bad, causes craziness, instead use the .C+ notation above

void run_mc () {

std::ifstream inFile("1.txt");
std::string myStr;
std::string fileInQuestion;
int counter = 0;
int lineInFile = 0;

while (std::getline(inFile, myStr))
{ if (counter == lineInFile) {fileInQuestion = myStr;}
  else if (counter > lineInFile) {break;}
  counter = counter + 1;
    
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

//gROOT->LoadMacro("xAna_mc.C+");

const char* inpaths[] = {ptr_to_the_string_that_is_char_array};

xAna_mc(inpaths, 1);

}

int main() {



run_mc();
return 0;
}