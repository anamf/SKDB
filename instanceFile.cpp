/* Open source system for classification learning from very large data
** Class for an instance stream read sequentially from a file
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/
#include "instanceFile.h"
#include "utils.h"
#include "globals.h"
#include <ctype.h>
#include <sys/stat.h>

InstanceFile::InstanceFile(const char* metaFileName, const char* dataFileName) : f(NULL)
{ metaData_ = &metadata_;

  // parse the metafile
  metadata_.parse(metaFileName);
  
  // open the data file for reading
  resetSource(dataFileName);
}


InstanceFile::~InstanceFile(void)
{ fclose(f);
}

void InstanceFile::resetSource(const char* fn) {
  if (f != NULL) fclose(f);

  metadata_.filename = fn;
  line = 0;
  count_ = 0;

  f = fopen(fn, "r");

  if (f == NULL) error("Cannot open input file %s", fn);
}

void InstanceFile::rewind() {
  ::rewind(f);
  line = 0;
  count_ = 0;
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceFile::advance() {
  int c = getc(f);

  // skip lines containing only non-printable characters
  while (c <= ' ') {
    if (c == EOF) return false;
    c = getc(f);
  }

  line++;
  while (c != '\n' && c != '\r' && c != EOF) {
    c = getc(f);
  }

  return true;
}

bool InstanceFile::advance(instance &inst) {
  if (metadata_.inputFormat_ == ioMetadata::gigal_FORMAT) {
    int c = getc(f);

    while (isspace(c)) {
      if (c == '\n') line++;
      c = getc(f);
    }

    ioMetadata::Attribute att = 0;

    while (c != '\n' && c!= '\r' && c != EOF) {
      if (att == metadata_.noOfAttributes()) {
        error("More values than attributes on line %" LCFMT, line);
      }

      switch (metadata_.attTypes[att]) {
        case CATEGORICAL:
          //catVals[metadata_.internalAtt[att]] = metadata_.getVal(f, metadata_.colWidth[att], metadata_.attValNames[metadata_.internalAtt[att]], metadata_.attNames[att], c, line);
          setCatVal(inst, metadata_.internalAtt[att], metadata_.readCatVal(f, att, c, line));
          break;
        case CLASS:
          //theClass = metadata_.getVal(f, metadata_.colWidth[att], metadata_.classNames, metadata_.attNames[att], c, line);
          setClass(inst, metadata_.readClass(f, c, line));
          break;
        case NUMERIC:
          setNumVal(inst, metadata_.internalAtt[att], metadata_.readNum(f, metadata_.internalAtt[att], c));
          break;
      }

      if (c == ',') c = getc(f);

      while (c != '\n' && c != '\r' && isspace(c)) c = getc(f);

      att++;
    }

    if (att < metadata_.noOfAttributes()) {
      if (att) {
        error("Fewer values than attributes on line %" LCFMT, line);
      }

      return false;
    }
  }
  else {
    // libSVM format input
    int c = getc(f);


    while (isspace(c)) {
      if (c == '\n') line++;
      c = getc(f);
    }

    if (c == EOF) {
      return false;
    }

    setClass(inst, metadata_.readClass(f, c, line, ' '));

    while (c == ' ') c = getc(f);

    for (NumericAttribute a = 0; a < getNoNumAtts(); a++) {
      setNumVal(inst, a, 0);
    }

    ioMetadata::Attribute att;

    while (c != '\n' && c != EOF) {
      att = 0;

      if (c < '0' || c > '9') error("libSVM format error: encountered '%c' when expecting an Attribute index", c);
      
      // get the Attribute index
      while (c >= '0' && c <= '9') {
        att *= 10;
        att += c - '0';
        c = getc(f);
      }

      if (att == 0 || att > metadata_.noOfAttributes()) {
        error("Attribute index %d out of range on line %" LCFMT, att, line);
      }

      att--;    // indexes start from 1, internal indexes start fro m0

      if (c != ':') {
        error("Missing : following Attribute index %d on line %" LCFMT, att, line);
      }
      
      c = getc(f);

      setNumVal(inst, att, metadata_.readNum(f, att, c));

      while (c == ' ') c = getc(f);
    }
  }

  line++;

  return true;
}

// check whether there is any more input in the file
bool InstanceFile::isAtEnd() {
  int c = fgetc(f);

  while (isspace(c)) c = fgetc(f);

  if (c == EOF) return true;
  
  ungetc(c, f);
  
  return false;
}

/// scan the file to count the number of instances it contains.
/// if a count file exists (a file whith the file's filename + ".cnt") and is more recet than the file, read the count from the count file instead.
InstanceCount InstanceFile::size() {
  if (isAtEnd()) return count_; // if we have completed a pass through the data we can simply return the count

  const bool update = false;  // allow future support for saving file counts

  char *cntfname;
  safeAlloc(cntfname, strlen(metadata_.filename) + 5);

  sprintf(cntfname, "%s%s", metadata_.filename, ".cnt");

  struct stat buf;

  int result;
   
  // Get data associated with cnt file: 
  result = stat( cntfname, &buf );

  // Check if the cnt file exists 
  if( result == 0 ) {
      struct stat buf2;

    // Get data associated with fname: 
    result = stat( metadata_.filename, &buf2 );

    // Check if the file exists 
    if( result == 0 ) {
      // check whether the cnt file is more recent
      if (buf.st_mtime > buf2.st_mtime) {
        InstanceCount cnt = 0;

        FILE *f = fopen(cntfname, "r");

        if (f != NULL) {
          (void)fscanf(f, "%" ICFMT, &cnt);
          fclose(f);
          delete []cntfname;
          return cnt;
        }
      }
    }
  }

  FILE *f = fopen(metadata_.filename, "r");

  int c = getc(f);
  InstanceCount cnt = 0;

  while (c != EOF) {
    // skip lines containing only non-printable characters
    while (c <= ' ') {
      if (c == EOF) goto exitloop;
      c = getc(f);
    }

    cnt++;
    while (c != '\n' && c != '\r' && c != EOF) {
      c = getc(f);
    }
exitloop:;
  }

  fclose(f);

  if (update) {
    FILE *f = fopen(cntfname, "w");
    fprintf(f, "%" ICFMT "\n", cnt);
  }

  delete []cntfname;
  
  return cnt;
}

InstanceFile::ioMetadata::~ioMetadata(void)
{
  for (Attribute a = 0; a < noOfAttributes(); a++) {
    delete []attNames[a];
  }

  for (CategoricalAttribute a = 0; a < attValNames.size(); a++) {
    for (CatValue v = 0; v < attValNames[a].size(); v++) {
      delete []attValNames[a][v];
    }
  }

  for (CatValue v = 0; v < classNames_.size(); v++) {
    delete []classNames_[v];
  }
}

const char* InstanceFile::ioMetadata::getName() {
  return filename;
}

///< true iff name comparisons are case sensitive
bool InstanceFile::ioMetadata::areNamesCaseSensitive() {
  return caseSensitive;
}

/// return the name for a class
const char* InstanceFile::ioMetadata::getClassName(CatValue att) const {
  return classNames_[att];
}

/// return the name for the class attribute
const char* InstanceFile::ioMetadata::getClassAttName() const {
  return attNames[class2io];
}

unsigned int InstanceFile::ioMetadata::getNoClasses() const {
  return classNames_.size();
}

unsigned int InstanceFile::ioMetadata::getNoCatAtts() const {
  return attValNames.size();
}

unsigned int InstanceFile::ioMetadata::getNoValues(CategoricalAttribute att) const {
  assert(att < attValNames.size());
  return attValNames[att].size();
}

const char* InstanceFile::ioMetadata::getCatAttName(CategoricalAttribute att) const {
  return attNames[cat2io[att]];
}

const char* InstanceFile::ioMetadata::getCatAttValName(CategoricalAttribute att, CatValue val) const {
  return attValNames[att][val];
}

unsigned int InstanceFile::ioMetadata::getNoNumAtts() const {
  return num2io.size();
}

const char* InstanceFile::ioMetadata::getNumAttName(NumericAttribute att) const {
  return attNames[num2io[att]];
}

/// return the precision to which values of a numeric attribute should be output
unsigned int InstanceFile::ioMetadata::getPrecision(NumericAttribute att) const {
  return precision[att];
}

/// return whether a categorical attribute contains missing values
int InstanceFile::ioMetadata::hasCatMissing(CategoricalAttribute att) const {
  return missingVals[cat2io[att]];
}

/// return whether a numeric attribute contains missing values
int InstanceFile::ioMetadata::hasNumMissing(NumericAttribute att) const {
  return missingVals[num2io[att]];
}

InstanceFile::ioMetadata::ioMetadata(const bool case_sensitive) : caseSensitive(case_sensitive)
{
}

void InstanceFile::ioMetadata::setAllAttsMissing(){
  missingVals.assign(missingVals.size(),1);
}

// reads a string terminated by EOF, \n or the nominated terminator
// names are truncated at maxNameLength
void InstanceFile::ioMetadata::readName(FILEtype *f, char *buffer, const char terminator, int &c) const {
  int i = 0;
  
  while (true) {
    if (c < ' ' || c == EOF || c == terminator) {
      while (i > 0 && isspace(buffer[i-1])) i--;  // strip trailing spaces
      buffer[i] = '\0';
      return;
    }

    if (c >= ' ' && i < MAX_NAME_LENGTH) {  // truncate names at maxNameLength, discard non printing characters
      buffer[i++] = static_cast<char>(c);
    }

    c = getc(f);
  }
}

// reads a fixed width string
// names are truncated at maxNameLength
void InstanceFile::ioMetadata::readFWName(FILEtype *f, char *buffer, const unsigned int width, int &c) const {
  unsigned int i = 0;
  
  while (true) {
    if (c == '\n' || c == '\r' || c == EOF || i >= width) {
      while (i > 0 && isspace(buffer[i-1])) i--;  // strip trailing spaces
      buffer[i] = '\0';
      return;
    }

    if (i < MAX_NAME_LENGTH) {  // truncate names at maxNameLength
      buffer[i++] = static_cast<char>(c);
    }

    c = getc(f);
  }
}

// parse a metadata file
// The format is
//   <line> ::= [ : <qualifier> { , <qualifier> } : ] <name> : numeric \n
//            | [ : <qualifier> { , <qualifier> } : ] <name> : <name> { , <name> } \n
//            | :: <comment> \n
//   <qualifier> ::= width=<unsigned int>
//                 | index=<unsigned int>-<unsigned int>
//                 | class
void InstanceFile::ioMetadata::parse(const char *fn) {
  FILEtype *f;
  int c;
  char buffer[MAX_NAME_LENGTH+1];
  LineCount line = 1;
  bool classDeclaredBySpecifier = false;
  bool classDeclared = false;
  unsigned int noNumAtts = 0;
  unsigned int noCatAtts = 0;
  unsigned int noAttributes = 0;      ///< the total number of attributes (including the class)

  f = fopen(fn, "r");

  if (f == NULL) {
    error("Cannot open metadata file '%s'", fn);
  }

  if (verbosity >= 1) printf("Loading metadata file %s\n", fn);

  noAttributes  = 0;
  colWidth.clear();
  precision.clear();
  attTypes.clear();
  attNames.clear();
  attValNames.clear();

  inputFormat_ = gigal_FORMAT;

  c = getc(f);

  while (c != EOF) {
    bool classAtt = false;
    unsigned int width = 0;
    bool indexed = false;
    int startIndex = 0;
    int endIndex = 0;

    while (isspace(c)) {
      if (c == '\n') line++;
      c = getc(f);
    }

    if (c == EOF) break;

    readName(f, buffer, ':', c);

    if (c != ':') {
      error("Missing colon on line %" LCFMT, line);
    }

    if (buffer[0] == '\0') {
      // this means that the first character was ':'
      // get qualifiers

      c = getc(f);

      readName(f, buffer, ':', c);

      if (buffer[0] == '\0') {
        // comment
        while (c != '\n' && c != '\r' && c != EOF) {
          c = getc(f);
        }
        continue;
      }

      if (streq(buffer, "libsvm")) {
        // libsvm format
        if (noAttributes)  error("Cannot combine libsvm declaration with attribute declarations");
        if (c != ':') error("Expect : after libsvm qualifier");

        c = getc(f);                // advance past the last terminator
        readName(f, buffer, ':', c); // get the number of attributes

        if (buffer[0] == '\0') error("Missing number of attributes after libsvm qualifier");
        if (c != '\n' && c != '\r') error("Unexpected elements following number of elements");

        int i = 0;

        while (buffer[i] >= '0' && buffer[i] <= '9') {
          noNumAtts *= 10;
          noNumAtts += buffer[i] - '0';
          i++;
        }

        // make the required number of copies
        const int indexLen = printWidth(noNumAtts);
        int nameLen = indexLen + 2;
        if (nameLen > MAX_NAME_LENGTH) {
          nameLen = MAX_NAME_LENGTH;
        }

        char *buf;

        for (NumericAttribute i = 1; i <= noNumAtts; i++) {
          safeAlloc(buf, nameLen+1);
          sprintf(buf, "a_%d", i);
          attNames.push_back(buf);
          colWidth.push_back(0);
          attTypes.push_back(NUMERIC);
          internalAtt.push_back(noAttributes);
          num2io.push_back(noAttributes++);
          precision.push_back(0);
          missingVals.push_back(0);
        }

        inputFormat_ = LIBSVM_FORMAT;

        continue;
      }

      bool finished = false;
      unsigned int i = 0;

      while (!finished) {
        // identify the next token
        const unsigned int start = i;

        while (buffer[i] != '\0' && buffer[i] != ',' && buffer[i] != '=') {
          i++;
        }

        const char terminator = buffer[i];
        buffer[i] = '\0';

        if (streq(buffer+start, "class")) {
          // this is the class attribute
          if (terminator == '=') {
            error("Unexpected = following class specifier on line %" LCFMT, line);
          }
          if (classDeclaredBySpecifier) {
            error("Only one class can be specified. Second class specifier on line %" LCFMT, line);
          }
          classAtt = true;
          classDeclaredBySpecifier = true;
        }
        else if (streq(buffer+start, "width")) {
          // the field is fixed width
          if (terminator != '=') {
            error("Expected = following width specifier on line %" LCFMT, line);
          }

          i++;  // skip '='

          while (buffer[i] >= '0' && buffer[i] <= '9') {
            width *= 10;
            width += buffer[i++] - '0';
          }
        }
        else if (streq(buffer+start, "index")) {
          // there are endIndex-startIndex+1 repetitions of an identical attribute
          // they all share a common name indexed by the specified range of indices
          if (terminator != '=') {
            error("Expected = following index specifier on line %" LCFMT, line);
          }

          indexed = true;
          i++;  // skip '='

          while (buffer[i] >= '0' && buffer[i] <= '9') {
            startIndex *= 10;
            startIndex += buffer[i++] - '0';
          }

          if (buffer[i] != '-') {
            error("Expected - following first index on line %" LCFMT, line);
          }

          i++;  // skip '-'

          while (buffer[i] >= '0' && buffer[i] <= '9') {
            endIndex *= 10;
            endIndex += buffer[i++] - '0';
          }
        }
        else {
            error("Error in meta information on line %" LCFMT, line);
        }

        if (buffer[i] == '\0') {
          finished = true;
        }
        else if (buffer[i] == ',') {
          i++;
        }
        else {
            error("Error in meta information on line %" LCFMT, line);
        }
      }

      c = getc(f);                // advance past the last terminator
      readName(f, buffer, ':', c); // get the attribute name
    }


    char *buf;

    if (indexed) {
      // make the required number of copies
      int nameLen = strlen(buffer);
      const int indexLen = printWidth(endIndex);
      if (nameLen + indexLen >= MAX_NAME_LENGTH) {
        nameLen = MAX_NAME_LENGTH - indexLen;
      }

      for (int i = startIndex; i <= endIndex; i++) {
        safeAlloc(buf, strlen(buffer)+indexLen+2);
        strcpy(buf, buffer);
        sprintf(buf+nameLen, "_%d", i);
        attNames.push_back(buf);
        colWidth.push_back(width);
      }
    }
    else {
      safeAlloc(buf, strlen(buffer)+1);
      strcpy(buf, buffer);
      attNames.push_back(buf);
      colWidth.push_back(width);
    }

    if (streq(buf, "class") && !classDeclaredBySpecifier) {
      classAtt = true;
    }

    c = getc(f);

    while (isspace(c)) {
      c = getc(f);
    }

    if (c == EOF || c == '\n' || c == '\r') {
      error("Missing attribute type on line %" LCFMT, line);
    }

    // get the first value or the type
    readName(f, buffer, ',', c);

    if (streq(buffer, "numeric")) {
      //c = getc(f); 
      int missing = 0;
      if(c == ','){
        c = getc(f);
        assert (c == '?');
        missing = 1;
        c = getc(f);
      }
      for (int i = startIndex; i <= endIndex; i++) {
        attTypes.push_back(NUMERIC);
        internalAtt.push_back(noNumAtts++);
        num2io.push_back(noAttributes++);
        precision.push_back(0);
        missingVals.push_back(missing);
      }
    }
    else {
      // must be a list of the allowed values for the attribute
      std::vector<char *> valNames;

      char *buf;
      safeAlloc(buf, strlen(buffer)+1);
      strcpy(buf, buffer);
      valNames.push_back(buf);
      
      int missing = 0;
      
      if (streq(buffer,"?"))
        missing = 1;

      while (c != '\n' && c != '\r' && c != EOF) {
        if (isspace(c)) {
          c = getc(f);
        }
        else {
          if (c != ',') {
            error("Syntax eror on line %" LCFMT, line);
          }
          
          c = getc(f);
          while (isspace(c)) c = getc(f);
          
          readName(f, buffer, ',', c);
          
          for (std::vector<char *>::const_iterator it = valNames.begin(); it != valNames.end(); it++) {
            if (streq(buffer, *it, caseSensitive)) {
              error("Value '%s' declared twice for attribute '%s'", buffer, attNames.back());
            }
          }

          safeAlloc(buf, strlen(buffer)+1);
          strcpy(buf, buffer);
          valNames.push_back(buf);
          
          if (streq(buffer,"?"))
            missing = 1;
        }
      }

      for (int i = startIndex; i <= endIndex; i++) {
        if (classAtt) {
          if (indexed) {
            error("Class cannot be indexed");
          }
          attTypes.push_back(CLASS);
          internalAtt.push_back(0);
          classDeclared = true;
          classNames_ = valNames;
          class2io = noAttributes++;
          missingVals.push_back(missing);
        }
        else {
          attTypes.push_back(CATEGORICAL);
          internalAtt.push_back(noCatAtts++);
          cat2io.push_back(noAttributes++);

          // need to duplicate the value names
          if (i > startIndex) {
            for (CatValue v = 0; v < valNames.size(); v++) {
              char *buf;
              safeAlloc(buf, strlen(valNames[v])+1);
              strcpy(buf, valNames[v]);
              valNames[v] = buf;
            }
          }
          attValNames.push_back(valNames);
          missingVals.push_back(missing);
        }
      }
    }

    if (c == '\n') {
      line++;
      c = getc(f);
    }
    else {
      if (c != EOF) {
        error("Syntax error on line %" LCFMT, line);
      }
    }
  }

  if (!classDeclared) {
    error("No class declared in metadata file '%s'", fn);
  }

  fclose(f);

  assert(noAttributes == noNumAtts + noCatAtts + 1);
  assert(attTypes.size() == noAttributes);
  assert(colWidth.size() == noAttributes);
  assert(attNames.size() == noAttributes);
  assert(precision.size() == noNumAtts);
  assert(attValNames.size() == noCatAtts);
  assert(internalAtt.size() == noAttributes);
  assert(cat2io.size() == noCatAtts);
  assert(num2io.size() == noNumAtts);
  assert(missingVals.size() == noAttributes);
}

CatValue InstanceFile::ioMetadata::getVal(const char *valname, const std::vector<char *> &valNames, const char *attname) const {
  unsigned int i = 0;

  while (i < valNames.size()) {
    if (streq(valname, valNames[i], caseSensitive)) {
      return i;
    }

    i++;
  }

  error("'%s' is not a value for attribute '%s'", valname, attname);
}

CatValue InstanceFile::ioMetadata::readVal(FILEtype *f, const unsigned int colWidth, const std::vector<char *> &valNames, const char * attName, int &c, LineCount line, const char delimiter) const {
  assert(noOfAttributes()>0); // check that the metadata has been parsed before it is used

  char buffer[MAX_NAME_LENGTH+1];

  if (colWidth) {
    readFWName(f, buffer, colWidth, c);
  }
  else {
    readName(f, buffer, delimiter, c);
  }

  unsigned int i = 0;

  while (i < valNames.size()) {
    if (streq(buffer, valNames[i], caseSensitive)) {
      return i;
    }

    i++;
  }

  error("'%s' is not a value for attribute '%s' on line %" LCFMT, buffer, attName, line);
}

CatValue InstanceFile::ioMetadata::readClass(FILEtype *f, int &c, const LineCount line, const char delimiter) const {
  return readVal(f, colWidth[class2io], classNames_, attNames[class2io], c, line, delimiter);
}

// get a signed floating point number
NumValue InstanceFile::ioMetadata::readSimpleNum(FILEtype *f, int &c, unsigned int &charsRemaining, unsigned int &precision) const {
  assert(noOfAttributes()>0); // check that the metadata has been parsed before it is used

  bool sign = false;
  double v = 0;

  precision = 0;  // count the number of decimal places that are specified

  if (c == '-') {
    sign = true;
    charsRemaining--;
    c = getc(f);
  }

  while (c >= '0' && c <= '9' && charsRemaining != 0) {
    v = 10 * v + c - '0';
    charsRemaining--;
    c = getc(f);
  }

  if (c == '.' && charsRemaining != 0) {
    charsRemaining--;
    c = getc(f);
    double divisor = 10;

    while (c >= '0' && c <= '9' && charsRemaining != 0) {
      v += (c - '0') / divisor;
      divisor *= 10;
      charsRemaining--;
      precision++;
      c = getc(f);
    }
  }

  if (sign) v = -v;

  return static_cast<NumValue>(v);
}

// get a number for the specified attribute
NumValue InstanceFile::ioMetadata::readNum(FILEtype *f, const NumericAttribute att, int &c) {
  assert(noOfAttributes()>0); // check that the metadata has been parsed before it is used

  if (c == '?') {
    // missing value
    c = getc(f);        // advance past character
    return MISSINGNUM;  // return the missing value
  }

  unsigned int charsRemaining = colWidth[att];
  unsigned int thisPrecision;

  if (charsRemaining == 0) charsRemaining = std::numeric_limits<unsigned int>::max();

  double v = readSimpleNum(f, c, charsRemaining, thisPrecision);

  if (c == 'e' || c == 'E' && charsRemaining != 0) {
    unsigned int ignore;

    c = getc(f);
    charsRemaining--;
    double exponent = readSimpleNum(f, c, charsRemaining, ignore);

    thisPrecision -= static_cast<unsigned int>(exponent);

    v *= pow(10.0, exponent);
  }

  if (thisPrecision > precision[att]) precision[att] = thisPrecision;
    
  return static_cast<NumValue>(v);
}

#if 0
bool instance::get(FILEtype *f, LineCount &line) {
  if (metadata_.inputFormat == ifgigal) {
    int c = getc(f);

    while (c == ' ') c = getc(f);

    Attribute att = 0;

    while (c != '\n' && c != EOF) {
      if (att == metadata_.noAttributes) {
        error("More values than attributes on line %" LCFMT, line);
      }

      switch (metadata_.attTypes[att]) {
        case categorical:
          //catVals[metadata_.internalAtt[att]] = metadata_.getVal(f, metadata_.colWidth[att], metadata_.attValNames[metadata_.internalAtt[att]], metadata_.attNames[att], c, line);
          catVals[metadata_.internalAtt[att]] = metadata_.getCatVal(f, att, c, line);
          break;
        case class_type:
          //theClass = metadata_.getVal(f, metadata_.colWidth[att], metadata_.classNames, metadata_.attNames[att], c, line);
          theClass = metadata_.getClass(f, c, line);
          break;
        case numeric:
          numVals[metadata_.internalAtt[att]] = metadata_.getNum(f, metadata_.internalAtt[att], c, line);
          break;
      }

      if (c == ',') c = getc(f);

      while (c == ' ') c = getc(f);

      att++;
    }

    if (att < metadata_.noAttributes) {
      if (att) {
        error("Fewer values than attributes on line %" LCFMT, line);
      }
      return false;
    }
  }
  else {
    // libSVM format input
    int c = getc(f);

    while (c == ' ') c = getc(f);

    if (c == '\n' || c == EOF) return false;

    theClass = metadata_.getClass(f, c, line, ' ');

    while (c == ' ') c = getc(f);

    numVals.assign(metadata_.noNumAtts, 0);

    Attribute att;

    while (c != '\n' && c != EOF) {
      att = 0;

      // get the Attribute index
      while (c >= '0' && c <= '9') {
        att *= 10;
        att += c - '0';
        c = getc(f);
      }

      if (att == 0 || att > metadata_.noAttributes) {
        error("Attribute index %d out of range on line %" LCFMT, att, line);
      }

      att--;    // indexes start from 1, internal indexes start fro m0

      if (c != ':') {
        error("Missing : following Attribute index %d on line %" LCFMT, att, line);
      }
      
      c = getc(f);

      numVals[att] = metadata_.getNum(f, att, c, line);

      while (c == ' ') c = getc(f);
    }
  }
    
  return true;
}
#endif
