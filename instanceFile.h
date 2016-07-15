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
#pragma once
#include "instanceStream.h"
#include "utils.h"
#include "FILEtype.h"

class InstanceFile : public InstanceStream
{
public:
  #ifdef SIXTYFOURBITCOUNTS
    typedef unsigned long long int LineCount; ///< a count of a number of instances
    #define LCFMT "llu"
  #else  // SIXTYFOURBITCOUNTS
    typedef unsigned long int LineCount; ///< a count of a number of instances
    #define LCFMT "lu"
  #endif // SIXTYFOURBITCOUNTS

  inline void testLineCount(LineCount &cnt) {
    if (cnt == std::numeric_limits<LineCount>::max()) {
      error("Line count exceeds system limit of %" LCFMT, std::numeric_limits<LineCount>::max());
    }
  }

  InstanceFile(const char* metaFileName, const char* dataFileName);
  ~InstanceFile(void);

  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd();                                             ///< true if we have advanced past the last instance
  InstanceCount size();                                       ///< the number of instances in the stream.  This may require a pass through the stream to determine so should be used only if absolutely necessary.

  // InstanceFile specific methods
  void resetSource(const char* name);                                 ///< change the source file from which the data are read

private:
  class ioMetadata : public InstanceStream::MetaData {
  public:
    ioMetadata(const bool case_sensitive = true);
    ~ioMetadata(void);

    unsigned int getNoClasses() const;                          ///< return the number of classes
    const char* getClassName(CatValue y) const;   ///< return the name for a class
    const char* getClassAttName() const;                        ///< return the name for the class attribute
    unsigned int getNoCatAtts() const;                          ///< return the number of categorical attributes
    unsigned int getNoValues(CategoricalAttribute att) const;   ///< return the number of values for a categorical Attribute
    const char* getCatAttName(CategoricalAttribute att) const;  ///< return the name for a categorical Attribute
    const char* getCatAttValName(CategoricalAttribute att, CatValue val) const; ///< return the name for a categorical Attribute value
    unsigned int getNoNumAtts() const;                          ///< return the number of numeric attributes
    const char* getNumAttName(CategoricalAttribute att) const;  ///< return the name for a numeric Attribute
    unsigned int getPrecision(NumericAttribute att) const;         ///< return the precision to which values of a numeric attribute should be output
    int hasCatMissing(CategoricalAttribute att) const;          ///<  return whether a categorical attribute contains missing values
    int hasNumMissing(NumericAttribute att) const;          ///<  return whether a numeric attribute contains missing values
    const char* getName();                                      ///< return a string that gives a meaningful name for the stream
    bool areNamesCaseSensitive();                               ///< true iff name comparisons are case sensitive
    void setAllAttsMissing();

    static const int MAX_NAME_LENGTH = 200;

    typedef unsigned int Attribute;   

    void parse(const char* fn);         ///< parse a metadata file
    
    /// read a categorical value from a file
    inline CatValue readCatVal(FILEtype *f, const Attribute att, int &c, const LineCount line) const {
      return readVal(f, colWidth[att], attValNames[internalAtt[att]], attNames[att], c, line);
    }

    /// read a class from a file
    virtual CatValue readClass(FILEtype *f, int &c, const LineCount line, const char delimiter = ',') const;
    
    /// read a number not in scientific format
    NumValue readSimpleNum(FILEtype *f, int &c, unsigned int &charsRemaining, unsigned int &precision) const;
    
    /// read a number including numbers in scientific format
    NumValue readNum(FILEtype *f, const NumericAttribute att, int &c);

    inline unsigned int noOfClasses() const { return classNames_.size(); }
    inline unsigned int noOfAttributes() const { return attNames.size(); }

    std::vector<AttType> attTypes;  ///< the type of each Attribute
    std::vector<unsigned int> colWidth;   ///< the width of a fixed width column.  0 for variable width delimited fields
    std::vector<unsigned int> precision;  ///< the precision to which a numeric value is specified
    std::vector<char *> attNames;         ///< the name of each Attribute
    std::vector<std::vector<char *> > attValNames;  ///< the names of the values for each categorical Attribute
    std::vector<int> missingVals;       ///< whether an attribute has o not missing values 
    std::vector<char *> classNames_;     ///< the names of each class
    std::vector<Attribute> internalAtt; ///< a map from the io Attribute index to the categorical or numeric Attribute index
    std::vector<Attribute> cat2io;      ///< a map from categorical Attribute indexes to io indexes
    std::vector<Attribute> num2io;      ///< a map from numeric Attribute indexes to io indexes
    Attribute class2io;                 ///< the io index of the class
    const bool caseSensitive;           /// true iff the input file is case sensitive

    typedef enum {gigal_FORMAT, LIBSVM_FORMAT} InputFormat;
    InputFormat inputFormat_;        ///< is the file in Gigal or libSVM format
    char const* filename;  ///< the name of the file

  protected:
    /// read a categorical value from the input file
    CatValue readVal(FILEtype *f, const unsigned int colWidth, const std::vector<char *> &valNames, const char* attName, int &c, LineCount line, const char delimiter = ',') const;
    
    /// get a categorical value from a string
    CatValue getVal(const char *valname, const std::vector<char *> &valNames, const char *attname) const;

  private: 
    /// read the next delimted token from the input file
    void readName(FILEtype *f, char *buffer, const char terminator, int &c) const;
    
    
    /// read the next fixed width token from the input file
    void readFWName(FILEtype *f, char *buffer, const unsigned int width, int &c) const;
  };

  FILE *f;               ///< file pointer for the file
  ioMetadata metadata_; ///< the metadata for the file
  LineCount line;        ///< the current input line
  InstanceCount count_;  ///< a count of the number of instances read so far
};
