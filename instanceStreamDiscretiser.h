/* Open source system for classification learning from very large data
** Class for a discretisation filter for instance streams
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
#include "instanceStreamFilter.h"
#include "discretiser.h"
#include <vector>

class InstanceStreamDiscretiser :
  public InstanceStreamFilter
{
public:
  InstanceStreamDiscretiser(){}
  InstanceStreamDiscretiser(const char* name, char*const*& argv, char*const* end);
  ~InstanceStreamDiscretiser(void);

  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool advanceNumeric(instance &inst);                        ///< advance to the next instance in the stream without discretizing numeric values.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd();                                             ///< true if we have advanced past the last instance
  InstanceCount size();                                       /// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

  void setSource(InstanceStream &source);                     ///< set the source for the filter

  CatValue discretise(const NumValue v, const NumericAttribute a) const;  ///< return the discretised value a numeric attribute value. 
  
  void discretiseInstance(const instance &inst, instance &instDisc);                      ///< return the discretised version of the instance. @param inst the instance to discretise. 

  class MetaData : public InstanceStream::MetaDataFilter {
  public:
    MetaData() {}
    unsigned int getNoCatAtts() const;                          ///< return the number of categorical attributes
    unsigned int getNoValues(CategoricalAttribute att) const;   ///< return the number of values for a categorical attribute
    const char* getCatAttName(CategoricalAttribute att) const;  ///< return the name for a categorical Attribute
    const char* getCatAttValName(CategoricalAttribute att, CatValue val) const; ///< return the name for a categorical attribute value
    unsigned int getNoNumAtts() const;                          ///< return the number of numeric attributes
    unsigned int getNoOrigCatAtts() const;                      ///< return the number of categorical attributes before discretization
    const char* getNumAttName(CategoricalAttribute att) const;  ///< return the name for a numeric attribute
    unsigned int getPrecision(NumericAttribute att) const;      ///< return the precision to which values of a numeric attribute should be output
    const char* getName();                                      ///< return a string that gives a meaningful name for the stream
    int hasCatMissing(CategoricalAttribute att) const;          ///< return whether a categorical attribute contains missing values
    int hasNumMissing(NumericAttribute att) const;              ///< return whether a numeric attribute contains missing values
    void setAllAttsMissing();                                   ///< set all attributes as missing (for compatibility with previous versions)

    void setSource(InstanceStream::MetaData* source);
    
    
    

  public:
    std::vector<std::vector<NumValue> > cuts;             ///< the cuts for each discretised numeric attribute
    std::vector<std::vector<char *> > discAttValNames_;   ///< the names of the values for each discretised attribute
    
  private:
    unsigned int noOfSourceCatAtts_;                      ///< stores the value of source->getNoCatAtts() to save repeated calls
  };

  inline MetaData* getMetaData() { return &metaData_; }

private:
  instance sourceInst_;            ///< the current instance from the source stream. Maintain one instance record to save repeated construction/destruction.
  discretiser *theDiscretiser;     ///< the discretiser used to select cuts
  InstanceCount targetSampleSize_; ///< The size of the sample on which the discretization is performed
  MetaData metaData_;              ///< the metaData for the discretised stream
  bool allNumWithMiss_;             ///< temporary flag to indicate (if true) that the metadata has no information about missing values in numeric attributes (only afects smoothing)
  bool printMetaData_;
  char * filename_;

};
