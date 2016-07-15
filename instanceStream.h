/* Open source system for classification learning from very large data
** Class for an input source for instances
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

#include "instance.h"

#include <assert.h>


#ifdef SIXTYFOURBITCOUNTS
typedef unsigned long long int InstanceCount; // a count of a number of instances
#define ICFMT "llu"
#else  // SIXTYFOURBITCOUNTS
typedef unsigned int InstanceCount; // a count of a number of instances
#define ICFMT "u"
#endif // SIXTYFOURBITCOUNTS


class InstanceStream
{
public:
  InstanceStream();
  ~InstanceStream(void);

  virtual void rewind() = 0;                                              ///< return to the first instance in the stream
  virtual bool advance() = 0;                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  virtual bool advance(instance &inst) = 0;                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  virtual bool isAtEnd() = 0;                                             ///< true if we have advanced past the last instance
  virtual InstanceCount size() = 0;                                       ///< the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.
  
  class MetaData {
  public:
    virtual unsigned int getNoClasses() const = 0;                          ///< return the number of classes
    virtual const char* getClassName(CatValue att) const = 0;               ///< return the name for a class
    virtual const char* getClassAttName() const = 0;                        ///< return the name for the class attribute
    virtual unsigned int getNoCatAtts() const = 0;                          ///< return the number of categorical attributes
    virtual int hasCatMissing(CategoricalAttribute att) const = 0;          ///< return whether a categorical attribute contains missing values
    virtual int hasNumMissing(NumericAttribute att) const = 0;              ///< return whether a numeric attribute contains missing values
    virtual unsigned int getNoValues(CategoricalAttribute att) const = 0;   ///< return the number of values for a categorical attribute
    virtual const char* getCatAttName(CategoricalAttribute att) const = 0;  ///< return the name for a categorical Attribute
    virtual const char* getCatAttValName(CategoricalAttribute att, CatValue val) const = 0; ///< return the name for a categorical attribute value
    virtual unsigned int getNoNumAtts() const = 0;                          ///< return the number of numeric attributes
    virtual const char* getNumAttName(NumericAttribute att) const = 0;      ///< return the name for a numeric attribute
    virtual unsigned int getPrecision(NumericAttribute att) const = 0;      ///< return the precision to which values of a numeric attribute should be output
    virtual const char* getName() = 0;                                      ///< return a string that gives a meaningful name for the stream
    virtual bool areNamesCaseSensitive() = 0;                               ///< true iff name comparisons are case sensitive
    virtual void setAllAttsMissing() = 0;
  };

  class MetaDataFilter : public MetaData {
  public:
    virtual unsigned int getNoClasses() const;                          ///< return the number of classes
    virtual const char* getClassName(CatValue att) const;               ///< return the name for a class
    virtual const char* getClassAttName() const ;                        ///< return the name for the class attribute
    virtual unsigned int getNoCatAtts() const;                          ///< return the number of categorical attributes
    virtual int hasCatMissing(CategoricalAttribute att) const;          ///< return whether a categorical attribute contains missing values
    virtual int hasNumMissing(NumericAttribute att) const;              ///< return whether a numeric attribute contains missing values
    virtual unsigned int getNoValues(CategoricalAttribute att) const;   ///< return the number of values for a categorical attribute
    virtual const char* getCatAttName(CategoricalAttribute att) const;  ///< return the name for a categorical Attribute
    virtual const char* getCatAttValName(CategoricalAttribute att, CatValue val) const; ///< return the name for a categorical attribute value
    virtual unsigned int getNoNumAtts() const;                          ///< return the number of numeric attributes
    virtual const char* getNumAttName(NumericAttribute att) const;      ///< return the name for a numeric attribute
    virtual unsigned int getPrecision(NumericAttribute att) const;      ///< return the precision to which values of a numeric attribute should be output
    virtual const char* getName();                                      ///< return a string that gives a meaningful name for the stream
    virtual bool areNamesCaseSensitive();                               ///< true iff name comparisons are case sensitive
    virtual void setAllAttsMissing(); 

    inline void setSource(MetaData* source) { source_ = source; }

  protected:
    MetaData* source_;
  };

  inline MetaData* getMetaData() const { return metaData_; }
  void printMetadata(const char* filename) const;                         /// print a Gigal metadatafile for the stream

  inline unsigned int getNoClasses() const { return metaData_->getNoClasses(); }                         ///< return the number of classes
  inline const char* getClassName(CatValue att) const { return metaData_->getClassName(att); }              ///< return the name for a class
  inline const char* getClassAttName() const { return metaData_->getClassAttName(); }                       ///< return the name for the class attribute
  inline unsigned int getNoCatAtts() const { return metaData_->getNoCatAtts(); }                         ///< return the number of categorical attributes
  inline int hasCatMissing(CategoricalAttribute att) const { return metaData_->hasCatMissing(att); }      ///< return whether a categorical attribute contains missing values
  inline int hasNumMissing(NumericAttribute att) const { return metaData_->hasNumMissing(att); }          ///< return whether a numeric attribute contains missing values
  inline unsigned int getNoValues(CategoricalAttribute att) const { return metaData_->getNoValues(att); }  ///< return the number of values for a categorical attribute
  inline const char* getCatAttName(CategoricalAttribute att) const { return metaData_->getCatAttName(att); } ///< return the name for a categorical Attribute
  inline const char* getCatAttValName(CategoricalAttribute att, CatValue val) const { return metaData_->getCatAttValName(att, val); } ///< return the name for a categorical attribute value
  inline unsigned int getNoNumAtts() const { return metaData_->getNoNumAtts(); }                         ///< return the number of numeric attributes
  inline const char* getNumAttName(NumericAttribute att) const { return metaData_->getNumAttName(att); }     ///< return the name for a numeric attribute
  inline unsigned int getPrecision(NumericAttribute att) const { return metaData_->getPrecision(att); }     ///< return the precision to which values of a numeric attribute should be output
  inline const char* getName() { return metaData_->getName(); }                                     ///< return a string that gives a meaningful name for the stream
  inline bool areNamesCaseSensitive() { return metaData_->areNamesCaseSensitive(); }  
  inline void setAllAttsMissing() {metaData_->setAllAttsMissing(); }

protected:
  inline void setCatVal(instance &inst, const CategoricalAttribute att, const CatValue v) { inst.setCatVal(att, v); }
  inline void setNumVal(instance &inst, const NumericAttribute att, const NumValue v) { inst.setNumVal(att, v); }
  inline void setClass(instance &inst, const CatValue v) { inst.setClass(v); }

  MetaData* metaData_;
};
