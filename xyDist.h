/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
** Class for handling a joint distribution between an Attribute and a class
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
#include "smoothing.h"

// model the joint distribution for each individual x-value and the class


typedef InstanceCount const* ySubDist;  ///< A pointer to the start of an array of InstanceCounts for a conditional class distribution

class xyDist
{
public:
  xyDist();                   ///< constructor without initialisation of InstanceStream specific data
  xyDist(InstanceStream *is); ///< constructor that reads the distribution from the stream
  ~xyDist(void);

  void reset(InstanceStream *is); ///< initialise with InstanceStream specific information but do not read the distribution

  void update(const instance &inst); ///< update the distribution according to the given instance
  
  void clear();

  // p(a=v|Y=y) using M-estimate
  inline double p(CategoricalAttribute a, CatValue v, CatValue y) {
    return mEstimate(counts_[a][v*noOfClasses_+y], classCounts[y], metaData_->getNoValues(a));
  }

  // p(a=v, Y=y) using M-estimate
  inline double jointP(CategoricalAttribute a, CatValue v, CatValue y) {
    return (counts_[a][v*noOfClasses_+y]+M/(metaData_->getNoValues(a)*metaData_->getNoClasses()))/(count+M);
  }

  // p(a=v) using M-estimate
  inline double p(CategoricalAttribute a, CatValue v) {
    return (getCount(a,v)+M/(metaData_->getNoValues(a)))/(count+M);
  }

  inline double p(CatValue y) {
    return (classCounts[y]+M/metaData_->getNoClasses())/(count+M);
  }

  // count[A=v,Y=y]
  inline InstanceCount getCount(CategoricalAttribute a, CatValue v, CatValue y) const {
    return counts_[a][v*noOfClasses_+y];
  }

  // count[A=v]
  inline InstanceCount getCount(CategoricalAttribute a, CatValue v) const {
    InstanceCount c = 0;
    const ySubDist ySD = getYSubDist(a, v);
    for (CatValue y = 0; y < noOfClasses_; y++) {
      c+= ySD[y];
    }
    return c;
  }

  // count[Y=y]
  inline InstanceCount getClassCount(CatValue y) const {
    return classCounts[y];
  }

  inline unsigned int getNoClasses() const { return noOfClasses_; }

  inline unsigned int getNoAtts() const { return counts_.size(); }

  inline unsigned int getNoCatAtts() const { return counts_.size(); }

  inline unsigned int getNoValues(CategoricalAttribute a) const { return metaData_->getNoValues(a); }

  inline const ySubDist getYSubDist(CategoricalAttribute a, CatValue v) const {
    return &counts_[a][v*noOfClasses_];
  }

  InstanceCount count;
  std::vector<InstanceCount> classCounts;

private:
 // InstanceStream *instanceStream_;
  InstanceStream::MetaData* metaData_;
  /// Instance counts indexed by attribute, then attribute value, then class.
  /// The inner two vectors are flattened into a single vector, indexed by val*noOfClasses + class.
  std::vector<std::vector<InstanceCount> > counts_;
  unsigned int noOfClasses_;    ///< store the number of classes for use in indexing the inner vector
};
