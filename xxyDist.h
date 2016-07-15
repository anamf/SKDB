/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
** Class for handling a joint distribution between two attributes and a class
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
#include "xyDist.h"

class constXYSubDist {
public:
  constXYSubDist(const std::vector<InstanceCount>* subDist, const unsigned int noOfClasses) : subDist_(subDist), noOfClasses_(noOfClasses) {}

  inline const ySubDist getYSubDist(const CategoricalAttribute x, const CatValue v) const { return &subDist_[x][v*noOfClasses_]; }
  inline const InstanceCount getCount(const CategoricalAttribute x, const CatValue v, const CatValue y) const { return subDist_[x][v*noOfClasses_+y]; }

private:
  const std::vector<InstanceCount>* subDist_;
  const unsigned int noOfClasses_;
};

class XYSubDist {
public:
  XYSubDist(std::vector<InstanceCount>* subDist, const unsigned int noOfClasses) : subDist_(subDist), noOfClasses_(noOfClasses) {}

  inline ySubDist getYSubDist(const CategoricalAttribute x, const CatValue v) { return &subDist_[x][v*noOfClasses_]; }
  inline InstanceCount getCount(const CategoricalAttribute x, const CatValue v, const CatValue y) { return subDist_[x][v*noOfClasses_+y]; }
  inline void incCount(const CategoricalAttribute x, const CatValue v, const CatValue y) { ++subDist_[x][v*noOfClasses_+y]; }

  std::vector<InstanceCount>* subDist_;
  const unsigned int noOfClasses_;
};

class xxyDist
{
public:
  xxyDist();
  xxyDist(InstanceStream& stream);
  ~xxyDist(void);

  void reset(InstanceStream& stream);

  void update(const instance& i);
  
  void clear();

  // p(x1=v1, x2=v2, Y=y) unsmoothed
  inline double rawJointP(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2, CatValue y) const {
    return (*constRef(x1,v1,x2,v2,y))/(xyCounts.count);
  }

  // p(x1=v1, x2=v2, Y=y) using M-estimate
  inline double jointP(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2, CatValue y) const {
    return (*constRef(x1,v1,x2,v2,y)+M/(metaData_->getNoValues(x1)*metaData_->getNoValues(x2)*noOfClasses_))/(xyCounts.count+M);
  }

  // p(x1=v1, x2=v2) using M-estimate
  inline double jointP(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2) const {
    return (getCount(x1,v1,x2,v2)+M/(metaData_->getNoValues(x1)*metaData_->getNoValues(x2)))/(xyCounts.count+M);
  }

  // p(x1=v1|Y=y, x2=v2) using M-estimate
  inline double p(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2, CatValue y) const {
    return (*constRef(x1,v1,x2,v2,y)+M/metaData_->getNoValues(x1))/(xyCounts.getCount(x2,v2,y)+M);
  }

  //inline double p(CatValue y) const {
  //  return xyCounts.p(y);
  //}

  // p(x1=v1, x2=v2, Y=y) unsmoothed
  inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2, CatValue y) const {
    return *constRef(x1,v1,x2,v2,y);
  }

  // count for instances x1=v1, x2=v2
  inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2) const {
    InstanceCount c = 0;

    for (CatValue y = 0; y < noOfClasses_; y++) {
      c+= getCount(x1,v1,x2,v2,y);
    }
    return c;
  }

  inline unsigned int getNoCatAtts() const { return count_.size(); }

  inline unsigned int getNoValues(const CategoricalAttribute a) const { return metaData_->getNoValues(a); }

  inline unsigned int getNoClasses() const { return noOfClasses_; }

  inline std::vector<InstanceCount>* getXYSubDist(CategoricalAttribute x1, CatValue v1) { 
    return &count_[x1][v1*x1];
  }

private:
  // count_[X1=x1][X2=x2][Y=y]
  inline InstanceCount *ref(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2, CatValue y) {
    if (x2 > x1) {
      CatValue t = x1;
      x1 = x2;
      x2 = t;
      t = v1;
      v1 = v2;
      v2 = t;
    }

    return &count_[x1][v1*x1+x2][v2*noOfClasses_+y];
  }

  // count_[X1=x1][X2=x2]
  inline InstanceCount *xxref(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2) {
    if (x2 > x1) {
      CatValue t = x1;
      x1 = x2;
      x2 = t;
      t = v1;
      v1 = v2;
      v2 = t;
    }

    return &count_[x1][v1*x1+x2][v2*noOfClasses_];
  }

  // count_[X1=x1][X2=x2][Y=y]
  inline const InstanceCount *constRef(CategoricalAttribute x1, CatValue v1, CategoricalAttribute x2, CatValue v2, CatValue y) const {
    if (x2 > x1) {
      CatValue t = x1;
      x1 = x2;
      x2 = t;
      t = v1;
      v1 = v2;
      v2 = t;
    }

    return &count_[x1][v1*x1+x2][v2*noOfClasses_+y];
  }

public:
  xyDist xyCounts;

private:
  InstanceStream::MetaData* metaData_;
  // a three-dimensional semiflattened representation of count_[X1=x1][X2=x2][Y=y] storing only X1 > X2
  // outer vector is indexed by X1
  // middle vector is indexed by x1*X2
  // inner vector is indexed by x2*y
  std::vector<std::vector<std::vector<InstanceCount> > > count_;
  unsigned int noOfClasses_;
};
