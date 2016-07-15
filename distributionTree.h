/* Open source system for classification learning from very large data
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

const NumericAttribute NOPARENT = std::numeric_limits<NumericAttribute>::max();  // used because some compilers won't accept std::numeric_limits<NumericAttribute>::max() here

class dtNode {
public:
  dtNode();   // default constructor - init must be called after construction
  dtNode(const CategoricalAttribute att);
  dtNode(const CategoricalAttribute a, unsigned int noValues); //For kdb-condDisc
  dtNode(InstanceStream::MetaData const* meta, const CategoricalAttribute att);
  ~dtNode();

  void init(InstanceStream::MetaData const* meta, const CategoricalAttribute a);  // initialise a new uninitialised node
  void clear();                          // reset a node to be empty

  // returns the start of the X=v,Y=y counts for value v
  inline InstanceCount &ref(const CatValue v, const CatValue y) {
    return xyCount.ref(v, y);
  }

  // returns the count X=v,Y=y
  inline InstanceCount getCount(const CatValue v, const CatValue y) {
    return xyCount.ref(v, y);
  }

  void updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned int depthRemaining, unsigned long long int &pc, double &apd, unsigned long long int &zc);

  fdarray<InstanceCount> xyCount;  // joint count indexed by x val the y val
  ptrVec<dtNode> children;
  CategoricalAttribute att;        // the Attribute whose values select the next child
  std::vector<NumValue> cuts_;     // stores the cuts for the numeric attributes (only for kdb-condDisc)
  std::vector<std::vector<NumValue> > numValues_;     // stores the numeric values for the different classes to be conditional discretised (only for kdb-condDisc2)

private:
  static InstanceStream::MetaData const* metaData_; // save just one metadata pointer for the whole tree
};

class distributionTree
{
public:
  distributionTree();   // default constructor - init must be called after construction
  distributionTree(InstanceStream::MetaData const* metaData, const CategoricalAttribute att);
  ~distributionTree(void);

  void init(InstanceStream const& stream, const CategoricalAttribute att);
  void clear();                            // reset a tree to be empty

  void update(const instance &i, const CategoricalAttribute att, const std::vector<CategoricalAttribute> &parents);

  // update classDist using the evidence from the tree about i
  void updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i); 
  // update classDist using the evidence from the tree about i for kdb k=k (required for kdb selectiveK)
  void updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, unsigned int k); 
  //This method discounts i (Pazzani's trick for loocv)
  void updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i);  
  //This method discounts i (Pazzani's trick for loocv) using kdb k=k (the specified k value)
  void updateClassDistributionloocv(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_);  
  //This method updates classDist by using the discretised value of the attribute a, that is v, conditioned on its parents (kdb-condDisc methods)
  void updateClassDistributionloocvWithNB(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_);

  dtNode* getdTNode();
private:
  dtNode dTree;
  InstanceStream::MetaData const* metaData_;
};
