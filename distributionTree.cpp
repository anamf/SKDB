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
#include "distributionTree.h"
#include "smoothing.h"
#include "utils.h"
#include <assert.h>

InstanceStream::MetaData const* dtNode::metaData_;

dtNode::dtNode(InstanceStream::MetaData const* meta, const CategoricalAttribute a) : att(NOPARENT), xyCount(meta->getNoValues(a),meta->getNoClasses()) {
  metaData_ = meta;
}

dtNode::dtNode(const CategoricalAttribute a) : att(NOPARENT), xyCount(metaData_->getNoValues(a),metaData_->getNoClasses()) {
}

//The parameter const CategoricalAttribute a is useless but intentionally added (used in kdb-condDisc)
dtNode::dtNode(const CategoricalAttribute a, unsigned int noValues) : att(NOPARENT), xyCount(noValues,metaData_->getNoClasses()) {
}

dtNode::dtNode() : att(NOPARENT) {
}

dtNode::~dtNode() {
}

void dtNode::init(InstanceStream::MetaData const* meta, const CategoricalAttribute a) {
  metaData_ = meta;
  att = NOPARENT;
  xyCount.assign(meta->getNoValues(a), meta->getNoClasses(), 0);
  children.clear();
}

void dtNode::clear() {
  xyCount.clear();
  children.clear();
  att = NOPARENT;
}

distributionTree::distributionTree() 
{
}

distributionTree::distributionTree(InstanceStream::MetaData const* metaData, const CategoricalAttribute att) : metaData_(metaData), dTree(metaData, att)
{
}

distributionTree::~distributionTree(void)
{
}

void distributionTree::init(InstanceStream const& stream, const CategoricalAttribute att)
{
  metaData_ = stream.getMetaData();
  dTree.init(metaData_, att);
}

void distributionTree::clear()
{
  dTree.clear();
}

dtNode* distributionTree::getdTNode(){
  return &dTree;
}

void distributionTree::update(const instance &i, const CategoricalAttribute a, const std::vector<CategoricalAttribute> &parents) {
  const CatValue y = i.getClass();
  const CatValue v = i.getCatVal(a);

  dTree.ref(v, y)++;

  dtNode *currentNode = &dTree;

  for (unsigned int d = 0; d < parents.size(); d++) { 

    const CategoricalAttribute p = parents[d];

    if (currentNode->att == NOPARENT || currentNode->children.empty()) {
      // children array has not yet been allocated
      currentNode->children.assign(metaData_->getNoValues(p), NULL);
      currentNode->att = p;
    }

    assert(currentNode->att == p);
    
    dtNode *nextNode = currentNode->children[i.getCatVal(p)];

    // the child has not yet been allocated, so allocate it
    if (nextNode == NULL) {
      currentNode = currentNode->children[i.getCatVal(p)] = new dtNode(a);
    }
    else {
      currentNode = nextNode;
    }

    currentNode->ref(v, y)++;
  }
}

// update classDist using the evidence from the tree about i
void distributionTree::updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    const unsigned int noOfVals = metaData_->getNoValues(a);

    for (CatValue v = 1; v < noOfVals; v++) {
      totalCount += dt->getCount(v, y);
    }

    classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, noOfVals);
  }
}

// update classDist using the evidence from the tree about i
void distributionTree::updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, unsigned int k) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT) && (depth<k) ) { //We want to consider kdb k=k, we stop when the depth reached is equal to k
    depth++;
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    const unsigned int noOfVals = metaData_->getNoValues(a);

    for (CatValue v = 1; v < noOfVals; v++) {
      totalCount += dt->getCount(v, y);
    }

    classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, noOfVals);
  }
}

// update classDist using the evidence from the tree about i and deducting it at the same time (Pazzani's trick for loocv)
// require that at least 1 value (minCount = 1) be used for probability estimation
void distributionTree::updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i) {
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNode *next = dt->children[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2) 
        break;

    dt = next;
    att = dt->att;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
      totalCount += dt->getCount(v, y);
    }    
    
    if(y!=i.getClass())
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
    else
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
  }
}

void distributionTree::updateClassDistributionloocv(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_){
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;
  
  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT)) { //We want to consider kdb k=k
    const CatValue v = i.getCatVal(att);
     // sum over all values of the Attribute for the class to obtain count[y, parents]
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      InstanceCount totalCount = dt->getCount(0, y);
      for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
        totalCount += dt->getCount(v, y);
      }    
     if(y!=i.getClass())
          classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
      else
          classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
    }
    dtNode *next = dt->children[v];
    if (next == NULL) {
      for(int k=depth+1; k<=k_; k++){
        for (CatValue y = 0; y < metaData_->getNoClasses(); y++) 
            classDist[k][y] = classDist[depth][y];
      }
      return;
    };

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2){ 
        depth++;
          // sum over all values of the Attribute for the class to obtain count[y, parents]
      for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
        InstanceCount totalCount = dt->getCount(0, y);
        for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
          totalCount += dt->getCount(v, y);
        }    

        if(y!=i.getClass())
            classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
        else
            classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
      }
      for(int k=depth+1; k<=k_; k++){
        for (CatValue y = 0; y < metaData_->getNoClasses(); y++) 
            classDist[k][y] = classDist[depth][y];
      }
      return;
    }

    dt = next;
    att = dt->att; 
  depth++;
  }
  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
      totalCount += dt->getCount(v, y);
    }    
   if(y!=i.getClass())
     classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
   else
     classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
  }
  for(int k=depth+1; k<=k_; k++){
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) 
      classDist[k][y] = classDist[depth][y];
  }
  
}

void dtNode::updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned int depth, unsigned long long int &pc, double &apd, unsigned long long int &zc) {
  if (depth == parents.size()  || children.empty()) {
    for (CatValue v = 0; v < metaData_->getNoValues(target); v++) {
      pc++;

      apd += (depth-apd) / static_cast<double>(pc);

      for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
        if (getCount(v, y) == 0) zc++;
      }
    }
  }
  else {
    for (CatValue v = 0; v < metaData_->getNoValues(parents[depth]); v++) {
      if (children[v] == NULL) {
          unsigned long int pathsMissing = 1;
          
          for (unsigned int i = depth; i < parents.size(); i++) pathsMissing *= metaData_->getNoValues(parents[i]);

          pc += pathsMissing;

          apd += pathsMissing * ((depth-apd)/(pc-pathsMissing/2.0));

          for (CatValue tv = 0; tv < metaData_->getNoValues(target); tv++) {
            for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
              if (getCount(tv, y) == 0) zc++;
            }
          }
      }
      else {
        children[v]->updateStats(target, parents, k, depth+1, pc, apd, zc);
      }
    }
  }
}


void distributionTree::updateClassDistributionloocvWithNB(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_){
  dtNode *dt = &dTree;
  CategoricalAttribute att = dTree.att;
  
  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT)) { //We want to consider kdb k=k
    const CatValue v = i.getCatVal(att);
     // sum over all values of the Attribute for the class to obtain count[y, parents]
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      InstanceCount totalCount = dt->getCount(0, y);
      for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
        totalCount += dt->getCount(v, y);
      }    
     if(y!=i.getClass())
          classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
      else
          classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
    }
    dtNode *next = dt->children[v];
    if (next == NULL) {
      for(int k=depth+1; k<=k_; k++){
        for (CatValue y = 0; y < metaData_->getNoClasses(); y++) 
            classDist[k][y] = classDist[depth][y];
      }
      return;
    };

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2){ 
        depth++;
          // sum over all values of the Attribute for the class to obtain count[y, parents]
      for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
        InstanceCount totalCount = dt->getCount(0, y);
        for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
          totalCount += dt->getCount(v, y);
        }    

        for(int k=depth; k<=k_; k++){
            if(y!=i.getClass())
              classDist[k][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
            else
             classDist[k][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
        }
      }
      return;
    }

    dt = next;
    att = dt->att; 
  depth++;
  }
  // sum over all values of the Attribute for the class to obtain count[y, parents]
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    InstanceCount totalCount = dt->getCount(0, y);
    for (CatValue v = 1; v < metaData_->getNoValues(a); v++) {
      totalCount += dt->getCount(v, y);
    }    
   if(y!=i.getClass())
     classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), totalCount, metaData_->getNoValues(a));
   else
     classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, totalCount-1, metaData_->getNoValues(a));
  }
  for(int k=depth+1; k<=k_; k++){
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) 
      classDist[k][y] = classDist[depth][y];
  }
  
}
