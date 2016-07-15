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

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>
#include <queue>

#include "kdbSelective.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "ALGLIB_specialfunctions.h"
#include "crosstab.h"

kdbSelective::kdbSelective(char*const*& argv, char*const* end) {
  name_ = "SELECTIVE-KDBclean";

  // defaults
  k_ = 1;
  
  selectiveK_ = false;
  onlyK_ = false;
  trainSize_ = 0;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }
    else if (streq(argv[0]+1, "selectiveK")) {
      selectiveK_ = true;
    }
    else if (streq(argv[0]+1, "onlyK")) {
      onlyK_ = true;
    }
    else {
      break;
    }
      
    name_ += argv[0];

    ++argv;
  }
}

kdbSelective::~kdbSelective(void)
{
}

void  kdbSelective::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void kdbSelective::reset(InstanceStream &is) {
  kdb::reset(is);
  
  order_.clear();
  active_.assign(noCatAtts_, true);
  
  if(onlyK_){
    foldLossFunct_.resize(k_+1); //+1 for NB (k=0)
  }else{
    foldLossFunct_.assign(noCatAtts_+1,0.0); //1 more for the prior
  }

  if(selectiveK_){
    foldLossFunctallK_.resize(k_+1); //+1 for NB (k=0)
    for(int i=0; i<= k_; i++){
      foldLossFunctallK_[i].assign(noCatAtts_+1,0);
    }
  }
  inactiveCnt_ = 0;
  trainSize_ = 0;  
}

void kdbSelective::train(const instance &inst) {
  if (pass_ == 1) {
    // in the first pass collect the xxy distribution
    dist_.update(inst);
    trainSize_++; // to calculate the RMSE for each LOOCV
  }
  else if(pass_ == 2){
    // on the second pass collect the distributions to the k-dependence classifier
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      dTree_[a].update(inst, a, parents_[a]);
    }
    classDist_.update(inst);
  }else{
      assert(pass_ == 3); //only for selective KDB
      if(selectiveK_){
          std::vector<std::vector<double> > posteriorDist(k_+1);//+1 for NB (k=0)
          for(int k=0; k< k_+1; k++){
              posteriorDist[k].assign(noClasses_,0.0);
          }
          //Only the class is considered
          for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist[0][y] = classDist_.ploocv(y, inst.getClass());//Discounting inst from counts
          }
          normalise(posteriorDist[0]);

          const CatValue trueClass = inst.getClass();          
          const double error = 1.0-posteriorDist[0][trueClass];
          foldLossFunctallK_[0][noCatAtts_] += error*error;
                
                    
          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                 it != order_.end(); it++){
              dTree_[*it].updateClassDistributionloocvWithNB(posteriorDist, *it, inst, k_);//Discounting inst from counts
              for(int k=0; k<= k_; k++){
                normalise(posteriorDist[k]);
                const double error = 1.0-posteriorDist[k][trueClass];
                foldLossFunctallK_[k][*it] += error*error;
              }
              
          }
      }else if(onlyK_){
          std::vector<std::vector<double> > posteriorDist(k_+1);//+1 for NB (k=0)
          for(int k=0; k<= k_; k++){
              posteriorDist[k].assign(noClasses_,0.0);
          }
          //Only the class is considered
          for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist[0][y] = classDist_.ploocv(y, inst.getClass());//Discounting inst from counts
          }      
                    
          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                 it != order_.end(); it++){
              dTree_[*it].updateClassDistributionloocvWithNB(posteriorDist, *it, inst, k_);//Discounting inst from counts           
              for(int k=0; k<= k_; k++){
                normalise(posteriorDist[k]);
              }
          }                    
          const CatValue trueClass = inst.getClass(); 
          for(int k=0; k<= k_; k++){
            normalise(posteriorDist[k]);
            const double error = 1.0-posteriorDist[k][trueClass];
            foldLossFunct_[k] += error*error;
          }
      }else{
         //Proper kdb selective
         std::vector<double> posteriorDist(noClasses_);
         std::vector<double> errorsAtts; // Store the att errors for this instance (needed for selectiveTest)
         errorsAtts.assign(noCatAtts_+1,0.0);
         //Only the class is considered
         for (CatValue y = 0; y < noClasses_; y++) {
           posteriorDist[y] = classDist_.ploocv(y,inst.getClass());//Discounting inst from counts
         }
         normalise(posteriorDist);
         const CatValue trueClass = inst.getClass();
         const double error = 1.0-posteriorDist[trueClass];

         foldLossFunct_[noCatAtts_] += error*error;
         errorsAtts[noCatAtts_] = error;


         for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                it != order_.end(); it++){
           dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst);//Discounting inst from counts
           normalise(posteriorDist);
           const double error = 1.0-posteriorDist[trueClass];

           foldLossFunct_[*it] += error*error;
           
           errorsAtts[*it] = error;
         }
      }
  }
}

/// true iff no more passes are required. updated by finalisePass()
bool kdbSelective::trainingIsFinished() {
    return pass_ > 3;
}

void kdbSelective::classify(const instance &inst, std::vector<double> &posteriorDist) {
  const unsigned int noClasses = noClasses_;

  for (CatValue y = 0; y < noClasses; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
    if (active_[x]) {
      if(selectiveK_ || onlyK_){
        dTree_[x].updateClassDistributionForK(posteriorDist, x, inst, bestK_);
      }else{
        dTree_[x].updateClassDistribution(posteriorDist, x, inst);
      }
    }
  }

  normalise(posteriorDist);
}

// creates a comparator for two attributes based on their relative mutual information with the class
class miCmpClass {
public:
  miCmpClass(std::vector<float> *m) {
    mi = m;
  }

  bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
    return (*mi)[a] > (*mi)[b];
  }

private:
  std::vector<float> *mi;
};


void kdbSelective::finalisePass() {
  if (pass_ == 1) {
    
    std::vector<float> mi;  
    crosstab<float> cmi = crosstab<float>(noCatAtts_);  //CMI(X;Y|C) = H(X|C) - H(X|Y,C) -> cmi[X][Y]
              
    getMutualInformation(dist_.xyCounts, mi);
    getCondMutualInf(dist_,cmi);
    
    dist_.clear();
    
    // sort the attributes on MI with the class

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      order_.push_back(a);
    }

    if (!order_.empty()) {
      miCmpClass cmp(&mi);

      std::sort(order_.begin(), order_.end(), cmp);          
       
       // proper KDB assignment of parents
       
       for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin()+1;  
                                                              it != order_.end(); it++){
         parents_[*it].push_back(order_[0]);
         for (std::vector<CategoricalAttribute>::const_iterator 
                                           it2 = order_.begin()+1; it2 != it; it2++) {
           // make parents into the top k attributes on mi that precede *it in order
           if (parents_[*it].size() < k_) {
             // create space for another parent
             // set it initially to the new parent.
             // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
             parents_[*it].push_back(*it2);
           }
           for (unsigned int i = 0; i < parents_[*it].size(); i++) {
             if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it]) {
               // move lower value parents down in order
               for (unsigned int j = parents_[*it].size()-1; j > i; j--) {
                 parents_[*it][j] = parents_[*it][j-1];
               }
               // insert the new att
               parents_[*it][i] = *it2;
               break;
             }
           }
         }
       }
    }
  }
  else if(pass_ == 3) {//only for selective KDB

    std::vector<CategoricalAttribute>::const_iterator bestattIt = order_.end()-1;
    bestK_ = 0;

    if(selectiveK_){
      //Proper kdb selective (RMSE)      
      for (unsigned int k=0; k<=k_;k++) {
        for (unsigned int att=0; att<noCatAtts_+1;att++) {
          foldLossFunctallK_[k][att] = sqrt(foldLossFunctallK_[k][att]/trainSize_);
        }
        foldLossFunctallK_[k][noCatAtts_] = foldLossFunctallK_[0][noCatAtts_]; //The prior is the same for all values of k_
      }
          
      double globalmin = foldLossFunctallK_[0][noCatAtts_];
      for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
        for (unsigned int k=0; k<=k_;k++) {
            if(foldLossFunctallK_[k][*it] < globalmin){
              globalmin = foldLossFunctallK_[k][*it];
              bestattIt = it;
              bestK_ = k;
            }
        }
      }
      if(verbosity>=2){
         for (unsigned int k=0; k<=k_;k++) {
            printf("k = %d : ",k);
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
              printf("%.3f,", foldLossFunctallK_[k][*it]);
            }
            printf("%.3f(class)\n", foldLossFunctallK_[k][noCatAtts_]);
          }
      }
      
    }else if(onlyK_){
      for (unsigned int k=0; k<=k_;k++) {
          foldLossFunct_[k] = sqrt(foldLossFunct_[k]/trainSize_);
          if(verbosity>=3){
                printf("k: %d = %.3f\n",k, foldLossFunct_[k]);
          }
      }
      double globalmin = foldLossFunct_[0];
      for (unsigned int k=0; k<=k_;k++) {
            if(foldLossFunct_[k] < globalmin){
              globalmin = foldLossFunct_[k];
              bestK_ = k;
            }
        }
    }else{//proper selective

      //Find the best attribute in order (to resolve ties in the best way possible)
      //It is the class only by default
      double min = foldLossFunct_[noCatAtts_]/trainSize_;
      for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
        foldLossFunct_[*it] = sqrt(foldLossFunct_[*it]/trainSize_);
        if(foldLossFunct_[*it] < min){
          min = foldLossFunct_[*it];
          bestattIt = it;
        }
      }
    }

    if(!onlyK_){
        for (std::vector<CategoricalAttribute>::const_iterator it = bestattIt+1; it != order_.end(); it++){
           active_[*it] = false;
          inactiveCnt_++;
        }
    }
    
    if(verbosity>=2){
      printf("Number of features selected is: %d out of %d\n",noCatAtts_-inactiveCnt_, noCatAtts_);
      if(selectiveK_ || onlyK_)
        printf("best k is: %d\n",bestK_);
    }
      
    
  }else{
    assert(pass_ == 2);
  }
  ++pass_;
}


void kdbSelective::printClassifier() {
}
