/* Gigal: An open source system for classification learning from very large data
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
#include "trainTest.h"
#include "instanceFile.h"
#include "utils.h"
#include "globals.h"
#include "instanceStreamDiscretiser.h"
#include "correlationMeasures.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#endif

void TrainTestArgs::getArgs(char*const*& argv, char*const* end) {
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (streq(argv[0]+1, "auprc")) {
      calcAUPRC_ = true;
    }
    else break;

    ++argv;
  }
}


void trainTest(learner *theLearner, InstanceStream &sourceInstanceStream, InstanceFile &instanceFile, FilterSet &filters, char * testfilename, const TrainTestArgs &args) {
  InstanceStream* instanceStream = filters.apply(&sourceInstanceStream);

  const unsigned int noClasses = instanceStream->getNoClasses();

  crosstab<InstanceCount> xtab(noClasses);
  
  long int trainTime = 0;
  long int testTime = 0;
  #ifdef __linux__
  struct rusage usage;
  #endif

  #ifdef __linux__
  getrusage(RUSAGE_SELF, &usage);
  trainTime = usage.ru_utime.tv_sec+usage.ru_stime.tv_sec;
  #endif

  theLearner->train(*instanceStream);
  
  #ifdef __linux__
  getrusage(RUSAGE_SELF, &usage);
  trainTime = ((usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)-trainTime);
  #endif

  if (testfilename != NULL) {
    instanceFile.resetSource(testfilename);
    instance inst(*instanceStream);

    if (verbosity >= 1) printf("Testing against file %s\n", testfilename);
    
    std::vector<double> classDist(noClasses);
    InstanceCount count = 0;
    unsigned int zeroOneLoss = 0;
    double squaredError = 0.0;
    double squaredErrorAll = 0.0;
    double logLoss = 0.0;
    std::vector<std::vector<float> > probs(instanceStream->getNoClasses()); //< the sequence of predicted probabilitys for each class
    std::vector<CatValue> trueClasses; //< the sequence of true classes
    
    #ifdef __linux__
    getrusage(RUSAGE_SELF, &usage);
    testTime= usage.ru_utime.tv_sec+usage.ru_stime.tv_sec;
    #endif

    while (!instanceStream->isAtEnd()) {
       if (instanceStream->advance(inst)) {
          count++;

          theLearner->classify(inst, classDist);

          const CatValue prediction = indexOfMaxVal(classDist);
          const CatValue trueClass = inst.getClass();

          if (prediction != trueClass) zeroOneLoss++;

          const double error = 1.0-classDist[trueClass];
          squaredError += error * error;
          squaredErrorAll += error * error;
          logLoss += log2(classDist[trueClass]);
          for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
            if (y != trueClass) {
              const double err = classDist[y];
              squaredErrorAll += err * err;
            }
          }
       xtab[trueClass][prediction]++;
      }
    }

    #ifdef __linux__
    getrusage(RUSAGE_SELF, &usage);
    testTime = ((usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)-testTime);
    #endif
    
    if (verbosity >= 1) {
      theLearner->printClassifier();
      printResults(xtab, *instanceStream);
      
      double MCC = calcMCC(xtab);
      printf("\nMCC:\n");
      printf("%0.4f\n", MCC);
    }

    if (args.calcAUPRC_) {
      calcAUPRC(probs, trueClasses, *instanceStream->getMetaData());
    }

    printf("\n%" ICFMT " test cases\n0-1 loss = %0.6f\nRoot mean squared error = %0.3f\n"
            "Root mean squared error all classes = %0.3f\nLogarithmic loss = %0.3f\n"
            "Training time: %ld\nClassification time: %ld\n", 
            count, zeroOneLoss/static_cast<double>(count), sqrt(squaredError/count), 
            sqrt(squaredErrorAll/(count*instanceStream->getNoClasses())), -logLoss/count,
            trainTime, testTime);
  }
}
