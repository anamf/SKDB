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
#include "utils.h"
#include "ALGLIB_specialfunctions.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#endif

void getUIntListFromStr(char *s, std::vector<unsigned int*> &vals, char const *context) {
  if (s == NULL) return;

  for (unsigned int i = 0; i < vals.size(); i++) {
    if (*s != '\0') {
      if (*s < '0' || *s > '9') {
        error("Encountered '%s' when expecting a list of unsigned integers for %s", s, context);
      }
      unsigned int v = 0;
      while (*s >= '0' && *s <= '9') {
        v *= 10;
        v += *s++ - '0';
      }
      *vals[i] = v;
      if (*s != ',' && *s != '\0') {
        error("Encountered '%s' when expecting a list of unsigned integers for %s", s, context);
      }
      if (*s == ',') s++;
    }
  }
}


bool streq(char const *s1, char const *s2, const bool caseSensitive) {
  if (caseSensitive) {
    while (*s1 != '\0' && *s1 == *s2) {
      s1++;
      s2++;
    }
  }
  else {
    while (*s1 != '\0' && tolower(*s1) == tolower(*s2)) {
      s1++;
      s2++;
    }
  }

  return *s1 == '\0' && *s2 == '\0';
}

void printResults(crosstab<InstanceCount> &xtab, const InstanceStream &is) {
  // find the maximum value to determine how wide the output fields need to be
  InstanceCount maxval = 0;
  for (CatValue predicted = 0; predicted < is.getNoClasses(); predicted++) {
    for (CatValue y = 0; y < is.getNoClasses(); y++) {
      if (xtab[y][predicted] > maxval) maxval = xtab[y][predicted];
    }
  }

  // calculate how wide the output fields should be
  const int printwidth = max(4, printWidth(maxval));

  // print the heading line of class names
  printf("\n");
  for (CatValue y = 0; y < is.getNoClasses(); y++) {
    printf(" %*.*s", printwidth, printwidth, is.getClassName(y));
  }
  printf(" <- Actual class\n");
  
  // print the counts of instances classified as each class
  for (CatValue predicted = 0; predicted < is.getNoClasses(); predicted++) {
    for (CatValue y = 0; y < is.getNoClasses(); y++) {
      printf(" %*" ICFMT, printwidth, xtab[y][predicted]);
    }
    printf(" <- %s predicted\n", is.getClassName(predicted));
  }
}

// print an error mesage and exit.  Supports printf style format and arguments
void error(const char *fmt, ...)
{ va_list v_args;
  va_start(v_args, fmt);
  vfprintf(stderr, fmt, v_args);
  va_end(v_args);
  putc('\n', stderr);
  exit(0);
}

void print(const std::vector<double> &vals) {
  const char *sep = "";
  for (std::vector<double>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%0.4f", sep, *it);
    sep = ", ";
  }
}

void print(const std::vector<float> &vals) {
  const char *sep = "";
  for (std::vector<float>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%0.4f", sep, *it);
    sep = ", ";
  }
}

void print(const std::vector<long int> &vals) {
  const char *sep = "";
  for (std::vector<long int>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%ld", sep, *it);
    sep = ", ";
  }
}

void print(const std::vector<unsigned int> &vals) {
  const char *sep = "";
  for (std::vector<unsigned int>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%d", sep, *it);
    sep = ", ";
  }
}

void summariseUsage() {
#ifdef __linux__
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  printf("total time: %ld seconds\nmaximum memory: %ld Kb\n", usage.ru_utime.tv_sec+usage.ru_stime.tv_sec, usage.ru_maxrss);
#endif
}

CatValue discretise(const NumValue val, std::vector<NumValue> &cuts) {
  if (val == MISSINGNUM) {
    return cuts.size()+1;
  }
  else if (cuts.size() == 0) {
    return 0;
  }
  else if (val > cuts.back()) {
    return cuts.size();
  }
  else {
    unsigned int upper = cuts.size()-1;
    unsigned int lower = 0;

    while (upper > lower) {
      const unsigned int mid = lower + (upper-lower) / 2;

      if (val <= cuts[mid]) {
        upper = mid;
      }
      else {
        lower = mid+1;
      }
    }

    assert(upper == lower);
    return upper;
  }
}

/// Calculate the Area Under the Precision Recall Curve (by linear interpolation)
void calcAUPRC(
        std::vector<std::vector<float> >& probs, //< the sequence of predicted probabilities for each class
        const std::vector<CatValue>& trueClasses, //< the sequence of true classes
        InstanceStream::MetaData& metadata
        ) {
  std::vector<unsigned int> order;  //< indexes of classifications to be sorted in ascending order on predicted probability of positive class

  for (int i = 0; i < trueClasses.size(); i++) {
    order.push_back(i);
  }

  for (CatValue y = 0; y < metadata.getNoClasses(); y++) {
    InstanceCount POS = 0;
    InstanceCount NEG = 0;
    InstanceCount TP = 0;
    InstanceCount PREDPOS = 0;

    std::vector<CatValue>::const_iterator it = trueClasses.begin();

    while (it != trueClasses.end()) {
      if (*it == y) ++POS;
      else ++NEG;
      ++it;
    }

    IndirectCmpClassAscending<float> cmp(&probs[y][0]);

    std::sort(order.begin(), order.end(), cmp);

    // initially everything is classified as positive
    TP = POS;
    PREDPOS = POS + NEG;

    double lastRecall = 1.0;  // up until the lowest prob prediction, recall is 1.0.
    double lastPrecision = TP / static_cast<double>(PREDPOS); // precision when everything classified pos
    
    double auc = 0.0;

    for (unsigned int i = 0; i < trueClasses.size()-1; ++i) {
      unsigned int idx = order[i];
            
      --PREDPOS;

      if (trueClasses[idx] == y) {
        --TP;
      }

      const double recall = TP / static_cast<double>(POS);
      const double precision = TP / static_cast<double>(PREDPOS); // precision before recall next changes
      
      auc += (lastPrecision + precision) * (recall - lastRecall);

      
      lastRecall = recall;
      lastPrecision = precision;
      
    }

    printf("\nArea under precision-recall curve (COFFIN) %s: %f", metadata.getClassName(y), -auc*0.5);
  }

  putchar('\n'); 
}

