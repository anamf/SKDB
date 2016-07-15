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
// Utility functions
#pragma once

#include "mtrand.h"
#include "FILEtype.h"
#include "crosstab.h"
#include "instanceStream.h"

#include <time.h>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <string>
#include <algorithm>

// print an error mesage and exit.  Supports printf style format and arguments
void error(const char *fmt, ...);


template <typename T>
inline int printWidth(T val) {
  return static_cast<int>(ceil(log(static_cast<double>(val))/log(10.0)));
}

void printResults(crosstab<InstanceCount> &xtab, const InstanceStream &instanceStream);

template <typename T>
inline unsigned int max(std::vector<T> v) {
  assert(v.size() > 0);

  T maxVal = v[0];

  for (unsigned int i = 1; i < v.size(); i++) {
    if (v[i] > maxVal) maxVal = v[i];
  }

  return maxVal;
}

template <typename T>
inline unsigned int indexOfMaxVal(std::vector<T> v) {
  unsigned int maxi = 0;

  for (unsigned int i = 1; i < v.size(); i++) {
    if (v[i] > v[maxi]) maxi = i;
  }

  return maxi;
}

template <typename T>
inline void normalise(std::vector<T> &v) {
  T sum = v[0];

  for (unsigned int i = 1; i < v.size(); i++) {
    sum += v[i];
  }

  assert(sum!=0);

  for (unsigned int i = 0; i < v.size(); i++) {
    v[i] /= sum;
  }
}

template <typename T>
inline T stddev(std::vector<T> &v) {
  T m = mean(v);
  T devsq = 0;
  for (unsigned int i = 0; i < v.size(); i++) {
    const T dev = (v[i]-m);
    devsq += dev * dev;
  }

  return sqrt(devsq/(v.size()-1));
}

template <typename T>
inline T mean(std::vector<T> &v) {
  return sum(v) / static_cast<T>(v.size());
}

template <typename T>
inline T sum(const std::vector<T> &v) {
  T sum = 0;

  for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
    sum += *it;
  }

  return sum;
}

// parse an unsigned int from a string.
// template used so as to allow unsigned ints of any size
template <typename T>
void getUIntFromStr(char const *s, T &val, char const *context) {
  if (s == NULL) return;

  if (*s != '\0') {
    if (*s < '0' || *s > '9') {
      error("Encountered '%s' when expecting an unsigned integer for %s", s, context);
    }
    unsigned int v = 0;
    while (*s >= '0' && *s <= '9') {
      v *= 10;
      v += *s++ - '0';
    }
    val = v;
    if (*s != '\0') {
      error("Encountered '%s' when expecting an unsigned integer for %s", s, context);
    }
  }
}

void getUIntListFromStr(char *s, std::vector<unsigned int*> &vals, char const *context);


template <typename T>
inline void randomise(std::vector<T> &order) {
  MTRand_int32 rand(static_cast<unsigned long>(time(NULL)));

  std::vector<T> newOrder;

  while (!order.empty()) {
    const unsigned int i = rand() % order.size();
    newOrder.push_back(order[i]);
    order.erase(order.begin()+i);
  }

  order = newOrder;
}

template <typename T>
inline T max(const T v1, const T v2) {
  if (v1 > v2) return v1;
  else return v2;
}

template <typename T>
inline T min(const T v1, const T v2) {
  if (v1 < v2) return v1;
  else return v2;
}

// allocate a single object
template <typename T>
inline void safeAlloc(T *&object) {
#ifdef _DEBUG
  try {
    object = new T;
  }
  catch (std::bad_alloc) {
    error("Out of memory");
  }
#else // _DEBUG
  object = new T;
#endif // _DEBUG
}

// allocate an array of objects
template <typename T>
inline void safeAlloc(T *&object, const unsigned int size) {
#ifdef _DEBUG
  try {
    object = new T[size];
  }
  catch (std::bad_alloc) {
    error("Out of memory");
  }
#else // _DEBUG
    object = new T[size];
#endif // _DEBUG
}

// allocate an array of objects and initialise to 0
template <typename T>
inline void allocAndClear(T *&object, const unsigned int size) {
#ifdef _DEBUG
  try {
    object = new T[size];
    memset(object, 0, size * sizeof(T));
  }
  catch (std::bad_alloc) {
    error("Out of memory");
  }
#else // _DEBUG
  object = new T[size];
  memset(object, 0, size * sizeof(T));
#endif // _DEBUG
}

// compare two attributes on the values stored in an array
// appropriate comparator for descending order
template <typename T>
class IndirectCmpClass {
public:
  IndirectCmpClass(T *p) : pv(p) {
  }

  bool operator() (unsigned int a, unsigned int b) {
    return pv[a] > pv[b];
  }

private:
  const T *pv;
};

// compare two attributes on the values stored in an array, resolving draws using a second array
template <typename T, typename T2>
class IndirectCmpClass2 {
public:
  IndirectCmpClass2(std::vector<T> &p, std::vector<T2> &p2) : pv(p), pv2(p2) {
  }

  bool operator() (unsigned int a, unsigned int b) {
    return pv[a] > pv[b] || (pv[a] == pv[b] && pv2[a] > pv2[b]);
  }

private:
  const std::vector<T> &pv;
  const std::vector<T2> &pv2;
};

// compare two attributes on the values stored in an array
// appropriate comparator for ascending order
template <typename T>
class IndirectCmpClassAscending {
public:
  IndirectCmpClassAscending(T *p) : pv(p) {
  }

  bool operator() (unsigned int a, unsigned int b) {
    return pv[a] < pv[b];
  }

private:
  const T *pv;
};

// compare two attributes on the values stored in an array
// appropriate comparator for ascending order
template <typename T>
class IndirectCmpClassDescending {
public:
  IndirectCmpClassDescending(T *p) : pv(p) {
  }

  bool operator() (unsigned int a, unsigned int b) {
    return pv[a] > pv[b];
  }

private:
  const T *pv;
};


// true iff two strings are identical, case insensitive
bool streq(char const *s1, char const *s2, const bool caseSensistive = false);

// output summary of process usage
void summariseUsage();

// a vector of ptrs to objects.  On destruction, the class is responsible for deleting the object to which it points
// Used instead of a vector of auto_ptrs because we want to be able to utilise temporary pointers to the same object
// Used instead of shared_ptr because we do not need the overhead of tracking multiple pointers
template <typename T>
class ptrVec : public std::vector<T*> {
public:
  ptrVec<T>() {}

  ~ptrVec<T>() {
    typedef typename ptrVec<T>::iterator iterator;
    for (iterator i = this->begin(); i != this->end(); i++) delete *i;
    //for (int i = 0; i < this->size(); i++) delete (*this)[i];
  }

  void clear()  {
    typedef typename ptrVec<T>::iterator iterator;
    for (iterator i = this->begin(); i != this->end(); i++) delete *i;
    //for (int i = 0; i < this->size(); i++) delete (*this)[i];
    std::vector<T*>::clear();
  }
};

// a two-dimensional fixed size array whose dimensions are not known at compile time
template <typename T>
class fdarray {
public:
  fdarray<T>() : d2size_(0) {
  }

  fdarray<T>(const unsigned int d1sz, const unsigned int d2sz) : store_(d1sz*d2sz), d2size_(d2sz)
    #ifndef NDEBUG
      , d1size_(d1sz)
    #endif
  {
  }

  ~fdarray<T>() {
  }

  inline void clear() {
    store_.assign(store_.size(), 0);
  }
  
  inline void resize(const unsigned int d1sz, const unsigned int d2sz) {
    store_.resize(d1sz*d2sz);
    d2size_ = d2sz;
    #ifndef NDEBUG
      d1size_ = d1sz;
    #endif
  }

  inline void assign(const unsigned int d1sz, const unsigned int d2sz, T v)  {
    store_.assign(d1sz*d2sz, v);
    d2size_ = d2sz;
    #ifndef NDEBUG
      d1size_ = d1sz;
    #endif
  }

  inline T& ref(unsigned int a, unsigned int b) {
    #ifndef NDEBUG
      assert(a >=0 && a < d1size_);
    #endif
    assert(b >= 0 && b < d2size_);
    return store_[a*d2size_+b];
  }

  inline T* operator[](unsigned int a) {
    #ifndef NDEBUG
      assert(a >=0 && a < d1size_);
    #endif
    return &store_[a*d2size_];
  }
  
  inline unsigned int getDim(){
    return store_.size();
  }

private:
  std::vector<T> store_;
  unsigned int d2size_;
#ifndef NDEBUG
  unsigned int d1size_;
#endif
};

void print(const std::vector<double> &vals);

void print(const std::vector<float> &vals);

void print(const std::vector<long int> &vals);

void print(const std::vector<unsigned int> &vals);

/// print a string replacing ' ' with '_'
inline void print_(FILE* f, std::string &s) {
  for (std::string::iterator it = s.begin(); it != s.end(); it++) {
    if (*it == ' ' || *it == '-') fputc('_', f);
    else fputc(*it, f);
  }
}

CatValue discretise(const NumValue val, std::vector<NumValue> &cuts);

#ifdef _MSC_VER
inline double log2( const double n ) {  
    // log2(n) = log(n)/log(2)  
    return log( n ) / 0.30102999566398119521373889472449;  
}
#endif

void calcAUPRC(
        std::vector<std::vector<float> >& probs, //< the sequence of predicted probabilitys for each class
        const std::vector<CatValue>& trueClasses, //< the sequence of true classes
        InstanceStream::MetaData& metadata
        );


