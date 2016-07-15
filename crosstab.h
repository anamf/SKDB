/* Open source system for classification learning from very large data
** template class for creating and manipulating nxn arrays that are intended for crosstabulation results
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
#include <vector>
#include <sstream>
#include <stdio.h>

template <typename T>
class crosstab
{
public:
  crosstab(const unsigned int n);
  ~crosstab(void);

  inline std::vector<T> &operator[](const unsigned int i) {
    return table[i];
  }

  void print() {
    typedef typename std::vector<std::vector<T> >::const_iterator outeriterator;
    typedef typename std::vector<T>::const_iterator inneriterator;

    for (outeriterator outer = table.begin(); outer != table.end(); outer++) {
      for (inneriterator inner = outer->begin(); inner != outer->end(); inner++) {
        std::stringstream s;
        s << *inner;
        printf("%12s ", s.str().c_str());
      }
      putchar('\n');
    }
  }

private:
  std::vector<std::vector<T> > table;
};


template <typename T>
crosstab<T>::crosstab(const unsigned int n) : table(n)
{
  for (unsigned int i = 0; i < n; i++) {
    table[i].assign(n, 0);
  }
}

template <typename T>
crosstab<T>::~crosstab(void)
{
}
