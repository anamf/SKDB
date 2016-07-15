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
 ** Please report any bugs to Shenglei Chen <tristan_chen@126.com>
 */
#pragma once

#include "discretiser.h"

#include <vector>

//this class stands for a node that represents one Attribute value and the corresponding class label
class SubInstance {
public:
	SubInstance(NumValue att, CatValue lab) :
			Attribute(att), label(lab) {
	}
	~SubInstance() {
	}
	bool operator()(const SubInstance* t1, const SubInstance* t2) {
		return t1->Attribute < t2->Attribute;
	}

	bool operator <(const SubInstance& si) const {
		return Attribute < si.Attribute;
	}

	NumValue Attribute;
	CatValue label;
};

class MDLDiscretiser: public discretiser {
public:
  MDLDiscretiser(char*const*& argv, char*const* end) {
	}
	;
  MDLDiscretiser() {
	}
	;
	~MDLDiscretiser(void) {
	}
	;

	virtual void discretise(std::vector<NumValue> &vals,
			const std::vector<CatValue> &classes,
			unsigned int noOfClasses, 
                        std::vector<NumValue> &cuts); // discretize with Attribute values vals and class label classes
protected:

	double compEntropy(unsigned int noInst,const std::vector<InstanceCount> classCount);
	//compute the cut points recursively
	//data: the vector that include the data
	//begin:the start position of the date set in the vector
	//end: the end position of the data set in the vector
	//entropy: the entropy of this data set
	//k: the number of classes in this data set
	//cuts:the cut points vector that will return
	void FindCutPoints(std::vector<SubInstance> data, unsigned int begin,
			unsigned int end, double entropy, unsigned int k,
			std::vector<NumValue> &cuts);

        unsigned int noOfClasses_;  ///< the number of classes in the data
};
